import json
import os
import re
import hashlib
import sqlite3
from pathlib import Path
from typing import Dict, List, Tuple, Optional

from dubpipeline.utils.logging import info, step, warn, error, debug
from dubpipeline.config import PipelineConfig

# =========================
# Translation step (EN -> RU) # Translation step (EN -> RU) Version 1.0
# =========================
#
# Goals:
# 1) Keep a single global CPU/GPU flag for the whole pipeline (as you asked).
# 2) Speed up translation:
#    - If GPU is enabled -> use HuggingFace seq2seq (batched, fp16 on CUDA).
#    - If GPU is disabled -> use ArgosTranslate (CPU) + cache.
# 3) Add persistent cache (sqlite) to avoid re-translating the same strings.
#
# Env overrides (optional, no config changes required):
#   DUBPIPELINE_TRANSLATE_BACKEND = "argos" | "hf"   (force backend)
#   DUBPIPELINE_HF_MODEL          = "Helsinki-NLP/opus-mt-en-ru" (default)
#   DUBPIPELINE_TRANSLATE_BATCH   = "32"            (HF batch size)
#   DUBPIPELINE_CACHE_DB          = "path/to/cache.sqlite"
#
# Notes:
# - ArgosTranslate is CPU-only (no GPU acceleration).
# - HF backend requires: torch + transformers (and will download the model).
# - Cache key includes backend + model id -> safe when you switch models.

_WS_RE = re.compile(r"\s+", re.UNICODE)


def run(cfg: PipelineConfig):
    """
    Creates a JSON file with EN segments + RU translation.
    Backend selection:
      - GPU enabled => HuggingFace (batched) by default
      - CPU only    => ArgosTranslate by default
    Can be overridden by env var DUBPIPELINE_TRANSLATE_BACKEND.
    """
    backend = _pick_backend(cfg)

    step(f"Translation backend: {backend}\n")

    if backend == "argos":
        _ensure_argos_model_installed()

    translate_segments(
        cfg=cfg,
        input_file=cfg.paths.segments_file,
        output_file=cfg.paths.segments_ru_file,
        backend=backend,
    )


# -------------------------
# Backend selection helpers
# -------------------------

def _get_attr_path(obj, path: str, default=None):
    """Safe getattr by dotted path: 'runtime.use_gpu' etc."""
    cur = obj
    for part in path.split("."):
        if cur is None or not hasattr(cur, part):
            return default
        cur = getattr(cur, part)
    return cur


def _is_gpu_enabled(cfg: PipelineConfig) -> bool:
    """
    Tries to infer the global CPU/GPU flag from cfg.
    Supports multiple possible attribute layouts without hard dependency.
    """
    # Common patterns in configs:
    # - cfg.use_gpu (bool)
    # - cfg.device ("cpu"/"cuda")
    # - cfg.runtime.use_gpu
    # - cfg.runtime.device
    use_gpu = _get_attr_path(cfg, "use_gpu", None)
    if use_gpu is None:
        use_gpu = _get_attr_path(cfg, "runtime.use_gpu", None)

    if use_gpu is not None:
        return bool(use_gpu)

    device = _get_attr_path(cfg, "device", None)
    if device is None:
        device = _get_attr_path(cfg, "runtime.device", None)

    if isinstance(device, str):
        return device.lower().startswith(("cuda", "gpu"))

    return False


def _pick_backend(cfg: PipelineConfig) -> str:
    forced = os.getenv("DUBPIPELINE_TRANSLATE_BACKEND", "").strip().lower()
    if forced in {"argos", "hf"}:
        return forced

    return "hf" if _is_gpu_enabled(cfg) else "argos"


# -------------------------
# Cache
# -------------------------

def _default_cache_path(output_file: str) -> Path:
    out = Path(output_file)
    return out.with_suffix(out.suffix + ".translate_cache.sqlite")


def _open_cache(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(str(db_path))
    con.execute("""
        CREATE TABLE IF NOT EXISTS translations (
            k TEXT PRIMARY KEY,
            v TEXT NOT NULL
        )
    """)
    return con


def _normalize_text(text: str) -> str:
    text = (text or "").strip()
    text = _WS_RE.sub(" ", text)
    return text


def _make_cache_key(backend: str, model_id: str, text: str) -> str:
    norm = _normalize_text(text)
    payload = f"{backend}|{model_id}|{norm}".encode("utf-8")
    return hashlib.sha1(payload).hexdigest()


def _cache_get_many(con: sqlite3.Connection, keys: List[str]) -> Dict[str, str]:
    if not keys:
        return {}
    # SQLite has a max vars limit; keep chunks small.
    out: Dict[str, str] = {}
    CHUNK = 500
    for i in range(0, len(keys), CHUNK):
        part = keys[i:i+CHUNK]
        q = ",".join("?" for _ in part)
        rows = con.execute(f"SELECT k, v FROM translations WHERE k IN ({q})", part).fetchall()
        out.update({k: v for k, v in rows})
    return out


def _cache_put_many(con: sqlite3.Connection, items: List[Tuple[str, str]]):
    if not items:
        return
    con.executemany("INSERT OR REPLACE INTO translations(k, v) VALUES(?, ?)", items)
    con.commit()


# -------------------------
# Argos (CPU)
# -------------------------

def _ensure_argos_model_installed():
    """Installs EN→RU package if it is not installed yet."""
    try:
        from argostranslate import package
    except Exception as e:
        raise RuntimeError(
            "ArgosTranslate is not installed, but backend=argos was selected."
        ) from e

    packages = package.get_installed_packages()
    for p in packages:
        if p.from_code == "en" and p.to_code == "ru":
            return

    step("Downloading EN→RU model (Argos)…\n")
    available = package.get_available_packages()
    for p in available:
        if p.from_code == "en" and p.to_code == "ru":
            download_path = p.download()
            package.install_from_path(download_path)
            info("[OK] Argos EN→RU model installed.\n")
            return

    raise RuntimeError("Could not download Argos EN→RU model.")


def _translate_argos(texts: List[str]) -> List[str]:
    """ArgosTranslate doesn't have true batching; we keep it simple + rely on cache."""
    from argostranslate import translate
    out = []
    for t in texts:
        out.append(translate.translate(t, "en", "ru"))
    return out


# -------------------------
# HuggingFace (GPU/CPU)
# -------------------------

def _load_hf_translator(device: str):
    """
    Lazy-load tokenizer+model for HF translation.
    Default model: opus-mt-en-ru (fast and decent for EN->RU).
    """
    model_id = os.getenv("DUBPIPELINE_HF_MODEL", "Helsinki-NLP/opus-mt-en-ru").strip()

    try:
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    except Exception as e:
        raise RuntimeError(
            "HF backend selected but torch/transformers are not available. "
            "Install: pip install torch transformers"
        ) from e

    step(f"Loading HF model: {model_id}\n")
    tok = AutoTokenizer.from_pretrained(model_id)

    # fp16 on CUDA usually speeds up and saves VRAM.
    torch_dtype = None
    if device.startswith("cuda"):
        torch_dtype = torch.float16

    model = AutoModelForSeq2SeqLM.from_pretrained(model_id, torch_dtype=torch_dtype)
    model.eval()
    model.to(device)

    return model_id, tok, model


def _translate_hf(texts: List[str], tok, model, device: str) -> List[str]:
    import torch

    batch_size = int(os.getenv("DUBPIPELINE_TRANSLATE_BATCH", "32"))
    max_new_tokens = int(os.getenv("DUBPIPELINE_TRANSLATE_MAX_NEW_TOKENS", "256"))

    results: List[str] = []
    # Sort by length to reduce padding -> speed
    order = sorted(range(len(texts)), key=lambda i: len(texts[i] or ""))
    inv = [0] * len(order)
    for rank, i in enumerate(order):
        inv[i] = rank
    sorted_texts = [texts[i] for i in order]

    def _gen(batch: List[str]) -> List[str]:
        inputs = tok(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # torch.inference_mode disables autograd; autocast helps on CUDA.
        if device.startswith("cuda"):
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
                out_ids = model.generate(**inputs, num_beams=1, do_sample=False, max_new_tokens=max_new_tokens)
        else:
            with torch.inference_mode():
                out_ids = model.generate(**inputs, num_beams=1, do_sample=False, max_new_tokens=max_new_tokens)

        return tok.batch_decode(out_ids, skip_special_tokens=True)

    tmp: List[str] = []
    for i in range(0, len(sorted_texts), batch_size):
        chunk = sorted_texts[i:i + batch_size]
        tmp.extend(_gen(chunk))

    # Restore original order
    out = [""] * len(texts)
    for orig_i, sorted_i in enumerate(inv):
        out[orig_i] = tmp[sorted_i]
    return out


# -------------------------
# Main logic
# -------------------------

def translate_segments(cfg: PipelineConfig, input_file: str, output_file: str, backend: str):
    with open(input_file, "r", encoding="utf-8") as f:
        segments = json.load(f)

    device = "cuda" if (_is_gpu_enabled(cfg) and backend == "hf") else "cpu"

    # Cache DB path
    cache_db_env = os.getenv("DUBPIPELINE_CACHE_DB", "").strip()
    cache_path = Path(cache_db_env) if cache_db_env else _default_cache_path(output_file)
    con = _open_cache(cache_path)

    # backend model id (part of cache key)
    hf_ctx = None
    if backend == "hf":
        model_id, tok, model = _load_hf_translator(device=device)
        hf_ctx = (model_id, tok, model)
    else:
        model_id = "argos-en-ru"

    step(f"Using device for translation: {device}\n")
    info(f"[CACHE] {cache_path}\n")

    # Prepare texts + keys
    texts: List[str] = []
    keys: List[str] = []
    for seg in segments:
        t = seg.get("text", "") or ""
        texts.append(t)
        keys.append(_make_cache_key(backend, model_id, t))

    # Load cached results
    cached = _cache_get_many(con, keys)

    # Collect misses (unique by key to reduce duplicate work)
    misses: Dict[str, str] = {}  # key -> original text
    miss_indices: List[int] = []
    for i, (k, t) in enumerate(zip(keys, texts)):
        if not _normalize_text(t):
            continue
        if k not in cached and k not in misses:
            misses[k] = t
        if k not in cached:
            miss_indices.append(i)

    info(f"[INFO] Segments: {len(segments)}\n")
    info(f"[INFO] Cache hits: {len(cached)}\n")
    info(f"[INFO] Need translate: {len(misses)} (unique), {len(miss_indices)} (total)\n\n")

    # Translate missing unique texts
    new_items: List[Tuple[str, str]] = []
    if misses:
        miss_keys = list(misses.keys())
        miss_texts = [misses[k] for k in miss_keys]

        if backend == "hf":
            _model_id, tok, model = hf_ctx
            ru_texts = _translate_hf(miss_texts, tok=tok, model=model, device=device)
        else:
            ru_texts = _translate_argos(miss_texts)

        for k, ru in zip(miss_keys, ru_texts):
            new_items.append((k, ru))

        _cache_put_many(con, new_items)
        cached.update({k: v for k, v in new_items})

    # Build output (preserve ordering)
    translated = []
    for idx, seg in enumerate(segments):
        text = seg.get("text", "") or ""
        k = keys[idx]
        text_ru = cached.get(k, "") if _normalize_text(text) else ""

        seg_out = {
            "id": idx,          # Stable numeric id for downstream steps
            **seg,
            "text_ru": text_ru
        }
        translated.append(seg_out)

        # Log only first N to avoid huge console spam
        if idx < 20:
            info(f"[{idx}] {text} -> {text_ru}\n")
        elif idx == 20:
            info("... (log truncated)\n")

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(translated, f, ensure_ascii=False, indent=2)

    info(f"\n[OK] Translated {len(translated)} segments.\n")
    info(f"[SAVE] {output_file}\n")
