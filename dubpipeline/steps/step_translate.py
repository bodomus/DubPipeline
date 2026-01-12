import json
import os
import re
import time
import hashlib
import sqlite3
from pathlib import Path
from typing import Dict, List, Tuple, Optional

from dubpipeline.utils.logging import info, step, warn, error, debug
from dubpipeline.config import PipelineConfig

# =========================
# Translation step (EN -> RU)
# =========================
#
# Goals:
# 1) Respect your current "global CPU/GPU" checkbox (single flag for the whole pipeline).
# 2) Speed up translation:
#    - If GPU is enabled -> use HuggingFace seq2seq (batched, fp16 on CUDA).
#    - If GPU is disabled -> use ArgosTranslate (CPU) + persistent cache.
# 3) Add sqlite cache to avoid re-translating repeated strings across runs.
#
# Env overrides (optional):
#   DUBPIPELINE_TRANSLATE_BACKEND = "argos" | "hf"         (force backend)
#   DUBPIPELINE_HF_MODEL          = "Helsinki-NLP/opus-mt-en-ru"
#   DUBPIPELINE_TRANSLATE_BATCH   = "64"                  (HF batch size)
#   DUBPIPELINE_TRANSLATE_MAX_NEW_TOKENS = "256"
#   DUBPIPELINE_CACHE_DB          = "path/to/cache.sqlite"
#
# IMPORTANT:
# - In your project you often use cfg.usegpu (without underscore). We detect BOTH: use_gpu and usegpu.
# - ArgosTranslate is CPU-only (no CUDA acceleration).
# - HF backend requires torch + transformers and will download the model once (cached by HF).

_WS_RE = re.compile(r"\s+", re.UNICODE)

# Module-level cache to avoid re-loading HF model/tokenizer multiple times in one process run.
_HF_CACHE = {}  # (device, model_id) -> (tokenizer, model)


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
    Infers the global CPU/GPU flag from cfg.
    Supports common layouts:
      - cfg.use_gpu / cfg.usegpu
      - cfg.runtime.use_gpu / cfg.runtime.usegpu
      - cfg.device / cfg.runtime.device ("cuda"/"cpu")
    """
    # 1) Explicit flags
    for key in ("use_gpu", "usegpu", "useGpu", "useGPU"):
        v = _get_attr_path(cfg, key, None)
        if v is not None:
            return bool(v)
        v = _get_attr_path(cfg, f"runtime.{key}", None)
        if v is not None:
            return bool(v)

    # 2) Device strings
    for key in ("device", "runtime.device"):
        device = _get_attr_path(cfg, key, None)
        if isinstance(device, str):
            return device.lower().startswith(("cuda", "gpu"))

    return False


def _pick_backend(cfg: PipelineConfig) -> str:
    forced = os.getenv("DUBPIPELINE_TRANSLATE_BACKEND", "").strip().lower()
    if forced in {"argos", "hf"}:
        # If user forced HF but GPU is disabled, still allow HF on CPU.
        return forced

    if _is_gpu_enabled(cfg):
        # If CUDA isn't available at runtime, fall back to argos (CPU)
        try:
            import torch
            if torch.cuda.is_available():
                return "hf"
            warn("[WARN] GPU flag is ON, but torch.cuda.is_available() is False. Falling back to Argos.\n")
        except Exception:
            warn("[WARN] GPU flag is ON, but torch is not available. Falling back to Argos.\n")
        return "argos"

    return "argos"


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
    out: Dict[str, str] = {}
    CHUNK = 500  # SQLite max vars
    for i in range(0, len(keys), CHUNK):
        part = keys[i:i + CHUNK]
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
    """ArgosTranslate doesn't have true batching; cache gives the main speed-up."""
    from argostranslate import translate
    return [translate.translate(t, "en", "ru") for t in texts]


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

    cache_key = (device, model_id)
    if cache_key in _HF_CACHE:
        tok, model = _HF_CACHE[cache_key]
        return model_id, tok, model

    step(f"Loading HF model: {model_id}\n")

    tok = AutoTokenizer.from_pretrained(model_id)

    # Prefer safetensors to avoid torch.load CVE gate (CVE-2025-32434)
    # and to keep loads safer by default.
    # Also note: transformers now prefers dtype= over torch_dtype=.
    dtype = None
    if device.startswith("cuda"):
        dtype = torch.float16

    load_kwargs = {}
    if dtype is not None:
        load_kwargs["dtype"] = dtype

    # 1) Try safetensors first (recommended)
    try:
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_id,
            use_safetensors=True,
            **load_kwargs
        )
    except Exception as e_sft:
        # 2) Fallback to regular PyTorch weights (may require torch>=2.6 per transformers security check)
        try:
            model = AutoModelForSeq2SeqLM.from_pretrained(
                model_id,
                use_safetensors=False,
                **load_kwargs
            )
        except Exception as e_bin:
            raise RuntimeError(
                "Failed to load HF translation model.\n"
                "Most common cause right now: transformers blocks loading PyTorch .bin weights when torch<2.6 "
                "because of CVE-2025-32434.\n\n"
                "Fix options:\n"
                "  A) Upgrade torch to >=2.6 (recommended)\n"
                "  B) Install safetensors and ensure the model has .safetensors weights (we already try that first)\n\n"
                f"Original errors:\n- safetensors load: {e_sft}\n- bin load: {e_bin}\n"
            )
    model.eval()
    model.to(device)

    # Optional minor speed-ups on newer PyTorch (safe no-ops otherwise)
    try:
        if device.startswith("cuda"):
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.set_float32_matmul_precision("high")
    except Exception:
        pass

    _HF_CACHE[cache_key] = (tok, model)
    return model_id, tok, model


def _translate_hf(texts: List[str], tok, model, device: str) -> List[str]:
    import torch

    batch_size = int(os.getenv("DUBPIPELINE_TRANSLATE_BATCH", "64"))
    max_new_tokens = int(os.getenv("DUBPIPELINE_TRANSLATE_MAX_NEW_TOKENS", "256"))

    # Sort by length to reduce padding -> speed
    order = sorted(range(len(texts)), key=lambda i: len(texts[i] or ""))
    rank_of_original = [0] * len(texts)
    for rank, orig_idx in enumerate(order):
        rank_of_original[orig_idx] = rank
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

        if device.startswith("cuda"):
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
                out_ids = model.generate(
                    **inputs,
                    num_beams=1,
                    do_sample=False,
                    max_new_tokens=max_new_tokens
                )
        else:
            with torch.inference_mode():
                out_ids = model.generate(
                    **inputs,
                    num_beams=1,
                    do_sample=False,
                    max_new_tokens=max_new_tokens
                )

        return tok.batch_decode(out_ids, skip_special_tokens=True)

    tmp: List[str] = []
    for i in range(0, len(sorted_texts), batch_size):
        tmp.extend(_gen(sorted_texts[i:i + batch_size]))

    # Restore original order
    out = [""] * len(texts)
    for orig_idx in range(len(texts)):
        out[orig_idx] = tmp[rank_of_original[orig_idx]]
    return out


# -------------------------
# Main logic
# -------------------------

def translate_segments(cfg: PipelineConfig, input_file: str, output_file: str, backend: str):
    t0 = time.perf_counter()

    with open(input_file, "r", encoding="utf-8") as f:
        segments = json.load(f)

    # Decide translation device:
    # - HF uses CUDA only if GPU flag is enabled AND torch.cuda.is_available() is True.
    # - Argos is always CPU.
    device = "cpu"
    if backend == "hf":
        try:
            import torch
            if _is_gpu_enabled(cfg) and torch.cuda.is_available():
                device = "cuda"
        except Exception:
            device = "cpu"

    # Cache DB path
    cache_db_env = os.getenv("DUBPIPELINE_CACHE_DB", "").strip()
    cache_path = Path(cache_db_env) if cache_db_env else _default_cache_path(output_file)
    con = _open_cache(cache_path)

    # backend model id (part of cache key)
    if backend == "hf":
        model_id, tok, model = _load_hf_translator(device=device)
    else:
        model_id, tok, model = "argos-en-ru", None, None

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
    t_tr0 = time.perf_counter()
    if misses:
        miss_keys = list(misses.keys())
        miss_texts = [misses[k] for k in miss_keys]

        if backend == "hf":
            ru_texts = _translate_hf(miss_texts, tok=tok, model=model, device=device)
        else:
            ru_texts = _translate_argos(miss_texts)

        new_items = list(zip(miss_keys, ru_texts))
        _cache_put_many(con, new_items)
        cached.update({k: v for k, v in new_items})
    t_tr1 = time.perf_counter()

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

    t1 = time.perf_counter()
    info(f"\n[OK] Translated {len(translated)} segments.\n")
    info(f"[SAVE] {output_file}\n")
    info(f"[TIME] translate_core={t_tr1 - t_tr0:.2f}s, total_step={t1 - t0:.2f}s\n")
    # Optional: release VRAM after HF translate so XTTS/align can use more memory.
    # Enabled by default when using HF+CUDA. Disable with:
    #   set DUBPIPELINE_TRANSLATE_RELEASE_VRAM=0
    if backend == "hf" and device.startswith("cuda"):
        release = os.getenv("DUBPIPELINE_TRANSLATE_RELEASE_VRAM", "1").strip().lower()
        if release not in {"0", "false", "no", "off"}:
            try:
                import gc
                import torch
                # Drop from module cache first (otherwise the model remains referenced).
                try:
                    _HF_CACHE.pop((device, model_id), None)
                except Exception:
                    pass
                try:
                    model.to("cpu")
                except Exception:
                    pass
                try:
                    del model
                except Exception:
                    pass
                torch.cuda.empty_cache()
                gc.collect()
                info("[VRAM] Released CUDA cache after translate.\n")
            except Exception as e:
                warn(f"[VRAM] Failed to release CUDA cache: {e}\n")
