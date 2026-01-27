import json
import os
import re
import time
import hashlib
import sqlite3
import unicodedata
from pathlib import Path
from typing import Dict, List, Tuple, Optional

from dubpipeline.utils.logging import info, step, warn, error, debug
from dubpipeline.config import PipelineConfig
from dubpipeline.consts import Const

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
#   DUBPIPELINE_TRANSLATE_SENT_FALLBACK = "1" | "0"  (default: 1)
# IMPORTANT:
# - In your project you often use cfg.usegpu (without underscore). We detect BOTH: use_gpu and usegpu.
# - ArgosTranslate is CPU-only (no CUDA acceleration).
# - HF backend requires torch + transformers and will download the model once (cached by HF).

_WS_RE = re.compile(r"\s+", re.UNICODE)

def _find_weird_chars(s: str):
    """Find control/format/line-separator unicode chars that can break translation."""
    out = []
    for pos, ch in enumerate(s or ""):
        cat = unicodedata.category(ch)
        # Cc = control, Cf = format (zero-width), Zl/Zp = line/paragraph separators
        if cat in ("Cc", "Cf", "Zl", "Zp"):
            out.append((pos, f"U+{ord(ch):04X}", cat, unicodedata.name(ch, "")))
    return out


_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+", re.UNICODE)


def _env_flag(name: str, default: bool) -> bool:
    """Parse environment variable flags like 1/0, true/false, yes/no."""
    raw = os.getenv(name, None)
    if raw is None:
        return default
    v = str(raw).strip().lower()
    if v in {"1", "true", "yes", "y", "on"}:
        return True
    if v in {"0", "false", "no", "n", "off", ""}:
        return False
    return default


def _sent_count(text: str) -> int:
    t = _normalize_text(text)
    if not t:
        return 0
    parts = [p for p in _SENT_SPLIT.split(t) if p.strip()]
    return len(parts) if parts else 1

def _split_sentences(text: str) -> List[str]:
    t = _normalize_text(text)
    if not t:
        return []
    parts = [p.strip() for p in _SENT_SPLIT.split(t) if p.strip()]
    return parts or [t]

def _looks_truncated(src: str, ru: str) -> bool:
    """
    Heuristic: detect when translation likely stopped early.
    Typical symptom: source has 2+ sentences, but RU has fewer.
    """
    src_n = _normalize_text(src)
    ru_n = _normalize_text(ru)
    if not src_n:
        return False
    if not ru_n:
        return True

    sc = _sent_count(src_n)
    rc = _sent_count(ru_n)
    if sc >= 2 and rc < sc:
        return True

    # Safety net for longer segments: RU too short compared to EN.
    if len(src_n) >= 120 and len(ru_n) < int(0.35 * len(src_n)):
        return True

    return False

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
    forced = str(cfg.translate.backend or "").strip().lower()
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


def _translate_argos(texts: List[str], *, sent_fallback: bool) -> List[str]:
    """ArgosTranslate (CPU). No true batching; cache provides most speed-up."""
    from argostranslate import translate

    out: List[str] = []
    for src in texts:
        ru = translate.translate(src, "en", "ru")
        if sent_fallback and _looks_truncated(src, ru):
            parts = _split_sentences(src)
            if len(parts) > 1:
                ru_parts = [translate.translate(p, "en", "ru") for p in parts]
                ru = " ".join(p.strip() for p in ru_parts if p and p.strip()).strip()
        out.append(ru)
    return out


# -------------------------
# HuggingFace (GPU/CPU)
# -------------------------
# -------------------------

def _load_hf_translator(cfg: PipelineConfig, device: str):
    """
    Lazy-load tokenizer+model for HF translation.
    Default model: opus-mt-en-ru (fast and decent for EN->RU).
    """
    model_id = str(cfg.translate.hf_model).strip()

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


def _translate_hf(cfg: PipelineConfig, texts: List[str], tok, model, device: str, *, sent_fallback: bool) -> List[str]:
    import torch

    batch_size = int(cfg.translate.batch_size)
    max_new_tokens = int(cfg.translate.max_new_tokens)

    for idx, t in enumerate(texts):
        if t and "After Effects" in t:
            debug(f"[SRC {idx}] repr={t!r}")
            bad = [(pos, hex(ord(ch))) for pos, ch in enumerate(t) if ord(ch) < 32 or ord(ch) == 127]
            debug(f"[SRC {idx}] control_chars={bad}")


    # Sort by length to reduce padding -> speed
    order = sorted(range(len(texts)), key=lambda i: len(texts[i] or ""))
    rank_of_original = [0] * len(texts)
    for rank, orig_idx in enumerate(order):
        rank_of_original[orig_idx] = rank
    sorted_texts = [texts[i] for i in order]

    def _gen_raw(batch: List[str]) -> List[str]:
        inputs = tok(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        )

        # токены входа (макс длина в батче)
        in_len = int(inputs["input_ids"].shape[1])
        debug(f"[GEN] input_max_tokens={in_len}")
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

        out = tok.batch_decode(out_ids, skip_special_tokens=True)

        # токены выхода (макс длина)
        out_len = int(out_ids.shape[1])
        debug(f"[GEN] output_max_tokens={out_len}")
        debug(f"[GEN] sample_out_0={repr(out[0])[:240]}")
        return out

    def _gen(batch: List[str]) -> List[str]:
        # покажем max_new_tokens и пару примеров входа
        debug(f"[GEN] batch={len(batch)} max_new_tokens={max_new_tokens}")
        debug(f"[GEN] sample_in_0={repr(batch[0])[:240]}")

        out = _gen_raw(batch)

        # Fallback: если модель преждевременно завершила генерацию (2+ предложений -> меньше в RU),
        # переводим по предложениям и склеиваем.
        fixed: List[str] = []
        for src, ru in zip(batch, out):
            if _looks_truncated(src, ru):
                parts = _split_sentences(src)
                if len(parts) > 1:
                    debug(f"[GEN] truncated detected -> sentence fallback (parts={len(parts)})")
                    ru_parts: List[str] = []
                    for j in range(0, len(parts), batch_size):
                        ru_parts.extend(_gen_raw(parts[j:j + batch_size]))
                    ru = " ".join(p.strip() for p in ru_parts if p and p.strip()).strip()
            fixed.append(ru)
        return fixed


    tmp: List[str] = []
    for i in range(0, len(sorted_texts), batch_size):
        debug(f"{i}: len(sorted_texts):{len(sorted_texts)} batch_size:{batch_size} ")
        g = _gen(sorted_texts[i:i + batch_size])
        tmp.extend(g)

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
    cache_db_env = str(cfg.translate.cache_db or "").strip()
    cache_path = Path(cache_db_env) if cache_db_env else _default_cache_path(output_file)
    con = _open_cache(cache_path)

    # backend model id (part of cache key)
    if backend == "hf":
        model_id, tok, model = _load_hf_translator(cfg, device=device)
    else:
        model_id, tok, model = "argos-en-ru", None, None

    step(f"Using device for translation: {device}\n")
    info(f"[CACHE] {cache_path}\n")
    # Sentence-fallback (default ON). Disable with:
    #   DUBPIPELINE_TRANSLATE_SENT_FALLBACK=0
    sent_fallback = _env_flag("DUBPIPELINE_TRANSLATE_SENT_FALLBACK", default=True)
    info(f"[INFO] Sentence fallback: {'ON' if sent_fallback else 'OFF'}\n")

    # Prepare texts + keys
    texts: List[str] = []
    keys: List[str] = []

    # Enable invisible-char diagnostics via env:
    #   DUBPIPELINE_TRANSLATE_DEBUG_CHARS=1
    debug_chars = str(os.getenv("DUBPIPELINE_TRANSLATE_DEBUG_CHARS", "")).strip().lower() not in {"", "0", "false", "no"}


    for idx, seg in enumerate(segments):
        t = seg.get("text", "") or ""

        # DEBUG: detect control/format/line-separator chars that can cause truncated translations
        if debug_chars:
            weird = _find_weird_chars(t)
            if weird or ("After Effects" in t):
                debug(f"[WEIRD] seg#{idx} repr={t!r}")
                if weird:
                    debug(f"[WEIRD] seg#{idx} chars={weird}")

        texts.append(t)
        keys.append(_make_cache_key(backend, model_id, t))

    # Load cached results
    cached = _cache_get_many(con, keys)

    # Collect misses (unique by key to reduce duplicate work)
    misses: Dict[str, str] = {}  # key -> original text
    miss_indices: List[int] = []
    bad_cache = 0

    for i, (k, t_src) in enumerate(zip(keys, texts)):
        if not _normalize_text(t_src):
            continue

        cached_ru = cached.get(k)
        is_bad_cached = False
        if sent_fallback and cached_ru is not None:
            is_bad_cached = _looks_truncated(t_src, cached_ru)

        if cached_ru is None or is_bad_cached:
            if is_bad_cached:
                bad_cache += 1
            if k not in misses:
                misses[k] = t_src
            miss_indices.append(i)

    info(f"[INFO] Segments: {len(segments)}\n")
    info(f"[INFO] Cache hits: {len(cached)}\n")
    info(f"[INFO] Bad cache entries: {bad_cache}\n")
    info(f"[INFO] Need translate: {len(misses)} (unique), {len(miss_indices)} (total)\n\n")

    # Translate missing unique texts
    t_tr0 = time.perf_counter()
    if misses:
        miss_keys = list(misses.keys())
        miss_texts = [misses[k] for k in miss_keys]

        if backend == "hf":
            ru_texts = _translate_hf(cfg, miss_texts, tok=tok, model=model, device=device, sent_fallback=sent_fallback)
        else:
            ru_texts = _translate_argos(miss_texts, sent_fallback=sent_fallback)

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
        release = str(cfg.translate.release_vram).strip().lower()
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
