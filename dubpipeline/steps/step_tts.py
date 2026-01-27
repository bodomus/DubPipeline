from __future__ import annotations

import json
import re
from time import perf_counter
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

import torch
import numpy as np
import soundfile as sf
from TTS.api import TTS

from dubpipeline.config import PipelineConfig
from dubpipeline.utils.concat_wavs import concat_wavs
from dubpipeline.consts import Const
from dubpipeline.utils.logging import info, step, warn, error

# ============================
# Coqui TTS / XTTS settings are now in pipeline.yaml (tts.*)
# ============================

# Пауза тишиной между чанками при склейке (мс).

# Разделители, по которым стараемся резать "красиво".


# ============================
# Small utilities
# ============================
_TTS_CACHE: dict[tuple[str, str], TTS] = {}

# Speaker latents cache for XTTS when using speaker_wav (reference voice audio).
# Key: (speaker_wav_path, device) -> (gpt_cond_latent, speaker_embedding, sample_rate)
_SPK_LATENTS_CACHE: dict[tuple[str, str], tuple[object, object, int]] = {}


def _truthy(v: object) -> bool:
    if isinstance(v, bool):
        return v
    if v is None:
        return False
    if isinstance(v, (int, float)):
        return v != 0
    if isinstance(v, str):
        return v.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(v)


def _select_device(cfg: PipelineConfig) -> str:
    # cfg.usegpu у вас из yaml обычно bool, но поддержим и строковые варианты.
    if _truthy(getattr(cfg, "usegpu", False)) and torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _load_tts(model_name: str, device: str) -> TTS:
    key = (model_name, device)
    if key not in _TTS_CACHE:
        step(f"Loading TTS model: {model_name} (device={device})\n")
        _TTS_CACHE[key] = TTS(model_name).to(device)
    return _TTS_CACHE[key]


def _norm_text(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "")).strip()


def split_ru_text(text: str, max_len: int, breaks: Sequence[str] | None = None) -> list[str]:
    """Режем текст на куски до max_len: пунктуация -> пробел -> жёстко."""
    text = _norm_text(text)
    if breaks is None:
        breaks = [". ", "! ", "? ", "; ", ": ", " — ", ", "]
    if len(text) <= max_len:
        return [text]

    parts: list[str] = []
    while len(text) > max_len:
        cut = -1
        for b in breaks:
            i = text.rfind(b, 0, max_len + 1)
            if i > cut:
                cut = i + len(b) - 1

        # если по пунктуации получилось слишком рано — режем по пробелу
        if cut < int(max_len * 0.5):
            cut = text.rfind(" ", 0, max_len + 1)

        # крайний случай — режем жёстко
        if cut <= 0:
            cut = max_len

        parts.append(text[:cut].strip())
        text = text[cut:].strip()

    if text:
        parts.append(text)

    return parts


def _resolve_language(cfg: PipelineConfig) -> str:
    lang = getattr(cfg, "languages", None)
    if isinstance(lang, str) and lang.strip():
        return lang.strip()
    return "ru"



def _resolve_speaker_wav(cfg: PipelineConfig) -> str | None:
    """Optional reference-voice WAV for XTTS."""
    tts_cfg = getattr(cfg, "tts", None)
    for key in ("speaker_wav", "speaker_wav_path", "speakerwav", "speakerWav"):
        v = getattr(tts_cfg, key, None) if tts_cfg is not None else None
        if isinstance(v, str) and v.strip():
            return v.strip()
    return None


def _get_xtts_model(tts: TTS):
    # Coqui TTS API wraps the actual model under synthesizer.tts_model
    return getattr(getattr(tts, "synthesizer", None), "tts_model", None)


def _get_xtts_sample_rate(model) -> int:
    # Best-effort sample rate discovery from model config.
    for path in (
        ("config", "audio", "sample_rate"),
        ("config", "audio", "output_sample_rate"),
    ):
        cur = model
        ok = True
        for p in path:
            cur = getattr(cur, p, None)
            if cur is None:
                ok = False
                break
        if ok and isinstance(cur, int):
            return cur
    return 24000


def _get_speaker_latents_cached(tts: TTS, speaker_wav: str, device: str):
    """Compute speaker latents once per run and reuse."""
    key = (str(speaker_wav), str(device))
    if key in _SPK_LATENTS_CACHE:
        return _SPK_LATENTS_CACHE[key]

    model = _get_xtts_model(tts)
    if model is None or not hasattr(model, "get_conditioning_latents"):
        return None

    wav_path = str(speaker_wav)
    if not Path(wav_path).exists():
        warn(f"[TTS] speaker_wav not found: {wav_path} (latents cache disabled)\n")
        return None

    step(f"[TTS] Computing speaker latents once from speaker_wav={wav_path}\n")

    try:
        gpt_cond_latent, speaker_embedding = model.get_conditioning_latents(audio_path=[wav_path])
    except TypeError:
        try:
            gpt_cond_latent, speaker_embedding = model.get_conditioning_latents(audio_path=wav_path)
        except Exception as ex:
            warn(f"[TTS] get_conditioning_latents failed; fallback to speaker_wav per call: {ex}\n")
            return None
    except Exception as ex:
        warn(f"[TTS] get_conditioning_latents failed; fallback to speaker_wav per call: {ex}\n")
        return None

    sr = _get_xtts_sample_rate(model)
    _SPK_LATENTS_CACHE[key] = (gpt_cond_latent, speaker_embedding, sr)
    return _SPK_LATENTS_CACHE[key]


def _xtts_infer_to_wav(tts: TTS, text: str, out_wav: Path, language: str, latents) -> None:
    """Fast-path XTTS inference using cached latents. Raises on unsupported API."""
    model = _get_xtts_model(tts)
    if model is None or not hasattr(model, "inference"):
        raise RuntimeError("XTTS direct inference API is not available")

    gpt_cond_latent, speaker_embedding, sr = latents

    try:
        out = model.inference(
            text=text,
            language=language,
            gpt_cond_latent=gpt_cond_latent,
            speaker_embedding=speaker_embedding,
        )
    except TypeError:
        out = model.inference(text, language, gpt_cond_latent, speaker_embedding)

    wav = out["wav"] if isinstance(out, dict) and "wav" in out else out

    if hasattr(wav, "detach"):
        wav = wav.detach()
    if hasattr(wav, "cpu"):
        wav = wav.cpu()
    if hasattr(wav, "numpy"):
        wav = wav.numpy()

    wav = np.asarray(wav, dtype=np.float32)
    out_wav.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out_wav), wav, sr, subtype="PCM_16")

def _resolve_speaker(tts: TTS, cfg: PipelineConfig) -> str:
    speakers = getattr(tts, "speakers", None) or []
    speaker_wav = _resolve_speaker_wav(cfg)

    # If XTTS doesn't expose speaker list, allow speaker_wav mode.
    if not speakers:
        if speaker_wav:
            warn("[TTS] XTTS did not provide speakers list; using speaker_wav mode.\n")
            return ""
        raise RuntimeError(
            "XTTS не возвращает список speakers. "
            "Нужно либо обновить coqui-tts, либо перейти на speaker_wav."
        )

    desired = (getattr(getattr(cfg, "tts", None), "voice", "") or "").strip()
    if desired:
        if desired in speakers:
            return desired
        warn(f"Requested speaker {desired!r} not found. Using speakers[0]={speakers[0]!r}\n")

    return speakers[0]


def _cleanup_files(paths: Iterable[Path]) -> None:
    for p in paths:
        try:
            p.unlink(missing_ok=True)
        except TypeError:
            try:
                p.unlink()
            except FileNotFoundError:
                pass


# ============================
# Public API
# ============================

def get_voices(cfg: Optional[PipelineConfig] = None) -> Sequence[str] | None:
    """Список доступных speakers (если модель отдаёт).

    В GUI список голосов может запрашиваться ДО запуска пайплайна,
    поэтому здесь нельзя требовать Const.bind(cfg).

    Логика:
      - если cfg передан: используем его (и корректно выбираем device)
      - если cfg не передан: пробуем загрузить дефолтный pipeline.yaml, чтобы учесть ENV overrides
      - если и это не получилось: используем безопасный дефолт модели
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_name = None

    if cfg is not None:
        device = _select_device(cfg)
        model_name = getattr(getattr(cfg, "tts", None), "model_name", None)

    if not model_name:
        try:
            # Берём дефолтный template yaml (dubpipeline/video.pipeline.yaml) + ENV overrides
            from dubpipeline.config import load_pipeline_config_ex, pipeline_path as _pipeline_path
            _cfg = load_pipeline_config_ex(_pipeline_path)
            model_name = _cfg.tts.model_name
            device = _select_device(_cfg)
        except Exception:
            model_name = "tts_models/multilingual/multi-dataset/xtts_v2"

    tts = _load_tts(str(model_name), device)
    return getattr(tts, "speakers", None)


# Backward compat (если где-то уже зовётся старое имя)
def get_voices_compat():
    return get_voices(None)

# Keep old name for backward compatibility
getVoices = get_voices_compat  # noqa: N802


def run(cfg: PipelineConfig) -> None:
    Const.bind(cfg)
    """Генерация русской озвучки по segments_ru_json -> seg_XXXX.wav."""

    device = _select_device(cfg)
    language = _resolve_language(cfg)

    segments_path = Path(getattr(cfg.paths, "segments_ru_file"))
    out_dir = Path(getattr(cfg.paths, "segments_path"))
    out_dir.mkdir(parents=True, exist_ok=True)

    rebuild = _truthy(getattr(cfg, "rebuild", False))

    # Metrics
    metrics = []
    t_step0 = perf_counter()
    tts_synth_total = 0.0
    tts_concat_total = 0.0
    fast_latents_used = 0
    fallback_used = 0

    info(f"Using device: {device}\n")
    info(f"TTS language: {language}\n")
    info(f"Rebuild: {rebuild}\n")

    tts = _load_tts(Const.tts_model_name(), device)

    speakers = getattr(tts, "speakers", None)
    languages = getattr(tts, "languages", None)
    info(f"Available speakers: {speakers}\n")
    info(f"Available languages: {languages}\n")

    default_speaker = _resolve_speaker(tts, cfg)
    info(f"Using speaker: {default_speaker!r}\n")
    speaker_wav = _resolve_speaker_wav(cfg)
    latents = None
    use_fast_latents = Const.tts_fast_latents()
    if speaker_wav:
        info(f"Using speaker_wav: {speaker_wav}\n")
        if use_fast_latents:
            latents = _get_speaker_latents_cached(tts, speaker_wav=speaker_wav, device=device)
            info("[TTS] speaker latents caching: " + ("ON" if latents is not None else "OFF") + "\n")
        else:
            info("[TTS] speaker latents caching: DISABLED via config tts.fast_latents\n")
    else:
        info("[TTS] speaker_wav not set; speaker latents caching not applicable\n")

    if not segments_path.exists():
        raise FileNotFoundError(f"Segments file not found: {segments_path}")

    info(f"Loading segments from {segments_path}\n")
    with segments_path.open("r", encoding="utf-8") as f:
        segments = json.load(f)

    # Сортируем, но не падаем если 'start' отсутствует
    segments = sorted(segments, key=lambda s: float(s.get("start", 0.0)))

    ok = failed = skipped = 0

    for seg in segments:
        seg_id = seg.get("id")
        if seg_id is None:
            warn("Segment without 'id', skipping\n")
            skipped += 1
            continue

        text_ru = _norm_text(seg.get("text_ru") or "")
        if text_ru:
            out_wav = out_dir / f"seg_{int(seg_id):04d}.wav"

            if out_wav.exists() and not rebuild:
                warn(f"[SKIP] {out_wav} already exists\n")
                skipped += 1
                metrics.append({"seg_id": seg_id, "status": "skip_exists"})
                continue

            # если rebuild=True — перезаписываем
            if out_wav.exists() and rebuild:
                try:
                    out_wav.unlink()
                except Exception:
                    pass

            start = float(seg.get("start", 0.0))
            end = float(seg.get("end", 0.0))

            info(f"[TTS] id={seg_id}  {start:.2f}s–{end:.2f}s")
            info(f"      RU: {text_ru}\n")
            seg_t0 = perf_counter()
            seg_synth_sec = 0.0
            seg_concat_sec = 0.0
            try:
                chunks = split_ru_text(text_ru, max_len=Const.tts_max_ru_chars(), breaks=Const.tts_breaks())

                # Попытка ускорения: если наш чанкер разбил текст на несколько кусочков,
                # пробуем один вызов tts_to_file на весь сегмент (до лимита по символам).
                # Если модель/библиотека не переваривает длину — откатываемся на чанкинг.
                direct_on = Const.tts_try_single_call()
                direct_max_chars = Const.tts_try_single_call_max_chars()
                used_direct = False
                if direct_on and len(chunks) > 1 and len(text_ru) <= direct_max_chars:
                    t_direct0 = perf_counter()
                    try:
                        if speaker_wav:
                            tts.tts_to_file(
                                text=text_ru,
                                file_path=str(out_wav),
                                language=language,
                                speaker_wav=speaker_wav,
                                split_sentences=True,
                            )
                        else:
                            tts.tts_to_file(
                                text=text_ru,
                                file_path=str(out_wav),
                                language=language,
                                speaker=default_speaker,
                                split_sentences=True,
                            )
                        t_direct1 = perf_counter()
                        dt = (t_direct1 - t_direct0)
                        seg_synth_sec += dt
                        tts_synth_total += dt
                        used_direct = True

                    except Exception as ex:
                        warn(f"[TTS] Single-call attempt failed (len={len(text_ru)}), falling back to chunking: {ex}\n")
                
                if (not used_direct) and len(chunks) == 1:
                    t_synth0 = perf_counter()
                    used_fast = False

                    if speaker_wav and latents is not None:
                        try:
                            _xtts_infer_to_wav(tts, text_ru, out_wav, language, latents)
                            used_fast = True
                            fast_latents_used += 1
                        except Exception as ex:
                            fallback_used += 1
                            warn(f"[TTS] Fast-path failed, falling back to tts_to_file: {ex}\n")

                    if not used_fast:
                        if speaker_wav:
                            tts.tts_to_file(
                                text=text_ru,
                                file_path=str(out_wav),
                                language=language,
                                speaker_wav=speaker_wav,
                                split_sentences=True,
                            )
                        else:
                            tts.tts_to_file(
                                text=text_ru,
                                file_path=str(out_wav),
                                language=language,
                                speaker=default_speaker,
                                split_sentences=True,
                            )

                    t_synth1 = perf_counter()
                    dt = (t_synth1 - t_synth0)
                    seg_synth_sec += dt
                    tts_synth_total += dt
                elif not used_direct:
                    parts: List[Path] = []

                    # чистим возможные старые parts этого сегмента
                    old_parts = sorted(out_dir.glob(f"seg_{int(seg_id):04d}.part*.wav"))
                    if old_parts:
                        _cleanup_files(old_parts)

                    t_chunk0 = perf_counter()
                    for i, ch in enumerate(chunks):
                        part = out_dir / f"seg_{int(seg_id):04d}.part{i:02d}.wav"
                        if speaker_wav:
                            tts.tts_to_file(
                                text=ch,
                                file_path=str(part),
                                language=language,
                                speaker_wav=speaker_wav,
                                split_sentences=True,
                            )
                        else:
                            tts.tts_to_file(
                                text=ch,
                                file_path=str(part),
                                language=language,
                                speaker=default_speaker,
                                split_sentences=True,
                            )
                        parts.append(part)
                    t_chunk1 = perf_counter()
                    dt = (t_chunk1 - t_chunk0)
                    seg_synth_sec += dt
                    tts_synth_total += dt

                    t_cat0 = perf_counter()
                    concat_wavs(parts, out_wav, gap_ms=Const.tts_gap_ms(), subtype="PCM_16")
                    t_cat1 = perf_counter()
                    dtc = (t_cat1 - t_cat0)
                    seg_concat_sec += dtc
                    tts_concat_total += dtc
                    _cleanup_files(parts)

                ok += 1
                seg_t1 = perf_counter()
                metrics.append({
                    "seg_id": seg_id,
                    "start": start,
                    "end": end,
                    "text_len": len(text_ru),
                    "chunks": (1 if used_direct else len(chunks)),
                    "direct_used": bool(used_direct),
                    "device": device,
                    "speaker": default_speaker if not speaker_wav else None,
                    "speaker_wav": speaker_wav,
                    "fast_latents": bool(speaker_wav and latents is not None),
                    "tts_synth_sec": round(seg_synth_sec, 4),
                    "tts_concat_sec": round(seg_concat_sec, 4),
                    "total_sec": round(seg_t1 - seg_t0, 4),
                })
                info(f"[TTS][TIME] seg_id={seg_id} total={(seg_t1 - seg_t0):.2f}s\n")

            except Exception as ex:
                failed += 1
                try:
                    seg_t1 = perf_counter()
                    metrics.append({"seg_id": seg_id, "status": "fail", "error": str(ex), "total_sec": round(seg_t1 - seg_t0, 4)})
                except Exception:
                    pass
                error(f"[FAIL] seg_id={seg_id}: {ex}\n")
            continue

        warn(f"Segment {seg_id} has empty 'text_ru', skipping\n")
        skipped += 1

    t_step1 = perf_counter()
    metrics_path = out_dir / "tts_metrics.json"
    try:
        payload = {
            "model": Const.tts_model_name(),
            "device": device,
            "language": language,
            "speaker": default_speaker if not speaker_wav else None,
            "speaker_wav": speaker_wav,
            "fast_latents_used": fast_latents_used,
            "fallback_used": fallback_used,
            "tts_synth_total_sec": round(tts_synth_total, 4),
            "tts_concat_total_sec": round(tts_concat_total, 4),
            "total_step_sec": round(t_step1 - t_step0, 4),
            "segments": metrics,
        }
        with metrics_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        info(f"[METRICS] Saved: {metrics_path}\n")
    except Exception as ex:
        warn(f"[METRICS] Failed to write tts_metrics.json: {ex}\n")

    info(f"[DONE] Russian TTS segments generated in: {out_dir}\n")
    info(f"Summary: ok={ok}, failed={failed}, skipped={skipped}\n")
    info(f"[TIME] step_tts total={(t_step1 - t_step0):.2f}s, synth={tts_synth_total:.2f}s, concat={tts_concat_total:.2f}s\n")