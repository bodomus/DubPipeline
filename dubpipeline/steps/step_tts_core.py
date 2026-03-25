from __future__ import annotations

import re
from pathlib import Path
from typing import Sequence

import numpy as np
import soundfile as sf
import torch
try:
    from TTS.api import TTS
except Exception:  # pragma: no cover
    TTS = None  # type: ignore[assignment]

from dubpipeline.config import PipelineConfig
from dubpipeline.utils.concat_wavs import concat_wavs
from dubpipeline.utils.logging import info, step, warn

_TTS_CACHE: dict[tuple[str, str], TTS] = {}
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


def _norm_text(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "")).strip()


def split_ru_text(text: str, max_len: int, breaks: Sequence[str] | None = None) -> list[str]:
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
        if cut < int(max_len * 0.5):
            cut = text.rfind(" ", 0, max_len + 1)
        if cut <= 0:
            cut = max_len
        parts.append(text[:cut].strip())
        text = text[cut:].strip()

    if text:
        parts.append(text)
    return parts


def _select_device(cfg: PipelineConfig, forced: str | None = None) -> str:
    if forced in {"cpu", "cuda"}:
        return forced
    if _truthy(getattr(cfg, "usegpu", False)) and torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _load_tts(model_name: str, device: str):
    key = (model_name, device)
    if key not in _TTS_CACHE:
        if TTS is None:
            raise RuntimeError("TTS package is not installed. Please install coqui-tts")
        step(f"Loading TTS model: {model_name} (device={device})\n")
        _TTS_CACHE[key] = TTS(model_name).to(device)
    return _TTS_CACHE[key]


def _get_xtts_model(tts: TTS):
    return getattr(getattr(tts, "synthesizer", None), "tts_model", None)


def _get_xtts_sample_rate(model) -> int:
    for path in (("config", "audio", "sample_rate"), ("config", "audio", "output_sample_rate")):
        cur = model
        for p in path:
            cur = getattr(cur, p, None)
            if cur is None:
                break
        if isinstance(cur, int):
            return cur
    return 24000


def _get_speaker_latents_cached(tts: TTS, speaker_wav: str, device: str):
    key = (str(speaker_wav), str(device))
    if key in _SPK_LATENTS_CACHE:
        return _SPK_LATENTS_CACHE[key]

    model = _get_xtts_model(tts)
    if model is None or not hasattr(model, "get_conditioning_latents"):
        return None
    if not Path(speaker_wav).exists():
        warn(f"[TTS] speaker_wav not found: {speaker_wav}\n")
        return None

    try:
        gpt_cond_latent, speaker_embedding = model.get_conditioning_latents(audio_path=[speaker_wav])
    except TypeError:
        gpt_cond_latent, speaker_embedding = model.get_conditioning_latents(audio_path=speaker_wav)

    sr = _get_xtts_sample_rate(model)
    _SPK_LATENTS_CACHE[key] = (gpt_cond_latent, speaker_embedding, sr)
    return _SPK_LATENTS_CACHE[key]


def _xtts_infer_to_wav(tts: TTS, text: str, out_wav: Path, language: str, latents) -> None:
    model = _get_xtts_model(tts)
    if model is None or not hasattr(model, "inference"):
        raise RuntimeError("XTTS direct inference API is not available")

    gpt_cond_latent, speaker_embedding, sr = latents
    out = model.inference(text=text, language=language, gpt_cond_latent=gpt_cond_latent, speaker_embedding=speaker_embedding)
    wav = out["wav"] if isinstance(out, dict) and "wav" in out else out
    if hasattr(wav, "detach"):
        wav = wav.detach()
    if hasattr(wav, "cpu"):
        wav = wav.cpu()
    if hasattr(wav, "numpy"):
        wav = wav.numpy()
    sf.write(str(out_wav), np.asarray(wav, dtype=np.float32), sr, subtype="PCM_16")


def _resolve_speaker(tts: TTS, cfg: PipelineConfig, voice: str | None, speaker_wav: str | None) -> str:
    speakers = getattr(tts, "speakers", None) or []
    if not speakers:
        if speaker_wav:
            return ""
        raise RuntimeError("XTTS does not expose speakers and speaker_wav is not set")

    desired = (voice if voice is not None else getattr(cfg.tts, "voice", "")).strip()
    if desired and desired in speakers:
        return desired
    if desired:
        warn(f"Requested speaker {desired!r} not found. Using {speakers[0]!r}\n")
    return speakers[0]


def _segment_stem(seg: dict, index: int) -> str:
    seg_id = str(seg.get("id", "")).strip() or f"seg_{index + 1:04d}"
    if re.match(r"^seg_\d+$", seg_id):
        return seg_id
    if seg_id.isdigit():
        return f"seg_{int(seg_id):04d}"
    return f"seg_{index + 1:04d}"


def _segment_text(seg: dict) -> str:
    return _norm_text(str(seg.get("text") or seg.get("text_ru") or ""))


def synthesize_segments_to_wavs(
    segments: list[dict],
    cfg: PipelineConfig,
    out_dir: Path,
    *,
    voice: str | None = None,
    lang: str | None = None,
    speaker_wav: Path | None = None,
    device: str | None = None,
    plan: bool = False,
    show_progress: bool = True,
) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    language = (lang or "ru").strip() or "ru"
    planned_paths: list[Path] = [out_dir / f"{_segment_stem(seg, i)}.wav" for i, seg in enumerate(segments) if _segment_text(seg)]

    if plan:
        info(f"[TTS][plan] segments={len(planned_paths)} out_dir={out_dir}\n")
        return planned_paths

    selected_device = _select_device(cfg, device)
    tts = _load_tts(cfg.tts.model_name, selected_device)
    speaker_wav_s = str(speaker_wav) if speaker_wav else (getattr(cfg.tts, "speaker_wav", "") or "")
    default_speaker = _resolve_speaker(tts, cfg, voice=voice, speaker_wav=speaker_wav_s or None)
    latents = _get_speaker_latents_cached(tts, speaker_wav_s, selected_device) if speaker_wav_s and cfg.tts.fast_latents else None

    built: list[Path] = []
    progress = None
    if show_progress and planned_paths:
        from dubpipeline.utils.progress import SegmentProgress
        progress = SegmentProgress(total=len(planned_paths))

    for index, seg in enumerate(segments):
        text = _segment_text(seg)
        if not text:
            continue

        out_wav = out_dir / f"{_segment_stem(seg, index)}.wav"
        chunks = split_ru_text(text, max_len=cfg.tts.max_ru_chars, breaks=cfg.tts.breaks)

        if len(chunks) == 1:
            if speaker_wav_s and latents is not None:
                try:
                    _xtts_infer_to_wav(tts, text, out_wav, language, latents)
                except Exception:
                    tts.tts_to_file(text=text, file_path=str(out_wav), language=language, speaker_wav=speaker_wav_s, split_sentences=True)
            elif speaker_wav_s:
                tts.tts_to_file(text=text, file_path=str(out_wav), language=language, speaker_wav=speaker_wav_s, split_sentences=True)
            else:
                tts.tts_to_file(text=text, file_path=str(out_wav), language=language, speaker=default_speaker, split_sentences=True)
            built.append(out_wav)
            if progress is not None:
                progress.update(len(built))
            continue

        parts: list[Path] = []
        for chunk_i, chunk in enumerate(chunks):
            part = out_dir / f"{out_wav.stem}.part{chunk_i:02d}.wav"
            if speaker_wav_s:
                tts.tts_to_file(text=chunk, file_path=str(part), language=language, speaker_wav=speaker_wav_s, split_sentences=True)
            else:
                tts.tts_to_file(text=chunk, file_path=str(part), language=language, speaker=default_speaker, split_sentences=True)
            parts.append(part)

        concat_wavs(parts, out_wav, gap_ms=cfg.tts.gap_ms, subtype="PCM_16")
        for part in parts:
            part.unlink(missing_ok=True)
        built.append(out_wav)
        if progress is not None:
            progress.update(len(built))

    if progress is not None:
        progress.finish()
    return built
