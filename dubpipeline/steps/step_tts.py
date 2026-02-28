from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import torch
try:
    from TTS.api import TTS
except Exception:  # pragma: no cover
    TTS = None  # type: ignore[assignment]

from dubpipeline.config import PipelineConfig
from dubpipeline.consts import Const
from dubpipeline.steps.step_tts_core import _load_tts, _select_device, synthesize_segments_to_wavs
from dubpipeline.utils.logging import info


@dataclass(frozen=True)
class VoiceInfo:
    id: str
    display_name: str


def get_voices(cfg: Optional[PipelineConfig] = None) -> Sequence[str] | None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_name = "tts_models/multilingual/multi-dataset/xtts_v2"
    if cfg is not None:
        device = _select_device(cfg)
        model_name = getattr(getattr(cfg, "tts", None), "model_name", model_name)
    tts = _load_tts(str(model_name), device)
    return getattr(tts, "speakers", None)


def list_voices(cfg: Optional[PipelineConfig] = None) -> list[VoiceInfo]:
    speakers = get_voices(cfg)
    if not speakers:
        return []
    return [VoiceInfo(id=str(s), display_name=str(s)) for s in speakers]


def synthesize_preview_text(
    *,
    model_name: str,
    voice_id: str,
    preview_text: str,
    out_file: Path,
    use_gpu: bool,
) -> None:
    if not preview_text.strip():
        raise ValueError("preview_text is empty")

    device = "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
    tts = _load_tts(model_name, device)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    tts.tts_to_file(text=preview_text, speaker=voice_id, language="ru", file_path=str(out_file))


def get_voices_compat():
    return get_voices(None)


getVoices = get_voices_compat  # noqa: N802


def run(cfg: PipelineConfig) -> None:
    Const.bind(cfg)

    provider = str(getattr(cfg.tts, "provider", "coqui")).strip().lower()
    if provider != "coqui":
        raise RuntimeError(f"TTS provider={provider} не поддержан в этой сборке")
    if TTS is None:
        raise RuntimeError("TTS provider=coqui выбран, но пакет TTS не установлен. Установите: pip install TTS")

    segments_path = Path(getattr(cfg.paths, "segments_ru_file"))
    out_dir = Path(getattr(cfg.paths, "segments_path"))
    if not segments_path.exists():
        raise FileNotFoundError(f"Segments file not found: {segments_path}")

    with segments_path.open("r", encoding="utf-8") as f:
        segments = json.load(f)

    segments = sorted(segments, key=lambda s: float(s.get("start", 0.0)))
    wavs = synthesize_segments_to_wavs(segments, cfg, out_dir, show_progress=False)
    info(f"[DONE] Russian TTS segments generated in: {out_dir}\n")
    info(f"Summary: ok={len(wavs)}, failed=0, skipped={max(0, len(segments) - len(wavs))}\n")
