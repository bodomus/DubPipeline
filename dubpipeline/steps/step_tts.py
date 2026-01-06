from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

import torch
from TTS.api import TTS

from dubpipeline.config import PipelineConfig
from dubpipeline.utils.concat_wavs import concat_wavs
from dubpipeline.utils.logging import info, step, warn, error

# ============================
# Coqui TTS / XTTS settings
# ============================
MODEL_NAME = "tts_models/multilingual/multi-dataset/xtts_v2"

# Предупреждение XTTS для ru обычно появляется после ~182 символов.
# Держим запас, чтобы не ловить truncated audio.
MAX_RU_CHARS = 170

# Пауза тишиной между чанками при склейке (мс).
GAP_MS = 80

# Разделители, по которым стараемся резать "красиво".
BREAKS = [". ", "! ", "? ", "; ", ": ", " — ", ", "]


# ============================
# Small utilities
# ============================
_TTS_CACHE: dict[tuple[str, str], TTS] = {}


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


def split_ru_text(text: str, max_len: int = MAX_RU_CHARS) -> list[str]:
    """Режем текст на куски до max_len: пунктуация -> пробел -> жёстко."""
    text = _norm_text(text)
    if len(text) <= max_len:
        return [text]

    parts: list[str] = []
    while len(text) > max_len:
        cut = -1
        for b in BREAKS:
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


def _resolve_speaker(tts: TTS, cfg: PipelineConfig) -> str:
    speakers = getattr(tts, "speakers", None) or []
    if not speakers:
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
    """Список доступных speakers (если модель отдаёт)."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if cfg is not None:
        device = _select_device(cfg)

    tts = _load_tts(MODEL_NAME, device)
    return getattr(tts, "speakers", None)


# Backward compat (если где-то уже зовётся старое имя)
def get_voices_compat():
    return get_voices(None)

# Keep old name for backward compatibility
getVoices = get_voices_compat  # noqa: N802


def run(cfg: PipelineConfig) -> None:
    """Генерация русской озвучки по segments_ru_json -> seg_XXXX.wav."""

    device = _select_device(cfg)
    language = _resolve_language(cfg)

    segments_path = Path(getattr(cfg.paths, "segments_ru_file"))
    out_dir = Path(getattr(cfg.paths, "segments_path"))
    out_dir.mkdir(parents=True, exist_ok=True)

    rebuild = _truthy(getattr(cfg, "rebuild", False))

    info(f"Using device: {device}\n")
    info(f"TTS language: {language}\n")
    info(f"Rebuild: {rebuild}\n")

    tts = _load_tts(MODEL_NAME, device)

    speakers = getattr(tts, "speakers", None)
    languages = getattr(tts, "languages", None)
    info(f"Available speakers: {speakers}\n")
    info(f"Available languages: {languages}\n")

    default_speaker = _resolve_speaker(tts, cfg)
    info(f"Using speaker: {default_speaker!r}\n")

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

            try:
                chunks = split_ru_text(text_ru, max_len=MAX_RU_CHARS)

                if len(chunks) == 1:
                    tts.tts_to_file(
                        text=text_ru,
                        file_path=str(out_wav),
                        language=language,
                        speaker=default_speaker,
                        split_sentences=True,
                    )
                else:
                    parts: List[Path] = []

                    # чистим возможные старые parts этого сегмента
                    old_parts = sorted(out_dir.glob(f"seg_{int(seg_id):04d}.part*.wav"))
                    if old_parts:
                        _cleanup_files(old_parts)

                    for i, ch in enumerate(chunks):
                        part = out_dir / f"seg_{int(seg_id):04d}.part{i:02d}.wav"
                        tts.tts_to_file(
                            text=ch,
                            file_path=str(part),
                            language=language,
                            speaker=default_speaker,
                            split_sentences=True,
                        )
                        parts.append(part)

                    concat_wavs(parts, out_wav, gap_ms=GAP_MS, subtype="PCM_16")
                    _cleanup_files(parts)

                ok += 1

            except Exception as ex:
                failed += 1
                error(f"[FAIL] seg_id={seg_id}: {ex}\n")
            continue

        warn(f"Segment {seg_id} has empty 'text_ru', skipping\n")
        skipped += 1

    info(f"[DONE] Russian TTS segments generated in: {out_dir}\n")
    info(f"Summary: ok={ok}, failed={failed}, skipped={skipped}\n")
