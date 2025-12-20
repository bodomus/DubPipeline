import json
from pathlib import Path

from argostranslate import package, translate
import torch
from TTS.api import TTS
from dubpipeline.utils.logging import info, step, warn, error, debug

from dubpipeline.config import PipelineConfig

model_name = "tts_models/multilingual/multi-dataset/xtts_v2"
device = "cuda" if torch.cuda.is_available() else "cpu"

def getVoices():
    tts = TTS(model_name).to(device)
    speakers = getattr(tts, "speakers", None)
    return speakers

def run(cfg:PipelineConfig):
    # Генерация русского аудио звука и запись его в wav файлы
    # --- НАСТРОЙКИ ---
    segments_path = cfg.paths.segments_ru_file
    out_dir = Path(cfg.paths.segments_path)

    out_dir.mkdir(parents=True, exist_ok=True)


    info(f"Using device: {device}\n")

    info(f"Loading TTS model: {model_name}\n")
    tts = TTS(model_name).to(device)

    # --- Дебаг: какие вообще есть спикеры и языки ---
    speakers = getattr(tts, "speakers", None)
    languages = getattr(tts, "languages", None)
    info("Available speakers:\n {speakers}")
    info("Available languages:\n {languages}")

    if not speakers:
        raise RuntimeError(
            "XTTS не возвращает список speakers. "
            "Нужно либо обновить coqui-tts, либо использовать speaker_wav."
        )

    default_speaker = speakers[0]
    info(f"Using default speaker: {default_speaker!r}\n")

    # --- ЗАГРУЗКА СЕГМЕНТОВ ---
    info(f"Loading segments from {segments_path}\n")
    with Path(segments_path).open("r", encoding="utf-8") as f:
        segments = json.load(f)

    segments = sorted(segments, key=lambda s: s["start"])

    # --- ГЕНЕРАЦИЯ ---
    for seg in segments:
        seg_id = seg["id"]
        text_ru = (seg.get("text_ru") or "").strip()

        if not text_ru:
            warn(f"Segment {seg_id} has empty 'text_ru', skipping\n")
            continue

        out_wav = out_dir / f"seg_{seg_id:04d}.wav"
        if out_wav.exists():
            warn(f"[SKIP] {out_wav} already exists\n")
            continue

        info(f"[TTS] id={seg_id}  {seg['start']:.2f}s–{seg['end']:.2f}s\n")
        info(f"      RU: {text_ru}\n")

        # КЛЮЧЕВАЯ ЧАСТЬ: задаём и language, и speaker
        tts.tts_to_file(
            text=text_ru,
            file_path=str(out_wav),
            language=cfg.languages,
            speaker=default_speaker,
        )

    info("[DONE] Russian TTS segments generated in:\n {out_dir}")