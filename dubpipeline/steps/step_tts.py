import json
from pathlib import Path

from argostranslate import package, translate
import torch
from TTS.api import TTS

from dubpipeline.config import PipelineConfig


def run(cfg:PipelineConfig):
    # Генерация русского аудио звука и запись его в wav файлы
    # --- НАСТРОЙКИ ---
    segments_path = cfg.paths.segments_ru_file
    out_dir = Path(cfg.paths.segments_path)

    model_name = "tts_models/multilingual/multi-dataset/xtts_v2"

    out_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Using device: {device}")

    print(f"[INFO] Loading TTS model: {model_name}")
    tts = TTS(model_name).to(device)

    # --- Дебаг: какие вообще есть спикеры и языки ---
    speakers = getattr(tts, "speakers", None)
    languages = getattr(tts, "languages", None)
    print("[INFO] Available speakers:", speakers)
    print("[INFO] Available languages:", languages)

    if not speakers:
        raise RuntimeError(
            "XTTS не возвращает список speakers. "
            "Нужно либо обновить coqui-tts, либо использовать speaker_wav."
        )

    default_speaker = speakers[0]
    print(f"[INFO] Using default speaker: {default_speaker!r}")

    # --- ЗАГРУЗКА СЕГМЕНТОВ ---
    print(f"[INFO] Loading segments from {segments_path}")
    with segments_path.open("r", encoding="utf-8") as f:
        segments = json.load(f)

    segments = sorted(segments, key=lambda s: s["start"])

    # --- ГЕНЕРАЦИЯ ---
    for seg in segments:
        seg_id = seg["id"]
        text_ru = (seg.get("text_ru") or "").strip()

        if not text_ru:
            print(f"[WARN] Segment {seg_id} has empty 'text_ru', skipping")
            continue

        out_wav = out_dir / f"seg_{seg_id:04d}.wav"
        if out_wav.exists():
            print(f"[SKIP] {out_wav} already exists")
            continue

        print(f"[TTS] id={seg_id}  {seg['start']:.2f}s–{seg['end']:.2f}s")
        print(f"      RU: {text_ru}")

        # КЛЮЧЕВАЯ ЧАСТЬ: задаём и language, и speaker
        tts.tts_to_file(
            text=text_ru,
            file_path=str(out_wav),
            language=cfg.languages,
            speaker=default_speaker,
        )

    print("[DONE] Russian TTS segments generated in:", out_dir)