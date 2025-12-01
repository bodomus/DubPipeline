import os
import torch
import whisperx
from datetime import timedelta
from pathlib import Path

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
COMPUTE_TYPE = "float16" if DEVICE == "cuda" else "int8"
MODEL_NAME = "small.en"  # экономим VRAM

ROOT = Path(__file__).resolve().parents[1]
INPUT_AUDIO = ROOT / "tests" / "data" / "21035.wav"
OUTPUT_SRT = ROOT / "tests" / "output" / "21035.en.srt"

OUTPUT_SRT.parent.mkdir(parents=True, exist_ok=True)

print(f"Device: {DEVICE}, compute_type: {COMPUTE_TYPE}")
print(f"Input audio: {INPUT_AUDIO}")

# 1. Загружаем модель ASR
model = whisperx.load_model(MODEL_NAME, device=DEVICE, compute_type=COMPUTE_TYPE)

# 2. Транскрибируем аудио
audio = whisperx.load_audio(str(INPUT_AUDIO))
result = model.transcribe(audio, batch_size=8)

segments = result["segments"]
print(f"Segments: {len(segments)}")

# 3. Сохраняем в SRT (без диаризации — пока)
def format_timestamp(seconds: float) -> str:
    td = timedelta(seconds=seconds)
    # Приводим к формату HH:MM:SS,mmm
    total_seconds = int(td.total_seconds())
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    secs = total_seconds % 60
    millis = int((seconds - int(seconds)) * 1000)
    return f"{hours:02}:{minutes:02}:{secs:02},{millis:03}"

with open(OUTPUT_SRT, "w", encoding="utf-8") as f:
    for i, seg in enumerate(segments, start=1):
        start = format_timestamp(seg["start"])
        end = format_timestamp(seg["end"])
        text = seg["text"].strip()

        f.write(f"{i}\n")
        f.write(f"{start} --> {end}\n")
        f.write(f"{text}\n\n")

print("SRT saved to:", OUTPUT_SRT)
