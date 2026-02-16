import os
from datetime import timedelta
from pathlib import Path

import pytest

if os.getenv("RUN_WHISPERX_TESTS") != "1":
    pytest.skip(
        "WhisperX integration test is disabled. Set RUN_WHISPERX_TESTS=1 to enable.",
        allow_module_level=True,
    )

import torch
import whisperx


MODEL_NAME = "small.en"
ROOT = Path(__file__).resolve().parents[1]
INPUT_AUDIO = ROOT / "tests" / "data" / "21035.wav"


def format_timestamp(seconds: float) -> str:
    td = timedelta(seconds=seconds)
    total_seconds = int(td.total_seconds())
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    secs = total_seconds % 60
    millis = int((seconds - int(seconds)) * 1000)
    return f"{hours:02}:{minutes:02}:{secs:02},{millis:03}"


@pytest.mark.integration
def test_whisperx_to_srt(tmp_path: Path) -> None:
    if not INPUT_AUDIO.exists():
        pytest.skip(f"Missing audio fixture: {INPUT_AUDIO}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    compute_type = "float16" if device == "cuda" else "int8"
    output_srt = tmp_path / "21035.en.srt"

    model = whisperx.load_model(MODEL_NAME, device=device, compute_type=compute_type)
    audio = whisperx.load_audio(str(INPUT_AUDIO))
    result = model.transcribe(audio, batch_size=8)
    segments = result["segments"]

    with open(output_srt, "w", encoding="utf-8") as f:
        for i, seg in enumerate(segments, start=1):
            start = format_timestamp(seg["start"])
            end = format_timestamp(seg["end"])
            text = seg["text"].strip()
            f.write(f"{i}\n{start} --> {end}\n{text}\n\n")

    assert output_srt.exists()
    assert output_srt.read_text(encoding="utf-8").strip() != ""
