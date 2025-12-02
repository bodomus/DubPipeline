from __future__ import annotations

import subprocess
from pathlib import Path

from ..config import PipelineConfig


def run(cfg: PipelineConfig) -> None:
    """
    Шаг extract_audio:
    - берёт входное видео cfg.paths.input_video
    - извлекает звуковую дорожку в WAV
    - приводит к моно, нужной частоте дискретизации и кодеку
    - сохраняет в cfg.paths.audio_wav
    """

    input_video: Path = cfg.paths.input_video
    output_wav: Path = cfg.paths.audio_wav

    if not input_video.exists():
        raise FileNotFoundError(f"Входное видео не найдено: {input_video}")

    # Гарантируем, что out-папка существует
    output_wav.parent.mkdir(parents=True, exist_ok=True)

    sr = cfg.ffmpeg.sample_rate
    ch = cfg.ffmpeg.channels
    codec = cfg.ffmpeg.audio_codec

    cmd = [
        "ffmpeg",
        "-y",                  # Перезаписывать, не спрашивая
        "-i",
        str(input_video),
        "-vn",                 # без видео
        "-acodec",
        codec,
        "-ar",
        str(sr),
        "-ac",
        str(ch),
        str(output_wav),
    ]

    print("[dubpipeline] extract_audio:")
    print("  input :", input_video)
    print("  output:", output_wav)
    print("  cmd   :", " ".join(cmd))

    # Запуск ffmpeg, ошибки не глотаем
    proc = subprocess.run(cmd, capture_output=True, text=True)

    if proc.returncode != 0:
        print(proc.stdout)
        print(proc.stderr)
        raise RuntimeError(
            f"ffmpeg завершился с ошибкой (код {proc.returncode})"
        )

    print("[dubpipeline] extract_audio: OK")

