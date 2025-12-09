import argparse
import subprocess
from pathlib import Path

from dubpipeline.config import PipelineConfig


def run_ffmpeg(cmd: list[str]) -> None:
    print("[FFMPEG]", " ".join(cmd))
    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if result.returncode != 0:
        print("[ERROR] ffmpeg failed")
        print(result.stderr)
        raise SystemExit(result.returncode)
    else:
        print("[OK] ffmpeg finished successfully")


def mux_replace(video: Path, audio: Path, out_path: Path, ffmpeg: str = "ffmpeg") -> None:
    """
    Заменить оригинальную аудиодорожку на русскую.
    Выход обычно MP4.
    """
    cmd = [
        ffmpeg,
        "-y",
        "-i",
        str(video),
        "-i",
        str(audio),
        "-map", "0:v:0",
        "-map", "1:a:0",
        "-c:v", "copy",
        "-c:a", "aac",
        "-shortest",
        "-metadata:s:a:0", "language=rus",
        "-metadata:s:a:0", "title=Russian Dub",
        str(out_path),
    ]
    run_ffmpeg(cmd)


def mux_add(
    video: Path,
    audio: Path,
    out_path: Path,
    ffmpeg: str = "ffmpeg",
    orig_lang: str = "eng",
) -> None:
    """
    Добавить русскую дорожку, оставив оригинальную.
    Лучше использовать контейнер MKV (но можно и MP4, если не страшно).
    """
    cmd = [
        ffmpeg,
        "-y",
        "-i",
        str(video),
        "-i",
        str(audio),
        "-map", "0:v:0",
        "-map", "0:a:0",
        "-map", "1:a:0",
        "-c:v", "copy",
        "-c:a", "aac",
        "-shortest",
        # метаданные для дорожек
        "-metadata:s:a:0", f"language={orig_lang}",
        "-metadata:s:a:0", "title=Original",
        "-metadata:s:a:1", "language=rus",
        "-metadata:s:a:1", "title=Russian Dub",
        str(out_path),
    ]
    run_ffmpeg(cmd)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Слияние видео и русской озвучки (full WAV) с помощью ffmpeg"
    )
    p.add_argument(
        "--video",
        type=Path,
        required=True,
        help="Путь к исходному видео (mp4/mkv/...)",
    )
    p.add_argument(
        "--audio",
        type=Path,
        required=True,
        help="Путь к русской аудиодорожке (ru_full.wav)",
    )
    p.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Путь к выходному видеофайлу",
    )
    p.add_argument(
        "--mode",
        choices=["replace", "add"],
        default="replace",
        help="Режим: replace=заменить оригинальную аудиодорожку, add=добавить русскую как вторую",
    )
    p.add_argument(
        "--ffmpeg",
        type=str,
        default="ffmpeg",
        help="Имя/путь к бинарнику ffmpeg (по умолчанию 'ffmpeg' в PATH)",
    )
    p.add_argument(
        "--orig-lang",
        type=str,
        default="eng",
        help="Код языка оригинальной аудиодорожки (для метаданных, только для mode=add)",
    )
    return p.parse_args()


def run(cfg:PipelineConfig) -> None:
    video = cfg.paths.input_video
    audio = cfg.paths.audio_wav
    out_path = cfg.paths.final_video
#
    #(venv) λ python .\tools\mux_ru_audio.py ^
  #--video in\lecture_sample.mp4 ^
  #--audio out\lecture_sample.ru_full.wav ^
  #--out out\lecture_sample.ru.with_both.mkv ^
  #--mode add

    #
    #
    if not video.exists():
        raise SystemExit(f"Video file not found: {video}")
    if not audio.exists():
        raise SystemExit(f"Audio file not found: {audio}")

    #out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Video: {video}")
    print(f"[INFO] Audio (RU full): {audio}")
    print(f"[INFO] Output: {out_path}")
    print(f"[INFO] Mode: {cfg.mode}")

    if cfg.mode == "replace":
        mux_replace(video, audio, out_path, ffmpeg="eng")
    else:
        mux_add(video, audio, out_path)


