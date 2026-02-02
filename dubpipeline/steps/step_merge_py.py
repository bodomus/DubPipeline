import argparse
import subprocess
from pathlib import Path

from dubpipeline.config import PipelineConfig
from dubpipeline.utils.audio_process import MuxMode, mux_smart, run_ffmpeg
from dubpipeline.utils.logging import info, step, warn, error, debug
from dubpipeline.utils.quote_pretty_run import norm_arg
from dubpipeline.consts import Const

def mux_replace(
    video: Path,
    audio: Path,
    out_path: Path,
    *,
    ffmpeg: str = "ffmpeg",
    audio_codec: str = "aac",
    audio_bitrate: str = "192k",
    ru_lang: str = "rus",
    ru_title: str = "Russian_Dub",
) -> None:
    """
    Заменить оригинальную аудиодорожку на русскую.
    Выход обычно MP4.
    """
    cmd = [
        ffmpeg,
        "-y",
        "-i",
        norm_arg(str(video)),
        "-i",
        norm_arg(str(audio)),
        "-map", "0:v:0",
        "-map", "1:a:0",
        "-c:v", "copy",
        "-c:a", audio_codec,
        "-b:a", audio_bitrate,
        "-shortest",
        "-metadata:s:a:0", f"language={ru_lang}",
        "-metadata:s:a:0", f"title={ru_title}",
        norm_arg(str(out_path)),
    ]
    run_ffmpeg(cmd)


def mux_add(
    video: Path,
    audio: Path,
    out_path: Path,
    *,
    ffmpeg: str = "ffmpeg",
    audio_codec: str = "aac",
    audio_bitrate: str = "192k",
    orig_lang: str = "eng",
    orig_title: str = "Original",
    ru_lang: str = "rus",
    ru_title: str = "Russian_Dub",
) -> None:
    """
    Добавить русскую дорожку, оставив оригинальную.
    Лучше использовать контейнер MKV (но можно и MP4, если не страшно).
    """
    cmd = [
        ffmpeg,
        "-y",
        "-i",
        norm_arg(str(video)),
        "-i",
        norm_arg(str(audio)),
        "-map", "0:v:0",
        "-map", "0:a:0",
        "-map", "1:a:0",
        "-c:v", "copy",
        "-c:a", audio_codec,
        "-b:a", audio_bitrate,
        "-shortest",
        # метаданные для дорожек
        "-metadata:s:a:0", f"language={orig_lang}",
        "-metadata:s:a:0", f"title={orig_title}",
        "-metadata:s:a:1", f"language={ru_lang}",
        "-metadata:s:a:1", f"title={ru_title}",
        norm_arg(str(out_path)),
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
    Const.bind(cfg)
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

    print(f"Video: {video}\n")
    print(f"Audio (RU full): {audio}\n")
    print(f"Output: {out_path}\n")
    print(f"Mode: {cfg.mode}\n")

    if cfg.mode == "Замена":
        mux_smart(video, audio, out_path, mode=MuxMode.REPLACE)
        # mux_replace(
        #     video,
        #     audio,
        #     out_path,
        #     ffmpeg=cfg.mux.ffmpeg_bin,
        #     audio_codec=cfg.mux.audio_codec,
        #     audio_bitrate=cfg.mux.audio_bitrate,
        #     ru_lang=cfg.mux.ru_lang,
        #     ru_title=cfg.mux.ru_track_title,
        # )
    elif cfg.mode == "Добавление":
        mux_smart(video, audio, out_path, mode=MuxMode.ADD)
        # mux_add(
        #     video,
        #     audio,
        #     out_path,
        #     ffmpeg=cfg.mux.ffmpeg_bin,
        #     audio_codec=cfg.mux.audio_codec,
        #     audio_bitrate=cfg.mux.audio_bitrate,
        #     orig_lang=(cfg.mux.orig_lang or cfg.languages.src or "eng"),
        #     orig_title=cfg.mux.orig_track_title,
        #     ru_lang=cfg.mux.ru_lang,
        #     ru_title=cfg.mux.ru_track_title,
        # )
    elif cfg.mode == "Изменить порядок":
        mux_smart(video, audio, out_path, mode=MuxMode.RUS_FIRST)



