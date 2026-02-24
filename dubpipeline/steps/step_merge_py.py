import argparse
import subprocess
from pathlib import Path

from dubpipeline.config import PipelineConfig, normalize_audio_update_mode
from dubpipeline.utils.audio_process import MuxMode, mux_smart, run_ffmpeg
from dubpipeline.utils.logging import info, step, warn, error, debug
from dubpipeline.utils.quote_pretty_run import norm_arg
from dubpipeline.consts import Const
from dubpipeline.utils.atomic_replace import AtomicFileReplacer
from dubpipeline.steps import step_merge_hq

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
        choices=["replace", "add", "rus_first", "overwrite", "overwrite_reorder"],
        default="add",
        help=(
            "Режим: add=добавить русскую дорожку, "
            "overwrite/replace=перезаписать текущую, "
            "overwrite_reorder/rus_first=перезаписать и поставить русскую первой"
        ),
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


def resolve_mux_mode(mode: str) -> MuxMode:
    normalized = normalize_audio_update_mode(mode)
    if normalized == "add":
        return MuxMode.ADD
    if normalized == "overwrite":
        return MuxMode.REPLACE
    if normalized == "overwrite_reorder":
        return MuxMode.RUS_FIRST
    raise ValueError(f"Unknown mux mode: {mode}")


def run(cfg:PipelineConfig) -> None:
    if getattr(cfg, "audio_merge", None) and cfg.audio_merge.mode == "hq_ducking":
        info("[merge] using hq_ducking mode")
        step_merge_hq.run(cfg)
        return

    Const.bind(cfg)
    video = cfg.paths.input_video
    audio = cfg.paths.audio_wav
    output_cfg = getattr(cfg, "output", None)
    update_existing = bool(getattr(output_cfg, "update_existing_file", False))
    configured_mode = getattr(output_cfg, "audio_update_mode", "") or cfg.mode
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
    mux_mode = resolve_mux_mode(configured_mode)
    effective_output = video if update_existing else out_path
    print(f"Output: {effective_output}\n")
    print(f"Mode: {configured_mode}\n")
    info(f"[merge] audio update mode: {configured_mode}")

    if not update_existing:
        mux_smart(video, audio, out_path, mode=mux_mode)
        return

    replacer = AtomicFileReplacer()
    temp_out = replacer.make_temp_path(video)
    info(f"[merge] update_existing_file enabled; temp output: {temp_out}")

    try:
        mux_smart(video, audio, temp_out, mode=mux_mode)
        replacer.replace_with_temp(video, temp_out, keep_backup=False)
    except Exception:
        replacer.cleanup_temp(temp_out)
        raise
