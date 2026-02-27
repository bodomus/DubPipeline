import json
import subprocess
from enum import Enum
from pathlib import Path

from dubpipeline.utils.quote_pretty_run import norm_arg
from logging import info, error


class MuxMode(str, Enum):
    REPLACE = "replace"        # заменить оригинальную аудио
    ADD = "add"                # добавить русскую (последней)
    RUS_FIRST = "rus_first"    # добавить русскую и сделать первой


def run_ffmpeg(cmd: list[str]) -> None:
    info(f"[FFMPEG] {' '.join(cmd)}")
    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if result.returncode != 0:
        error("ffmpeg failed")
        error(result.stderr)
        raise SystemExit(result.returncode)
    else:
        info("[OK] ffmpeg finished successfully")


def ffprobe_info(media: Path, ffprobe: str = "ffprobe") -> dict:
    cmd = [
        ffprobe,
        "-v", "error",
        "-print_format", "json",
        "-show_streams",
        "-show_format",
        "-show_chapters",
        norm_arg(str(media)),
    ]
    p = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if p.returncode != 0:
        raise RuntimeError(f"ffprobe failed:\n{p.stderr}")
    return json.loads(p.stdout or "{}")


def _audio_streams(info: dict) -> list[dict]:
    return [s for s in info.get("streams", []) if s.get("codec_type") == "audio"]


def _media_duration_seconds(info: dict) -> float | None:
    fmt = info.get("format") or {}
    raw = fmt.get("duration")
    if raw is None:
        return None
    try:
        value = float(raw)
    except Exception:
        return None
    if value <= 0:
        return None
    return value


def _default_audio_pos(audio_streams: list[dict]) -> int | None:
    """
    Возвращает позицию (0..N-1) default-аудио среди аудио-потоков, если есть.
    """
    for i, s in enumerate(audio_streams):
        disp = s.get("disposition") or {}
        if int(disp.get("default", 0)) == 1:
            return i
    return None


def mux_smart(
    video: Path,
    ru_audio: Path,
    out_path: Path,
    mode: MuxMode = MuxMode.ADD,
    ffmpeg: str = "ffmpeg",
    ffprobe: str = "ffprobe",
    orig_lang: str = "eng",
    orig_title: str = "Original",
    ru_lang: str = "rus",
    ru_title: str = "Russian (DubPipeline)",
    copy_subtitles: bool = True,
    set_default: bool = True,
) -> None:
    # Smart mux:
    # - replace/add/rus_first
    # - prefer copy for all audio, then fall back to re-encode if needed
    video_info = ffprobe_info(video, ffprobe=ffprobe)
    ru_info = ffprobe_info(ru_audio, ffprobe=ffprobe)

    a_streams = _audio_streams(video_info)
    video_duration = _media_duration_seconds(video_info)
    ru_duration = _media_duration_seconds(ru_info)

    # Avoid truncating output video when RU track is shorter.
    # Keep -shortest only when RU is at least as long as source video.
    use_shortest = False
    if video_duration is not None and ru_duration is not None:
        use_shortest = ru_duration >= (video_duration - 0.05)
        if not use_shortest:
            info(
                f"[MUX] RU audio is shorter ({ru_duration:.3f}s) than video "
                f"({video_duration:.3f}s). Omitting -shortest to keep full video."
            )

    cmd = [
        ffmpeg,
        "-y",
        "-i", norm_arg(str(video)),
        "-i", norm_arg(str(ru_audio)),
        "-map_metadata", "0",
        "-map_chapters", "0",
        "-map", "0:v:0",
    ]

    if copy_subtitles:
        cmd += ["-map", "0:s?", "-c:s", "copy"]

    mapped_orig_count = 0
    if mode == MuxMode.REPLACE:
        cmd += ["-map", "1:a:0"]
        out_ru_idx = 0
        out_orig_start = None
        mapped_orig_count = 0
    elif mode == MuxMode.ADD:
        cmd += ["-map", "0:a:0", "-map", "1:a:0"]
        out_ru_idx = 1
        out_orig_start = 0
        mapped_orig_count = 1
    elif mode == MuxMode.RUS_FIRST:
        cmd += ["-map", "1:a:0", "-map", "0:a:0"]
        out_ru_idx = 0
        out_orig_start = 1
        mapped_orig_count = 1
    else:
        raise ValueError(f"Unknown mode: {mode}")

    cmd += ["-c:v", "copy"]
    if use_shortest:
        cmd += ["-shortest"]

    cmd_best = cmd.copy()
    cmd_best += [
        "-c:a", "copy",
        f"-c:a:{out_ru_idx}", "aac",
    ]

    def add_dispositions(cmd_list: list[str], total_audio_out: int) -> None:
        if not set_default or total_audio_out <= 0:
            return
        for i in range(total_audio_out):
            cmd_list += [f"-disposition:a:{i}", "0"]
        if mode in (MuxMode.REPLACE, MuxMode.RUS_FIRST):
            cmd_list += [f"-disposition:a:{out_ru_idx}", "default"]
        else:
            cmd_list += ["-disposition:a:0", "default"]

    total_audio_out = 1 if mode == MuxMode.REPLACE else (mapped_orig_count + 1)

    def add_metadata(cmd_list: list[str]) -> None:
        cmd_list += [
            f"-metadata:s:a:{out_ru_idx}", f"language={ru_lang}",
            f"-metadata:s:a:{out_ru_idx}", f"title={ru_title}",
        ]
        if mapped_orig_count > 0 and out_orig_start is not None:
            for k in range(mapped_orig_count):
                out_i = out_orig_start + k
                title = orig_title if mapped_orig_count == 1 else f"{orig_title} {k+1}"
                cmd_list += [
                    f"-metadata:s:a:{out_i}", f"language={orig_lang}",
                    f"-metadata:s:a:{out_i}", f"title={title}",
                ]

    add_dispositions(cmd_best, total_audio_out)
    add_metadata(cmd_best)
    if out_path.suffix.lower() in {".mp4", ".mov", ".m4a"}:
        cmd_best += ["-movflags", "+faststart"]
    cmd_best += [norm_arg(str(out_path))]

    cmd_copy = cmd.copy()
    cmd_copy += ["-c:a", "copy"]
    add_dispositions(cmd_copy, total_audio_out)
    add_metadata(cmd_copy)
    if out_path.suffix.lower() in {".mp4", ".mov", ".m4a"}:
        cmd_copy += ["-movflags", "+faststart"]
    cmd_copy += [norm_arg(str(out_path))]

    cmd_fallback = cmd.copy()
    cmd_fallback += ["-c:a", "aac"]
    add_dispositions(cmd_fallback, total_audio_out)
    add_metadata(cmd_fallback)
    if out_path.suffix.lower() in {".mp4", ".mov", ".m4a"}:
        cmd_fallback += ["-movflags", "+faststart"]
    cmd_fallback += [norm_arg(str(out_path))]

    try:
        run_ffmpeg(cmd_copy)
    except SystemExit:
        try:
            run_ffmpeg(cmd_best)
        except Exception:
            run_ffmpeg(cmd_fallback)
    except Exception:
        try:
            run_ffmpeg(cmd_best)
        except Exception:
            run_ffmpeg(cmd_fallback)
