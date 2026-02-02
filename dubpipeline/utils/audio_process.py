import json
import subprocess
from enum import Enum
from logging import info, error
from pathlib import Path

from dubpipeline.utils.quote_pretty_run import norm_arg

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
    ru_title: str = "Russian_Dub",
    copy_subtitles: bool = True,
    set_default: bool = True,
) -> None:
    """
    Умный mux:
    - умеет replace/add/rus_first
    - старается НЕ перекодировать оригинальную аудио (copy), только русскую -> aac
    - при несовместимости откатывается на aac для всех аудио
    """
    info = ffprobe_info(video, ffprobe=ffprobe)
    a_streams = _audio_streams(info)
    orig_a_count = len(a_streams)
    orig_default_pos = _default_audio_pos(a_streams)

    # -------- базовый cmd --------
    cmd = [
        ffmpeg,
        "-y",
        "-i", norm_arg(str(video)),
        "-i", norm_arg(str(ru_audio)),

        # сохранить глобальные метаданные/главы
        "-map_metadata", "0",
        "-map_chapters", "0",

        # видео (как у вас: первый видеопоток)
        "-map", "0:v:0",
    ]

    # subtitles (опционально, безопасно: ?)
    if copy_subtitles:
        cmd += ["-map", "0:s?", "-c:s", "copy"]

    # -------- маппинг аудио --------
    # Важно: порядок -map определяет порядок дорожек в выходе.
    if mode == MuxMode.REPLACE:
        # только русская
        cmd += ["-map", "1:a:0"]
        out_ru_idx = 0
        out_orig_start = None

    elif mode == MuxMode.ADD:
        # сначала все оригинальные, потом русская
        cmd += ["-map", "0:a?", "-map", "1:a:0"]
        out_ru_idx = orig_a_count  # если orig_a_count=0 -> 0
        out_orig_start = 0

    elif mode == MuxMode.RUS_FIRST:
        # сначала русская, потом все оригинальные
        cmd += ["-map", "1:a:0", "-map", "0:a?"]
        out_ru_idx = 0
        out_orig_start = 1 if orig_a_count > 0 else None

    else:
        raise ValueError(f"Unknown mode: {mode}")

    # -------- кодеки --------
    cmd += ["-c:v", "copy", "-shortest"]

    # “умно”: оригинал копируем, русскую кодируем в AAC
    # (если это не получится — ниже сделаем fallback на aac для всех)
    cmd_best = cmd.copy()
    cmd_best += [
        "-c:a", "copy",
        f"-c:a:{out_ru_idx}", "aac",
    ]

    # -------- disposition default --------
    # Ставим default “по уму”:
    # - replace/rus_first: русская default
    # - add: сохраняем default оригинала (если был), иначе оставим 0-ю как default
    def add_dispositions(cmd_list: list[str], total_audio_out: int) -> None:
        if not set_default or total_audio_out <= 0:
            return

        # сначала сбросим default у всех
        for i in range(total_audio_out):
            cmd_list += [f"-disposition:a:{i}", "0"]

        if mode in (MuxMode.REPLACE, MuxMode.RUS_FIRST):
            cmd_list += [f"-disposition:a:{out_ru_idx}", "default"]
        else:  # ADD
            if orig_a_count == 0:
                cmd_list += [f"-disposition:a:{out_ru_idx}", "default"]
            else:
                # сохранить default оригинала, если найден
                keep = orig_default_pos if orig_default_pos is not None else 0
                cmd_list += [f"-disposition:a:{keep}", "default"]

    total_audio_out = 1 if mode == MuxMode.REPLACE else (orig_a_count + 1)

    # -------- metadata дорожек --------
    def add_metadata(cmd_list: list[str]) -> None:
        # русская
        cmd_list += [
            f"-metadata:s:a:{out_ru_idx}", f"language={ru_lang}",
            f"-metadata:s:a:{out_ru_idx}", f"title={ru_title}",
        ]

        # оригинальные (если есть)
        if orig_a_count > 0 and out_orig_start is not None:
            for k in range(orig_a_count):
                out_i = out_orig_start + k
                # чтобы не было одинаковых title при множестве дорожек:
                title = orig_title if orig_a_count == 1 else f"{orig_title} {k+1}"
                cmd_list += [
                    f"-metadata:s:a:{out_i}", f"language={orig_lang}",
                    f"-metadata:s:a:{out_i}", f"title={title}",
                ]

    # финализируем cmd_best
    add_dispositions(cmd_best, total_audio_out)
    add_metadata(cmd_best)
    cmd_best += [norm_arg(str(out_path))]

    # -------- fallback (aac для всех аудио) --------
    cmd_fallback = cmd.copy()
    cmd_fallback += ["-c:a", "aac"]
    add_dispositions(cmd_fallback, total_audio_out)
    add_metadata(cmd_fallback)
    cmd_fallback += [norm_arg(str(out_path))]

    # -------- запуск --------
    try:
        run_ffmpeg(cmd_best)
    except Exception:
        # если copy оригинала невозможен (например, DTS в MP4) — перекодируем всё в AAC
        run_ffmpeg(cmd_fallback)
