from __future__ import annotations

import json
import re
from pathlib import Path
from typing import TYPE_CHECKING, Iterable

from dubpipeline.utils.logging import info

if TYPE_CHECKING:
    from dubpipeline.config import PipelineConfig


SUPPORTED_SUBTITLE_EXTENSIONS: tuple[str, ...] = (".srt", ".vtt", ".ass", ".ssa")
_SRT_VTT_TIMECODE_RE = re.compile(
    r"^\s*(?P<start>\d{1,2}:\d{2}(?::\d{2})?[.,]\d{1,3})\s*-->\s*"
    r"(?P<end>\d{1,2}:\d{2}(?::\d{2})?[.,]\d{1,3})(?:\s+.*)?$"
)
_ASS_STYLE_TAG_RE = re.compile(r"\{[^{}]*\}")
_HTML_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\s+", re.UNICODE)


def supported_subtitle_extensions() -> tuple[str, ...]:
    return SUPPORTED_SUBTITLE_EXTENSIONS


def missing_subtitles_error(video_path: str | Path) -> str:
    video = Path(video_path)
    exts = ", ".join(SUPPORTED_SUBTITLE_EXTENSIONS)
    return (
        "Не найден файл субтитров для выбранного медиафайла. "
        f"Ожидается файл '{video.stem}.*' в папке '{video.parent}' "
        f"с одним из расширений: {exts}."
    )


def find_external_subtitle_for_video(
    video_path: str | Path,
    *,
    extensions: Iterable[str] | None = None,
) -> Path | None:
    video = Path(video_path)
    if not video.exists():
        return None

    exts = tuple(extensions or SUPPORTED_SUBTITLE_EXTENSIONS)
    base_stem = video.stem
    for ext in exts:
        candidate = video.with_suffix(ext)
        if candidate.is_file() and candidate.stem == base_stem:
            return candidate
    return None


def _read_text_with_fallback(path: Path) -> str:
    for enc in ("utf-8-sig", "utf-8", "cp1251", "latin-1"):
        try:
            return path.read_text(encoding=enc)
        except UnicodeDecodeError:
            continue
    return path.read_text(encoding="utf-8", errors="replace")


def _normalize_text(value: str) -> str:
    cleaned = value.replace("\\N", " ").replace("\\n", " ").replace("\n", " ")
    cleaned = _HTML_TAG_RE.sub("", cleaned)
    return _WS_RE.sub(" ", cleaned).strip()


def _parse_timestamp(value: str) -> float:
    raw = value.strip().replace(",", ".")
    parts = raw.split(":")
    if len(parts) == 3:
        hours = int(parts[0])
        minutes = int(parts[1])
        seconds = float(parts[2])
    elif len(parts) == 2:
        hours = 0
        minutes = int(parts[0])
        seconds = float(parts[1])
    else:
        raise ValueError(f"Unsupported subtitle timestamp format: {value!r}")
    return hours * 3600 + minutes * 60 + seconds


def _parse_srt_segments(text: str) -> list[dict]:
    segments: list[dict] = []
    blocks = re.split(r"\n\s*\n", text.replace("\r\n", "\n"))
    for block in blocks:
        lines = [line.strip() for line in block.split("\n") if line.strip()]
        if len(lines) < 2:
            continue
        line_index = 0
        if lines[0].isdigit():
            line_index = 1
        if line_index >= len(lines):
            continue
        match = _SRT_VTT_TIMECODE_RE.match(lines[line_index])
        if not match:
            continue
        text_lines = lines[line_index + 1 :]
        text_raw = _normalize_text(" ".join(text_lines))
        if not text_raw:
            continue
        start = _parse_timestamp(match.group("start"))
        end = _parse_timestamp(match.group("end"))
        if end <= start:
            continue
        segments.append({"start": start, "end": end, "text": text_raw})
    return segments


def _parse_vtt_segments(text: str) -> list[dict]:
    lines = text.replace("\r\n", "\n").split("\n")
    segments: list[dict] = []
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line or line.upper() == "WEBVTT" or line.startswith("NOTE"):
            i += 1
            continue

        match = _SRT_VTT_TIMECODE_RE.match(line)
        if not match and i + 1 < len(lines):
            match = _SRT_VTT_TIMECODE_RE.match(lines[i + 1].strip())
            if match:
                i += 1
        if not match:
            i += 1
            continue

        i += 1
        cue_lines: list[str] = []
        while i < len(lines):
            cue_line = lines[i].strip()
            if not cue_line:
                break
            cue_lines.append(cue_line)
            i += 1
        text_raw = _normalize_text(" ".join(cue_lines))
        if text_raw:
            start = _parse_timestamp(match.group("start"))
            end = _parse_timestamp(match.group("end"))
            if end > start:
                segments.append({"start": start, "end": end, "text": text_raw})
        i += 1
    return segments


def _parse_ass_segments(text: str) -> list[dict]:
    segments: list[dict] = []
    for raw_line in text.replace("\r\n", "\n").split("\n"):
        line = raw_line.strip()
        if not line.lower().startswith("dialogue:"):
            continue
        payload = line.split(":", 1)[1].strip()
        parts = payload.split(",", 9)
        if len(parts) < 10:
            continue
        start_raw = parts[1].strip()
        end_raw = parts[2].strip()
        text_raw = _ASS_STYLE_TAG_RE.sub("", parts[9]).strip()
        text_raw = _normalize_text(text_raw)
        if not text_raw:
            continue
        start = _parse_timestamp(start_raw)
        end = _parse_timestamp(end_raw)
        if end <= start:
            continue
        segments.append({"start": start, "end": end, "text": text_raw})
    return segments


def segments_from_subtitle_file(subtitle_path: str | Path) -> list[dict]:
    path = Path(subtitle_path)
    suffix = path.suffix.lower()
    text = _read_text_with_fallback(path)

    if suffix == ".srt":
        parsed = _parse_srt_segments(text)
    elif suffix == ".vtt":
        parsed = _parse_vtt_segments(text)
    elif suffix in {".ass", ".ssa"}:
        parsed = _parse_ass_segments(text)
    else:
        raise ValueError(f"Unsupported subtitle extension: {suffix}")

    parsed.sort(key=lambda item: (float(item["start"]), float(item["end"])))
    segments: list[dict] = []
    for idx, seg in enumerate(parsed):
        segments.append(
            {
                "id": idx,
                "start": float(seg["start"]),
                "end": float(seg["end"]),
                "text": str(seg["text"]),
            }
        )
    return segments


def prepare_external_subtitles(cfg: PipelineConfig) -> Path:
    video_path = Path(cfg.paths.input_video)
    subtitle_path = find_external_subtitle_for_video(video_path)
    if subtitle_path is None:
        raise FileNotFoundError(missing_subtitles_error(video_path))

    segments = segments_from_subtitle_file(subtitle_path)
    if not segments:
        raise ValueError(
            f"Файл субтитров '{subtitle_path}' не содержит валидных реплик для обработки."
        )

    segments_path = Path(cfg.paths.segments_file)
    segments_path.parent.mkdir(parents=True, exist_ok=True)
    with segments_path.open("w", encoding="utf-8") as f:
        json.dump(segments, f, ensure_ascii=False, indent=2)

    info(f"[dubpipeline] External subtitles: {subtitle_path}")
    info(f"[dubpipeline] Segments from subtitles: {segments_path} ({len(segments)} rows)")
    return subtitle_path
