from __future__ import annotations

import json
import re
from pathlib import Path


def load_text(*, text: str | None, text_file: Path | None, encoding: str = "utf-8") -> str:
    if bool(text) == bool(text_file):
        raise ValueError("Exactly one of 'text' or 'text_file' must be provided")
    if text_file is not None:
        return text_file.read_text(encoding=encoding)
    return text or ""


def normalize_text(raw: str) -> str:
    normalized = (raw or "").replace("\r\n", "\n").replace("\r", "\n")
    normalized = normalized.replace("“", '"').replace("”", '"').replace("…", "...")
    normalized = re.sub(r"[ \t]+", " ", normalized)
    normalized = re.sub(r"\n{3,}", "\n\n", normalized)
    lines = [line.strip() for line in normalized.split("\n")]
    normalized = "\n".join(lines)
    return normalized.strip()


def split_to_segments(text: str, *, max_chars: int, min_chars: int = 20) -> list[dict]:
    if max_chars <= 0:
        raise ValueError("max_chars must be > 0")

    clean_text = normalize_text(text)
    if not clean_text:
        return []

    paragraphs = [p.strip() for p in clean_text.split("\n\n") if p.strip()]
    pieces: list[str] = []
    for paragraph in paragraphs:
        sentences = re.split(r"(?<=[.!?;:])\s+", paragraph)
        buffer = ""
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            candidate = f"{buffer} {sentence}".strip() if buffer else sentence
            if len(candidate) <= max_chars:
                buffer = candidate
                continue

            if buffer:
                pieces.append(buffer)
                buffer = ""

            while len(sentence) > max_chars:
                split_pos = sentence.rfind(" ", 0, max_chars)
                if split_pos <= 0:
                    split_pos = max_chars
                part = sentence[:split_pos].strip()
                if part:
                    pieces.append(part)
                sentence = sentence[split_pos:].strip()
            buffer = sentence

        if buffer:
            pieces.append(buffer)

    merged: list[str] = []
    for piece in pieces:
        if not piece:
            continue
        if merged and len(merged[-1]) < min_chars and len(merged[-1]) + 1 + len(piece) <= max_chars:
            merged[-1] = f"{merged[-1]} {piece}".strip()
        else:
            merged.append(piece)

    return [{"id": f"seg_{idx:04d}", "text": value} for idx, value in enumerate(merged, start=1) if value.strip()]


def save_segments_json(segments: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(segments, f, ensure_ascii=False, indent=2)
