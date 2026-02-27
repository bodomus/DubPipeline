from __future__ import annotations

from dubpipeline.steps.step_text_input import normalize_text, split_to_segments


def test_normalize_text_preserves_ru_symbols() -> None:
    raw = "  Привет,   мир!\r\n\r\nЭто  тест…  "
    out = normalize_text(raw)
    assert "Привет, мир!" in out
    assert "Это тест..." in out


def test_split_max_chars_and_order() -> None:
    text = "Первое предложение. Второе предложение длиннее. Третье предложение."
    segments = split_to_segments(text, max_chars=25)
    assert segments
    assert all(len(seg["text"]) <= 25 for seg in segments)
    rebuilt = " ".join(seg["text"] for seg in segments)
    assert rebuilt.startswith("Первое предложение")


def test_split_removes_empty_segments() -> None:
    text = "\n\n  \n\nАбзац один.\n\n\nАбзац два.\n"
    segments = split_to_segments(text, max_chars=50)
    assert len(segments) >= 1
    assert all(seg["text"].strip() for seg in segments)
