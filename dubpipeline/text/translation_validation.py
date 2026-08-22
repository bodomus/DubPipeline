from __future__ import annotations

import re

_WS_RE = re.compile(r"\s+", re.UNICODE)
_WORD_RE = re.compile(r"[^\W\d_]+", re.UNICODE)


class TranslationValidationError(ValueError):
    """Raised when translated text is unsafe to treat as a target-language result."""


def normalize_for_translation_validation(text: str) -> str:
    return _WS_RE.sub(" ", (text or "").strip()).casefold()


def validate_translation_text(
    *,
    source_text: str,
    translated_text: str,
    source_lang: str,
    target_lang: str,
    segment_id: str | int | None = None,
) -> None:
    source_norm = normalize_for_translation_validation(source_text)
    translated_norm = normalize_for_translation_validation(translated_text)
    src_lang = (source_lang or "").strip().lower()
    tgt_lang = (target_lang or "").strip().lower()
    segment = "unknown" if segment_id is None else str(segment_id)

    if not source_norm:
        return
    if not translated_norm:
        raise TranslationValidationError(f"segment_id={segment} reason=empty_translation")
    if src_lang and tgt_lang and src_lang == tgt_lang:
        return

    if source_norm == translated_norm and _is_meaningful_sentence(source_norm):
        raise TranslationValidationError(f"segment_id={segment} reason=target_matches_source")

    if tgt_lang == "ru" and _looks_like_untranslated_english(translated_norm):
        raise TranslationValidationError(
            f"segment_id={segment} reason=target_language_script_mismatch"
        )


def _is_meaningful_sentence(text: str) -> bool:
    words = _WORD_RE.findall(text)
    alpha_chars = sum(1 for char in text if char.isalpha())
    return len(words) >= 3 and alpha_chars >= 14


def _looks_like_untranslated_english(text: str) -> bool:
    words = _WORD_RE.findall(text)
    alpha_chars = [char for char in text if char.isalpha()]
    if len(words) < 4 or len(alpha_chars) < 24:
        return False

    cyrillic = sum(1 for char in alpha_chars if "\u0400" <= char <= "\u04ff")
    latin = sum(1 for char in alpha_chars if "a" <= char.lower() <= "z")
    if cyrillic > 0:
        return False
    return latin / max(1, len(alpha_chars)) >= 0.75
