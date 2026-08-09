from __future__ import annotations

import re


_DIGIT_WORDS_RU = {
    "0": "\u043d\u043e\u043b\u044c",
    "1": "\u043e\u0434\u0438\u043d",
    "2": "\u0434\u0432\u0430",
    "3": "\u0442\u0440\u0438",
    "4": "\u0447\u0435\u0442\u044b\u0440\u0435",
    "5": "\u043f\u044f\u0442\u044c",
    "6": "\u0448\u0435\u0441\u0442\u044c",
    "7": "\u0441\u0435\u043c\u044c",
    "8": "\u0432\u043e\u0441\u0435\u043c\u044c",
    "9": "\u0434\u0435\u0432\u044f\u0442\u044c",
}

_SMALL_INT_WORDS_RU = {
    0: "\u043d\u043e\u043b\u044c",
    1: "\u043e\u0434\u0438\u043d",
    2: "\u0434\u0432\u0430",
    3: "\u0442\u0440\u0438",
    4: "\u0447\u0435\u0442\u044b\u0440\u0435",
    5: "\u043f\u044f\u0442\u044c",
    6: "\u0448\u0435\u0441\u0442\u044c",
    7: "\u0441\u0435\u043c\u044c",
    8: "\u0432\u043e\u0441\u0435\u043c\u044c",
    9: "\u0434\u0435\u0432\u044f\u0442\u044c",
    10: "\u0434\u0435\u0441\u044f\u0442\u044c",
    11: "\u043e\u0434\u0438\u043d\u043d\u0430\u0434\u0446\u0430\u0442\u044c",
    12: "\u0434\u0432\u0435\u043d\u0430\u0434\u0446\u0430\u0442\u044c",
    13: "\u0442\u0440\u0438\u043d\u0430\u0434\u0446\u0430\u0442\u044c",
    14: "\u0447\u0435\u0442\u044b\u0440\u043d\u0430\u0434\u0446\u0430\u0442\u044c",
    15: "\u043f\u044f\u0442\u043d\u0430\u0434\u0446\u0430\u0442\u044c",
    16: "\u0448\u0435\u0441\u0442\u043d\u0430\u0434\u0446\u0430\u0442\u044c",
    17: "\u0441\u0435\u043c\u043d\u0430\u0434\u0446\u0430\u0442\u044c",
    18: "\u0432\u043e\u0441\u0435\u043c\u043d\u0430\u0434\u0446\u0430\u0442\u044c",
    19: "\u0434\u0435\u0432\u044f\u0442\u043d\u0430\u0434\u0446\u0430\u0442\u044c",
    20: "\u0434\u0432\u0430\u0434\u0446\u0430\u0442\u044c",
    30: "\u0442\u0440\u0438\u0434\u0446\u0430\u0442\u044c",
    40: "\u0441\u043e\u0440\u043e\u043a",
    50: "\u043f\u044f\u0442\u044c\u0434\u0435\u0441\u044f\u0442",
    60: "\u0448\u0435\u0441\u0442\u044c\u0434\u0435\u0441\u044f\u0442",
    70: "\u0441\u0435\u043c\u044c\u0434\u0435\u0441\u044f\u0442",
    80: "\u0432\u043e\u0441\u0435\u043c\u044c\u0434\u0435\u0441\u044f\u0442",
    90: "\u0434\u0435\u0432\u044f\u043d\u043e\u0441\u0442\u043e",
}

_SCI_RE = re.compile(r"(?<![\w_])([+-]?\d+(?:[.,]\d+)?)[eE]([+-]?\d+)(?![\w_])")
_DECIMAL_RE = re.compile(r"(?<![\w_])([+-]?\d+[.,]\d+)(?![\w_])")
_VERSION_CONTEXT_RE = re.compile(
    r"(?i)(?:\bversion\b|\bengine\b|\bunreal\s+engine\b|\b\u0432\u0435\u0440\u0441\u0438\u044f\b)\s+$"
)


def normalize_text_for_tts(text: str, language: str) -> str:
    if (language or "").strip().lower() != "ru":
        return text
    return _normalize_ru(text or "")


def _normalize_ru(text: str) -> str:
    text = _SCI_RE.sub(lambda match: _sci_to_ru(match.group(1), match.group(2)), text)
    text = _DECIMAL_RE.sub(lambda match: _decimal_match_to_ru(text, match), text)
    return text


def _decimal_match_to_ru(full_text: str, match: re.Match[str]) -> str:
    prefix = full_text[max(0, match.start() - 32):match.start()]
    if _VERSION_CONTEXT_RE.search(prefix):
        return match.group(1)
    return _decimal_to_ru(match.group(1))


def _sci_to_ru(mantissa: str, exponent: str) -> str:
    mantissa_words = _decimal_to_ru(mantissa) if re.search(r"[.,]", mantissa) else _signed_int_to_ru(mantissa)
    exponent_words = _signed_int_to_ru(exponent)
    return (
        f"{mantissa_words} "
        "\u0443\u043c\u043d\u043e\u0436\u0438\u0442\u044c \u043d\u0430 "
        "\u0434\u0435\u0441\u044f\u0442\u044c \u0432 \u0441\u0442\u0435\u043f\u0435\u043d\u0438 "
        f"{exponent_words}"
    )


def _decimal_to_ru(token: str) -> str:
    sign = ""
    value = token
    if value.startswith(("-", "+")):
        if value[0] == "-":
            sign = "\u043c\u0438\u043d\u0443\u0441 "
        value = value[1:]

    integer, fractional = re.split(r"[.,]", value, maxsplit=1)
    words = [sign + _unsigned_int_to_ru(integer), "\u0437\u0430\u043f\u044f\u0442\u0430\u044f"]
    words.extend(_DIGIT_WORDS_RU[digit] for digit in fractional)
    return " ".join(words)


def _signed_int_to_ru(token: str) -> str:
    value = token
    sign = ""
    if value.startswith(("-", "+")):
        if value[0] == "-":
            sign = "\u043c\u0438\u043d\u0443\u0441 "
        value = value[1:]
    return sign + _unsigned_int_to_ru(value)


def _unsigned_int_to_ru(token: str) -> str:
    normalized = token.lstrip("0") or "0"
    number = int(normalized)
    if number in _SMALL_INT_WORDS_RU:
        return _SMALL_INT_WORDS_RU[number]
    if number < 100:
        tens = number // 10 * 10
        ones = number % 10
        return " ".join(part for part in (_SMALL_INT_WORDS_RU[tens], _SMALL_INT_WORDS_RU.get(ones, "")) if part)
    return " ".join(_DIGIT_WORDS_RU[digit] for digit in normalized)

