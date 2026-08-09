from __future__ import annotations

from dataclasses import dataclass
import re


PLACEHOLDER_RE = re.compile(
    r"(?<![A-Za-z0-9])(?:__)?DUB[_\s-]*TECH[_\s-]*TOKEN[_\s-]*\d{4}(?:__)?(?![A-Za-z0-9])",
    re.IGNORECASE,
)

_TOKEN_RE = re.compile(
    r"""
    (?<![\w_])
    (
        [+-]?\d+(?:[.,]\d+)?[eE][+-]?\d+
        |
        [+-]?\d+(?:[.,]\d+)?%
        |
        \d+\s*[xX\u00d7]\s*\d+
        |
        [+-]?\d+(?:[.,]\d+)?\u00b0
        |
        [+-]?\d+[.,]\d+
    )
    (?![\w_])
    """,
    re.VERBOSE,
)


class TechnicalTokenError(ValueError):
    """Raised when protected technical tokens cannot be restored safely."""


@dataclass(frozen=True)
class ProtectedText:
    text: str
    tokens: dict[str, str]


class TechnicalTokenProtector:
    def protect(self, text: str) -> ProtectedText:
        tokens: dict[str, str] = {}

        def replace(match: re.Match[str]) -> str:
            placeholder = f"DUBTECHTOKEN{len(tokens):04d}"
            tokens[placeholder] = match.group(1)
            return placeholder

        return ProtectedText(text=_TOKEN_RE.sub(replace, text or ""), tokens=tokens)

    def restore(self, text: str, tokens: dict[str, str]) -> str:
        restored = text or ""
        for placeholder, value in tokens.items():
            placeholder_re = _placeholder_variant_re(placeholder)
            matches = list(placeholder_re.finditer(restored))
            if len(matches) != 1:
                raise TechnicalTokenError(
                    f"Expected placeholder {placeholder} exactly once, found {len(matches)}"
                )
            restored = placeholder_re.sub(lambda _match: value, restored, count=1)

        unresolved = PLACEHOLDER_RE.findall(restored)
        if unresolved:
            unique = ", ".join(sorted(set(unresolved)))
            raise TechnicalTokenError(f"Unresolved technical placeholders remain: {unique}")

        return restored


def _placeholder_variant_re(placeholder: str) -> re.Pattern[str]:
    chars = [re.escape(char) for char in placeholder]
    body = r"[\s_-]*".join(chars)
    return re.compile(rf"(?<![A-Za-z0-9]){body}(?![A-Za-z0-9])", re.IGNORECASE)
