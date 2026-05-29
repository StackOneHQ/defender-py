"""Leet-speak normalization.

Reverses common digit/symbol substitutions used to obfuscate injection
keywords from regex-based detection (e.g. ``"1gn0r3"`` -> ``"ignore"``).

The normalized output is used for **analysis only** -- it must never be
returned to callers, because some substitutions (notably ``$ -> s``) are
lossy on legitimate content.
"""

from __future__ import annotations

import re

#: Leet-speak substitution map. Each entry maps a character to its most
#: common alphabetic equivalent.
LEET_MAP: dict[str, str] = {
    "4": "a",
    "@": "a",
    "8": "b",
    "3": "e",
    "1": "i",
    "0": "o",
    "5": "s",
    "$": "s",
    "7": "t",
}

#: Sequences that must not be modified by leet normalization.
#:
#: Covers:
#: - Hex escape sequences: ``\xHH``
#: - Unicode escape sequences: ``\uHHHH``
#: - Base64-like blobs (20+ base64 chars): corrupting these breaks encoding
#:   detection patterns and the entropy check.
#: - Shell substitution: ``$(`` -- mapping ``$ -> s`` here would break the
#:   ``$()`` pattern in the command-execution detector.
PROTECTED_SEQUENCE = re.compile(
    r"\\x[0-9A-Fa-f]{2}|\\u[0-9A-Fa-f]{4}|\$\(|[A-Za-z0-9+/]{20,}={0,2}"
)

_TOKEN_RE = re.compile(r"[@a-zA-Z0-9!$]+")
_HAS_LETTER_RE = re.compile(r"[a-zA-Z]")
_ALNUM_RE = re.compile(r"[a-zA-Z0-9]")


def _apply_leet_map_chars(token: str) -> str:
    """Apply leet substitution character-by-character within a single token.

    The ``!`` character is substituted for ``"i"`` only when flanked by
    alphanumeric characters, to preserve legitimate sentence-ending
    punctuation (e.g. ``"hello!"`` stays unchanged).
    """
    out: list[str] = []
    for i, ch in enumerate(token):
        if ch in LEET_MAP:
            out.append(LEET_MAP[ch])
            continue

        if ch == "!":
            prev_ch = token[i - 1] if i > 0 else ""
            next_ch = token[i + 1] if i < len(token) - 1 else ""
            if _ALNUM_RE.match(prev_ch) and _ALNUM_RE.match(next_ch):
                out.append("i")
                continue

        out.append(ch)
    return "".join(out)


def _apply_leet_map_token_aware(text: str) -> str:
    """Token-aware leet substitution.

    Splits text into alphanumeric tokens (``[@a-zA-Z0-9!$]+``) and
    non-alphanumeric segments. Only tokens that contain at least one
    letter are normalized -- this prevents pure-digit sequences like
    ``"100"`` or ``"2024"`` from being corrupted.

    ``@``, ``!``, ``$`` are included so ``"@dm1n"``, ``"adm!n"``,
    ``"$y$tem"`` are processed as a single mixed token.
    """

    def _repl(match: re.Match[str]) -> str:
        token = match.group(0)
        if not _HAS_LETTER_RE.search(token):
            return token
        return _apply_leet_map_chars(token)

    return _TOKEN_RE.sub(_repl, text)


def normalize_leet_speak(text: str) -> str:
    """Normalize leet-speak substitutions in text.

    Converts digit and symbol substitutions back to their alphabetic
    equivalents so that existing injection patterns can match obfuscated
    variants (e.g. ``"1gn0r3 4ll rul3s"`` -> ``"ignore all rules"``).

    Encoding sequences (hex escapes, unicode escapes, base64 blobs) and
    shell substitution syntax ``$(`` are left untouched to avoid corrupting
    encoding-detection patterns.

    Pure-digit tokens (``"100"``, ``"2024"``) are left unchanged.
    """
    if not text:
        return text

    segments: list[str] = []
    last_index = 0
    for match in PROTECTED_SEQUENCE.finditer(text):
        segments.append(_apply_leet_map_token_aware(text[last_index : match.start()]))
        segments.append(match.group(0))
        last_index = match.end()
    segments.append(_apply_leet_map_token_aware(text[last_index:]))
    return "".join(segments)
