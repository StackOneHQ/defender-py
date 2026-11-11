"""Leet-speak normalization.

Reverses common digit/symbol substitutions used to obfuscate injection
keywords from regex-based detection (e.g. ``"1gn0r3"`` -> ``"ignore"``).

The normalized output is used for **analysis only** -- it must never be
returned to callers, because some substitutions (notably ``$ -> s``) are
lossy on legitimate content.
"""

from __future__ import annotations

import re
from collections.abc import Callable

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
_NON_LEET_DIGIT_RE = re.compile(r"[269]")
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


# Short words attacks spell in leet with one letter left ("4n 4dm1n", "4r3", "7h3").
# Other tokens with under 2 letters ("v1", "Q3") are IDs.
_SHORT_WORDS = frozenset(
    (  # noqa: SIM905
        "am an as at be by do go if in is it me my no of on or so to up us we act all and any are can for get "
        "has how its may new not now off our out own run set the use was who why you"
    ).split()
)
_HEX_RE = re.compile(r"[0-9a-fA-F]+")
_EDGE_RE = re.compile(r"^([0-9]+)(.*)$|^(.*?)([0-9]+)$", re.S)
_LEET_CHAR_RE = re.compile(r"[0134578@!$]")


def _is_identifier_like(token: str) -> bool:
    """IDs, hashes and versions (``"user123"``, ``"d14f0615"``, ``"v2"``).

    Under 2 letters (unless a short word), hex with a digit, or at either end
    4+ digits, or 2-3 with a 2, 6 or 9 when the rest has no leet (``"COVID19"``,
    ``"user123"``, not ``"rul35"``, ``"573al"`` or ``"22ign0re"``). Leet in them
    would only produce noise.
    """
    if len(_HAS_LETTER_RE.findall(token)) < 2:
        return _NON_LEET_DIGIT_RE.sub("", _apply_leet_map_chars(token)).lower() not in _SHORT_WORDS
    if len(token) >= 6 and _HEX_RE.fullmatch(token) and any(c.isdigit() for c in token):
        return True
    # A short digit run marks an ID only when the rest has no leet ("user123", not "22ign0re").
    edge = _EDGE_RE.match(token)
    if not edge:
        return False
    run = edge.group(1) if edge.group(1) is not None else edge.group(4)
    rest = edge.group(2) if edge.group(1) is not None else edge.group(3)
    return len(run) >= 4 or (len(run) >= 2 and bool(_NON_LEET_DIGIT_RE.search(run)) and not _LEET_CHAR_RE.search(rest))


LeetGate = Callable[[str, str], bool]
"""Decides whether a leet token is replaced by its decoded form: ``gate(token, decoded)``."""


def _apply_leet_map_token_aware(text: str, protect_ids: bool, gate: LeetGate | None = None) -> str:
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
        if not _HAS_LETTER_RE.search(token) or (protect_ids and _is_identifier_like(token)):
            return token
        mapped = _apply_leet_map_chars(token)
        if mapped == token:
            return token
        # Digits left in a leet word are padding ("d1sreg4rd2").
        decoded = _NON_LEET_DIGIT_RE.sub("", mapped)
        return token if gate is not None and not gate(token, decoded) else decoded

    return _TOKEN_RE.sub(_repl, text)


def normalize_leet_speak(text: str, protect_ids: bool = True, gate: LeetGate | None = None) -> str:
    """Normalize leet-speak substitutions in text.

    Converts digit and symbol substitutions back to their alphabetic
    equivalents so that existing injection patterns can match obfuscated
    variants (e.g. ``"1gn0r3 4ll rul3s"`` -> ``"ignore all rules"``).

    Encoding sequences (hex escapes, unicode escapes, base64 blobs) and
    shell substitution syntax ``$(`` are left untouched to avoid corrupting
    encoding-detection patterns.

    Pure-digit tokens (``"100"``, ``"2024"``) are left unchanged. When ``gate``
    is given, a token is decoded only when ``gate(token, decoded)`` is true.
    """
    if not text:
        return text

    segments: list[str] = []
    last_index = 0
    for match in PROTECTED_SEQUENCE.finditer(text):
        segments.append(_apply_leet_map_token_aware(text[last_index : match.start()], protect_ids, gate))
        segments.append(match.group(0))
        last_index = match.end()
    segments.append(_apply_leet_map_token_aware(text[last_index:], protect_ids, gate))
    return "".join(segments)
