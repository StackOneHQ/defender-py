"""Text helpers that reproduce JavaScript string semantics.

The shared conformance fixtures pin the TypeScript package's output, so code that cuts or
matches untrusted text follows JS rules: lengths in UTF-16 code units, and JS's ``\\s`` and
``\\p{Cf}`` sets (Python's differ, and ``unicodedata`` varies by Python version).
"""

from __future__ import annotations

import re
from typing import Any

# JS `\s`: what `/\s/` matches (and `String.prototype.trim` removes).
JS_WS = "\t\n\x0b\x0c\r \xa0\u1680\u2000-\u200a\u2028\u2029\u202f\u205f\u3000\ufeff"
JS_WS_CHARS = "\t\n\x0b\x0c\r \xa0\u1680\u2000\u2001\u2002\u2003\u2004\u2005\u2006\u2007\u2008\u2009\u200a\u2028\u2029\u202f\u205f\u3000\ufeff"
# JS `\p{Cf}` (format characters), Unicode 16.0 as in Node 22.
JS_CF = (
    "\xad\u0600-\u0605\u061c\u06dd\u070f\u0890\u0891\u08e2\u180e\u200b-\u200f\u202a-\u202e"
    "\u2060-\u2064\u2066-\u206f\ufeff\ufff9-\ufffb\U000110bd\U000110cd\U00013430-\U0001343f"
    "\U0001bca0-\U0001bca3\U0001d173-\U0001d17a\U000e0001\U000e0020-\U000e007f"
)

_ASTRAL = re.compile("[\U00010000-\U0010ffff]")
_SURROGATE = re.compile("[\ud800-\udfff]")


def _pair(m: re.Match[str]) -> str:
    c = ord(m[0]) - 0x10000
    return chr(0xD800 + (c >> 10)) + chr(0xDC00 + (c & 0x3FF))


def to_utf16(s: str) -> str:
    """``s`` with astral characters split into surrogate pairs, so ``len`` and slicing match JS."""
    return _ASTRAL.sub(_pair, s) if _ASTRAL.search(s) else s


def from_utf16(s: str) -> str:
    """Join surrogate pairs back into characters. An unpaired surrogate becomes U+FFFD, so a
    consumer that encodes to UTF-8 can't be made to fail by one."""
    if not _SURROGATE.search(s):
        return s
    return s.encode("utf-16-le", "surrogatepass").decode("utf-16-le", "replace")


def utf16_len(s: str) -> int:
    if not _ASTRAL.search(s):
        return len(s)
    return len(s.encode("utf-16-le", "surrogatepass")) // 2


def _is_high(c: str) -> bool:
    return "\ud800" <= c <= "\udbff"


def _is_low(c: str) -> bool:
    return "\udc00" <= c <= "\udfff"


def safe_end(s: str, end: int) -> int:
    """An end index <= ``end`` that doesn't split a surrogate pair (``s`` in a UTF-16 view)."""
    if 0 < end < len(s) and _is_high(s[end - 1]) and _is_low(s[end]):
        return end - 1
    return end


def safe_start(s: str, start: int) -> int:
    """A start index >= ``start`` that doesn't split a surrogate pair (``s`` in a UTF-16 view)."""
    if 0 < start < len(s) and _is_low(s[start]) and _is_high(s[start - 1]):
        return start + 1
    return start


def truncate(s: str, max_units: int) -> str:
    """The first ``max_units`` UTF-16 code units of ``s``, never half a surrogate pair."""
    if utf16_len(s) <= max_units:
        return s
    u = to_utf16(s)
    return from_utf16(u[: safe_end(u, max(0, max_units))])


def js_trim(s: str) -> str:
    return s.strip(JS_WS_CHARS)


def js_typeof(v: Any) -> str:
    """JS ``typeof`` for a value decoded from JSON (``null`` is ``"object"``)."""
    if v is None:
        return "object"
    if isinstance(v, bool):
        return "boolean"
    if isinstance(v, (int, float)):
        return "number"
    if isinstance(v, str):
        return "string"
    if callable(v):
        return "function"
    return "object"
