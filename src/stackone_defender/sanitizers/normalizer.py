"""Unicode Normalization.

NFKC normalization + homoglyph replacement to prevent bypass attacks.

``normalize_unicode`` is **safe to return to callers** (preserves legitimate
accents like ``"café"``). The analysis-only helpers ``strip_combining_marks``
and ``normalize_whitespace`` are used by ``PatternDetector.analyze`` and the
high-risk branch of ``Sanitizer`` -- never by code paths that surface text
to consumers.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass


def normalize_unicode(text: str) -> str:
    """Normalize Unicode text using NFKC normalization.

    Safe to return to callers: NFKC + zero-width strip + Cyrillic homoglyph
    fold + curly-punctuation fold. Does **not** decompose accents or strip
    combining marks (those would be data loss on benign content).
    """
    if not text:
        return text
    normalized = unicodedata.normalize("NFKC", text)
    normalized = _normalize_special_characters(normalized)
    return normalized


_COMBINING_MARKS_RE = re.compile(
    r"[\u0300-\u036f\u1ab0-\u1aff\u1dc0-\u1dff\u20d0-\u20ff\ufe20-\ufe2f]"
)


def strip_combining_marks(text: str) -> str:
    """Strip combining diacritical marks across all 5 Unicode ranges.

    Analysis-only -- destroys legitimate accents (``"café"`` -> ``"cafe"``).
    Callers should typically run ``unicodedata.normalize("NFD", text)`` first
    so precomposed characters decompose into base + combining mark.

    Ranges covered:
    - ``U+0300-U+036F`` Combining Diacritical Marks
    - ``U+1AB0-U+1AFF`` Combining Diacritical Marks Extended
    - ``U+1DC0-U+1DFF`` Combining Diacritical Marks Supplement
    - ``U+20D0-U+20FF`` Combining Diacritical Marks for Symbols
    - ``U+FE20-U+FE2F`` Combining Half Marks
    """
    if not text:
        return text
    return _COMBINING_MARKS_RE.sub("", text)


_LETTER_SPACING_RE = re.compile(r"\b(?:[a-zA-Z] ){2,}[a-zA-Z]\b")
_EMBEDDED_NEWLINE_RE = re.compile(r"([a-zA-Z])[\r\n]+([a-zA-Z])")


def normalize_whitespace(text: str) -> str:
    """Collapse obfuscation-via-whitespace into compact forms.

    Two passes:

    1. Letter-by-letter spacing (3+ letters) -- e.g. ``"S Y S T E M"`` ->
       ``"SYSTEM"``. Runs of 2 letters (``"I a"``) are untouched.
    2. Embedded newlines between adjacent letters -- e.g. ``"ign\\nore"`` ->
       ``"ignore"``. **Does not** consume surrounding spaces, so legitimate
       word boundaries survive: ``"ignore\\n previous"`` -> ``"ignore\\n previous"``.

    Operates on ASCII letters only. Must run **after** ``normalize_unicode``
    so any Cyrillic/fullwidth homoglyphs are already folded to ASCII.
    """
    if not text:
        return text
    result = _LETTER_SPACING_RE.sub(lambda m: m.group(0).replace(" ", ""), text)
    result = _EMBEDDED_NEWLINE_RE.sub(r"\1\2", result)
    return result


_REPLACEMENTS: list[tuple[re.Pattern, str]] = [
    # Zero-width characters
    (re.compile(r"[\u200b-\u200d\ufeff]"), ""),
    # Cyrillic homoglyphs
    (re.compile(r"[\u0430]"), "a"),
    (re.compile(r"[\u0435]"), "e"),
    (re.compile(r"[\u043e]"), "o"),
    (re.compile(r"[\u0440]"), "p"),
    (re.compile(r"[\u0441]"), "c"),
    (re.compile(r"[\u0443]"), "y"),
    (re.compile(r"[\u0445]"), "x"),
    (re.compile(r"[\u0456]"), "i"),
    # Quotes
    (re.compile(r"[\u2018\u2019\u201b\u0060\u00b4]"), "'"),
    (re.compile(r"[\u201c\u201d\u201e\u201f]"), '"'),
    # Dashes
    (re.compile(r"[\u2010-\u2015\u2212]"), "-"),
    # Dots
    (re.compile(r"[\u2024]"), "."),
    (re.compile(r"[\u2026]"), "..."),
    # Colons
    (re.compile(r"[\u02d0]"), ":"),
    (re.compile(r"[\ua789]"), ":"),
]


def _normalize_special_characters(text: str) -> str:
    result = text
    for pattern, replacement in _REPLACEMENTS:
        result = pattern.sub(replacement, result)
    return result


def contains_suspicious_unicode(text: str) -> bool:
    if not text:
        return False
    result = analyze_suspicious_unicode(text)
    return result["has_suspicious"]


@dataclass
class SuspiciousUnicodeAnalysis:
    has_suspicious: bool = False
    zero_width: bool = False
    mixed_script: bool = False
    math_symbols: bool = False
    fullwidth: bool = False
    combining_marks: bool = False


def analyze_suspicious_unicode(text: str) -> dict[str, bool]:
    """Return a detailed breakdown of suspicious Unicode in *text*.

    Returns a dict with keys: has_suspicious, zero_width, mixed_script,
    math_symbols, fullwidth, combining_marks.
    """
    if not text:
        return {
            "has_suspicious": False,
            "zero_width": False,
            "mixed_script": False,
            "math_symbols": False,
            "fullwidth": False,
            "combining_marks": False,
        }

    zero_width = bool(re.search(r"[\u200b-\u200d\ufeff]", text))
    has_cyrillic = bool(re.search(r"[\u0400-\u04ff]", text))
    has_latin = bool(re.search(r"[a-zA-Z]", text))
    mixed_script = has_cyrillic and has_latin
    math_symbols = bool(re.search(r"[\U0001d400-\U0001d7ff]", text))
    fullwidth = bool(re.search(r"[\uff00-\uffef]", text))
    combining_marks = len(_COMBINING_MARKS_RE.findall(text)) >= 3
    has_suspicious = zero_width or mixed_script or math_symbols or fullwidth or combining_marks

    return {
        "has_suspicious": has_suspicious,
        "zero_width": zero_width,
        "mixed_script": mixed_script,
        "math_symbols": math_symbols,
        "fullwidth": fullwidth,
        "combining_marks": combining_marks,
    }
