"""Unicode Normalization.

NFKC normalization + homoglyph replacement to prevent bypass attacks.

``normalize_unicode`` is **safe to return to callers** (preserves legitimate
accents like ``"café"``). The analysis-grade helpers ``strip_combining_marks``
and ``normalize_whitespace`` are used by ``PatternDetector.analyze`` and by
the high-risk branch of ``Sanitizer``. Because Sanitizer Step 1.5 *returns*
the heavy-normalized string, these helpers can reach consumers, but only at
``risk_level="high"`` -- where defanging obfuscation outweighs preserving
the original surface form.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

from .leet_normalizer import LeetGate, normalize_leet_speak


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


_COMBINING_MARKS_RE = re.compile(r"[\u0300-\u036f\u1ab0-\u1aff\u1dc0-\u1dff\u20d0-\u20ff\ufe20-\ufe2f]")


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


# ASCII \b, as in JavaScript.
_LETTER_SPACING_RE = re.compile(r"\b(?:[a-zA-Z] ){2,}[a-zA-Z]\b", re.ASCII)
_EMBEDDED_NEWLINE_RE = re.compile(r"([a-zA-Z])[\r\n]+([a-zA-Z])")
_JOINED_PAIR_RE = re.compile(r"([A-Za-z])[-_.]+(?=[A-Za-z])")


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
    return join_line_split_words(collapse_spaced_letters(text))


def collapse_spaced_letters(text: str) -> str:
    """Collapse letter-by-letter spacing: ``"S Y S T E M"`` -> ``"SYSTEM"``."""
    if not text:
        return text
    return _LETTER_SPACING_RE.sub(lambda m: m.group(0).replace(" ", ""), text)


def split_joined_word_pairs(text: str) -> str:
    """Split two words joined by ``- _ .`` (``"ignore-previous"``, ``"ignore__all"``).

    Tier 1 only: its patterns are specific, but Tier 2 would read every
    compound and hostname apart.
    """
    if not text:
        return text
    return _JOINED_PAIR_RE.sub(r"\1 ", text)


def join_line_split_words(text: str) -> str:
    """Remove line breaks between adjacent letters (``"ign\\nore"`` -> ``"ignore"``).

    Tier 1 only: it also joins the words of ordinary multi-line text, which
    would hurt Tier 2's scoring.
    """
    if not text:
        return text
    return _EMBEDDED_NEWLINE_RE.sub(r"\1\2", text)


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


# Greek and Cyrillic letters that look like Latin ones, and their Latin letter
# (the 8 common Cyrillic ones are mapped unconditionally in normalize_unicode).
_CONFUSABLE_FROM = "αγεικνορτυχωϲϳΑΒΕΖΗΙΚΜΝΟΡΤΥΧјѕԁһӏԛԝкАВЕКМНОРСТХУІЈЅıɡɑ"
_CONFUSABLE_TO = "ayeikvoptuxwcjABEZHIKMNOPTYXjsdhlqwkABEKMHOPCTXYIJSiga"
_CONFUSABLES = str.maketrans(_CONFUSABLE_FROM, _CONFUSABLE_TO)
_CONFUSABLE_SET = frozenset(_CONFUSABLE_FROM)
_CONFUSABLE_RE = re.compile(f"[{_CONFUSABLE_FROM}]")
_GREEK_CYRILLIC_RE = re.compile(r"[\u0370-\u03ff\u0400-\u04ff\u0500-\u052f]")
# Letters and digits, like JavaScript's [\p{L}\p{N}].
_WORD_RE = re.compile(r"[^\W_]+")
_HAS_ASCII_LETTER_RE = re.compile(r"[A-Za-z]")


def _is_native(s: str) -> bool:
    return any(c not in _CONFUSABLE_SET for c in _GREEK_CYRILLIC_RE.findall(s))


def fold_confusables(text: str) -> str:
    """Fold Greek/Cyrillic look-alikes to Latin.

    In words that hold a Latin letter or only look-alikes. Words with any other
    Greek/Cyrillic letter are real Greek or Russian and are left alone, as are
    one-letter words (Greek omicron) in Greek or Russian text.
    """
    if not text or not _CONFUSABLE_RE.search(text):
        return text
    native_text = _is_native(text)

    def _fold(m: re.Match[str]) -> str:
        word = m.group(0)
        if not _CONFUSABLE_RE.search(word) or _HAS_ASCII_LETTER_RE.search(word):
            return word.translate(_CONFUSABLES)
        if _is_native(word) or (native_text and len(word) < 2):
            return word
        return word.translate(_CONFUSABLES)

    return _WORD_RE.sub(_fold, text)


# 3+ single characters split by one space . - or _ ("i.g.n.o.r.e a-l-l").
_SPACED_CHARS_RE = re.compile(r"(?<![A-Za-z0-9@$])[A-Za-z0-9@$](?:[ ._-][A-Za-z0-9@$]){2,}(?![A-Za-z0-9@$])")
_THREE_LETTERS_RE = re.compile(r"[A-Za-z](?:.*[A-Za-z]){2}")
_PUNCT_SEPARATOR_RE = re.compile(r"[._-]")


def _join_spaced_chars(m: re.Match[str]) -> str:
    """Join each space-separated part of a spaced-out run when the run has 3+ letters."""
    run = m.group(0)
    if not _THREE_LETTERS_RE.search(run):
        return run
    return " ".join(_PUNCT_SEPARATOR_RE.sub("", w) for w in run.split(" "))


_JOINED_WORDS_RE = re.compile(r"(?<![A-Za-z0-9])[A-Za-z0-9]+(?:[-_./,][A-Za-z0-9]+){2,}(?![A-Za-z0-9])")
_WORD_SEPARATOR_RE = re.compile(r"[-_./,]")


def _split_joined_words(m: re.Match[str]) -> str:
    parts = _WORD_SEPARATOR_RE.split(m.group(0))
    # Digit-only runs (dates, zero GUIDs, versions) stay as they are.
    if sum(1 for w in parts if _HAS_ASCII_LETTER_RE.search(w)) < 2:
        return m.group(0)
    return " ".join(parts)


def normalize_word_separators(text: str) -> str:
    """Join spaced-out letters, then split words joined by separators.

    ``"i.g.n.o.r.e a-l-l"`` -> ``"ignore all"``; runs of 3+ words joined by ``- _ . / ,``
    (``"ignore-previous-instructions"``) become space-separated. Two-word
    compounds (``"well-known"``), short abbreviations (``"a.m."``) and
    digit-only runs are left alone.
    """
    if not text:
        return text
    return _JOINED_WORDS_RE.sub(_split_joined_words, _SPACED_CHARS_RE.sub(_join_spaced_chars, text))


# ANSI escapes (7- and 8-bit) and C0/C1 control characters (not tab, LF, CR).
# A space, not "": they are spliced in place of spaces ("ignore<ESC>all") to break \s+ patterns.
_CONTROL_RE = re.compile(
    r"\x1b\[[0-9;?]*[A-Za-z]|\x9b[0-9;?]*[A-Za-z]|\x1b[@-_]|[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]"
)
# Invisible format characters normalize_unicode keeps: soft hyphen, Mongolian vowel
# separator, LRM/RLM, word joiner and invisible operators, variation selectors.
_INVISIBLE_RE = re.compile(r"[\u00ad\u180e\u200e\u200f\u2060-\u2064\ufe00-\ufe0f]")
# The same plus the zero-width characters normalize_unicode drops, for the alternate reading.
_ALL_INVISIBLE_RE = re.compile(r"[\u00ad\u180e\u200b-\u200f\u2060-\u2064\ufeff\ufe00-\ufe0f]")


# URLs and emails, which separator splitting would turn into prose.
_LINK_RE = re.compile(
    r"((?:[A-Za-z][A-Za-z0-9+.-]*://|www\.)[^\s\"'<>]+|[A-Za-z0-9_.+-]+@[A-Za-z0-9_-]+(?:\.[A-Za-z0-9_-]+)+)"
)


def normalize_for_detection(
    text: str, *, alternate: bool = False, keep_links: bool = False, leet_gate: LeetGate | None = None
) -> str:
    """The analysis copy both tiers scan. Never returned to callers.

    Undoes control characters, combining marks, unicode variants, look-alikes,
    joined words, spaced letters and leet. ``alternate`` gives the other reading
    of ambiguous text, for Tier 1: invisible characters as spaces rather than
    removed, and leet reversed in ID-like tokens too. ``keep_links`` leaves URLs
    and emails out of separator splitting. ``leet_gate`` decides which leet
    tokens are decoded.
    """
    if not text:
        return text
    text = _CONTROL_RE.sub(" ", text)
    text = _ALL_INVISIBLE_RE.sub(" ", text) if alternate else _INVISIBLE_RE.sub("", text)
    text = fold_confusables(normalize_unicode(strip_combining_marks(unicodedata.normalize("NFD", text))))
    if keep_links:
        parts = _LINK_RE.split(text)
        text = "".join(p if i % 2 else normalize_word_separators(p) for i, p in enumerate(parts))
    else:
        text = normalize_word_separators(text)
    return normalize_leet_speak(collapse_spaced_letters(text), protect_ids=not alternate, gate=leet_gate)


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
