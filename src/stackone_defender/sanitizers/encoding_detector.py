"""Encoding Detection.

Detects and handles Base64, URL-encoded, hex/Unicode escapes, HTML entities,
ROT13/ROT47, binary strings, and Morse code that might hide injection
attempts. ``decode_all_levels`` iteratively unwraps chained encodings (e.g.
base64 of hex-escaped content) so deep-nested payloads can be classified.
"""

from __future__ import annotations

import base64
import re
import urllib.parse
from dataclasses import dataclass, field
from typing import Literal

EncodingType = Literal[
    "base64",
    "url",
    "hex",
    "unicode_escape",
    "html_entity",
    "rot13",
    "rot47",
    "binary",
    "morse",
]


@dataclass
class EncodingDetection:
    type: EncodingType
    original: str
    decoded: str | None = None
    position: int = 0
    length: int = 0
    suspicious: bool = False


@dataclass
class EncodingDetectionResult:
    has_encoding: bool = False
    encoding_types: list[str] = field(default_factory=list)
    detections: list[EncodingDetection] = field(default_factory=list)
    processed_text: str | None = None


# Shared keyword check used by every detector's ``suspicious`` flag.
_SUSPICIOUS_RE = re.compile(r"system|ignore|instruction|assistant|bypass|override", re.I)


def detect_encoding(
    text: str,
    *,
    min_base64_length: int = 20,
    decode_base64: bool = True,
    decode_url: bool = True,
    decode_html_entities: bool = True,
    decode_rot13: bool = True,
    decode_rot47: bool = True,
    decode_binary: bool = True,
    decode_morse: bool = True,
    action: Literal["flag", "decode", "redact"] = "flag",
    redact_replacement: str = "[ENCODED DATA DETECTED]",
) -> EncodingDetectionResult:
    if not text:
        return EncodingDetectionResult()

    detections: list[EncodingDetection] = []

    if decode_base64:
        detections.extend(_detect_base64(text, min_base64_length))
    if decode_url:
        detections.extend(_detect_url_encoding(text))
    detections.extend(_detect_hex_encoding(text))
    detections.extend(_detect_unicode_escapes(text))
    if decode_html_entities:
        detections.extend(_detect_html_entities(text))
    if decode_rot13:
        detections.extend(_detect_rot13(text))
    if decode_rot47:
        detections.extend(_detect_rot47(text))
    if decode_binary:
        detections.extend(_detect_binary_strings(text))
    if decode_morse:
        detections.extend(_detect_morse(text))

    encoding_types = list({d.type for d in detections})
    result = EncodingDetectionResult(
        has_encoding=bool(detections),
        encoding_types=encoding_types,
        detections=detections,
    )

    if detections and action in ("decode", "redact"):
        result.processed_text = _process_encoded_content(text, detections, action, redact_replacement)

    return result


def _detect_base64(text: str, min_length: int) -> list[EncodingDetection]:
    detections: list[EncodingDetection] = []
    for m in re.finditer(r"[A-Za-z0-9+/]{20,}={0,2}", text):
        candidate = m.group(0)
        if len(candidate) < min_length:
            continue
        try:
            # Pad to a multiple of 4; b64decode rejects unpadded input, which
            # the except below would silently drop (JS atob tolerates it).
            padded = candidate + "=" * (-len(candidate) % 4)
            decoded_bytes = base64.b64decode(padded, validate=False)
            try:
                decoded = decoded_bytes.decode("ascii")
            except UnicodeDecodeError:
                decoded = decoded_bytes.decode("latin-1", errors="replace")
            is_printable = all(0x20 <= ord(c) <= 0x7e or c in "\t\n\r" for c in decoded)
            is_suspicious = is_printable and bool(_SUSPICIOUS_RE.search(decoded))
            detections.append(
                EncodingDetection(
                    type="base64",
                    original=candidate,
                    decoded=decoded if is_printable else None,
                    position=m.start(),
                    length=len(candidate),
                    suspicious=is_suspicious,
                )
            )
        except Exception:
            pass
    return detections


def _detect_url_encoding(text: str) -> list[EncodingDetection]:
    detections: list[EncodingDetection] = []
    for m in re.finditer(r"(?:%[0-9A-Fa-f]{2}){3,}", text):
        candidate = m.group(0)
        try:
            decoded = urllib.parse.unquote(candidate)
            if decoded != candidate:
                is_suspicious = bool(_SUSPICIOUS_RE.search(decoded))
                detections.append(
                    EncodingDetection(
                        type="url",
                        original=candidate,
                        decoded=decoded,
                        position=m.start(),
                        length=len(candidate),
                        suspicious=is_suspicious,
                    )
                )
        except Exception:
            pass
    return detections


def _detect_hex_encoding(text: str) -> list[EncodingDetection]:
    detections: list[EncodingDetection] = []
    for m in re.finditer(r"(?:\\x[0-9A-Fa-f]{2}){4,}", text):
        candidate = m.group(0)
        try:
            decoded = re.sub(
                r"\\x([0-9A-Fa-f]{2})",
                lambda hm: chr(int(hm.group(1), 16)),
                candidate,
            )
            is_suspicious = bool(_SUSPICIOUS_RE.search(decoded))
            detections.append(
                EncodingDetection(
                    type="hex",
                    original=candidate,
                    decoded=decoded,
                    position=m.start(),
                    length=len(candidate),
                    suspicious=is_suspicious,
                )
            )
        except Exception:
            pass
    return detections


def _detect_unicode_escapes(text: str) -> list[EncodingDetection]:
    detections: list[EncodingDetection] = []
    for m in re.finditer(r"(?:\\u[0-9A-Fa-f]{4}){3,}", text):
        candidate = m.group(0)
        try:
            decoded = re.sub(
                r"\\u([0-9A-Fa-f]{4})",
                lambda um: chr(int(um.group(1), 16)),
                candidate,
            )
            is_suspicious = bool(_SUSPICIOUS_RE.search(decoded))
            detections.append(
                EncodingDetection(
                    type="unicode_escape",
                    original=candidate,
                    decoded=decoded,
                    position=m.start(),
                    length=len(candidate),
                    suspicious=is_suspicious,
                )
            )
        except Exception:
            pass
    return detections


#: Security-relevant named HTML entities (subset of HTML5 -- enough to decode
#: injection keywords). Numeric ``&#NNN;`` / hex ``&#xHH;`` entities are
#: handled separately by the decoder.
HTML_NAMED_ENTITIES: dict[str, str] = {
    "amp": "&",
    "lt": "<",
    "gt": ">",
    "quot": '"',
    "apos": "'",
    "nbsp": " ",
    "sol": "/",
    "colon": ":",
    "lpar": "(",
    "rpar": ")",
    "comma": ",",
    "period": ".",
    "semi": ";",
    "excl": "!",
    "num": "#",
    "dollar": "$",
    "percnt": "%",
    "ast": "*",
    "plus": "+",
    "equals": "=",
    "lsqb": "[",
    "rsqb": "]",
    "lcub": "{",
    "rcub": "}",
    "vert": "|",
    "Hat": "^",
    "grave": "`",
    "tilde": "~",
    "lowbar": "_",
    "hyphen": "-",
}

_HTML_ENTITY_GATE = re.compile(
    r"(?:&#\d{2,5};|&#x[0-9A-Fa-f]{2,5};|&[a-zA-Z]{2,8};){3,}"
)
_HTML_ENTITY_TOKEN = re.compile(
    r"&#(\d{2,5});|&#x([0-9A-Fa-f]{2,5});|&([a-zA-Z]{2,8});"
)


def _decode_html_entity_token(m: re.Match[str]) -> str:
    dec, hex_, named = m.group(1), m.group(2), m.group(3)
    if dec:
        try:
            return chr(int(dec, 10))
        except (ValueError, OverflowError):
            return m.group(0)
    if hex_:
        try:
            return chr(int(hex_, 16))
        except (ValueError, OverflowError):
            return m.group(0)
    if named:
        return HTML_NAMED_ENTITIES.get(named, f"&{named};")
    return m.group(0)


def _detect_html_entities(text: str) -> list[EncodingDetection]:
    """HTML entity detection. Gate: 3+ grouped entity tokens.

    Emits a detection for every grouped run that decodes to a different
    string. ``suspicious`` is set via the shared injection-keyword regex.
    The REDACT-mode filter in ``_process_encoded_content`` drops the
    non-suspicious detections so benign escapes like ``&#49;&#48;&#37;`` =
    ``"10%"`` survive sanitization.
    """
    detections: list[EncodingDetection] = []
    for m in _HTML_ENTITY_GATE.finditer(text):
        candidate = m.group(0)
        decoded = _HTML_ENTITY_TOKEN.sub(_decode_html_entity_token, candidate)
        if decoded == candidate:
            continue
        is_suspicious = bool(_SUSPICIOUS_RE.search(decoded))
        detections.append(
            EncodingDetection(
                type="html_entity",
                original=candidate,
                decoded=decoded,
                position=m.start(),
                length=len(candidate),
                suspicious=is_suspicious,
            )
        )
    return detections


def _rot13(text: str) -> str:
    out: list[str] = []
    for ch in text:
        code = ord(ch)
        if 0x41 <= code <= 0x5a:  # A-Z
            out.append(chr((code - 0x41 + 13) % 26 + 0x41))
        elif 0x61 <= code <= 0x7a:  # a-z
            out.append(chr((code - 0x61 + 13) % 26 + 0x61))
        else:
            out.append(ch)
    return "".join(out)


def _rot47(text: str) -> str:
    out: list[str] = []
    for ch in text:
        code = ord(ch)
        if 0x21 <= code <= 0x7e:  # ! .. ~
            out.append(chr((code - 0x21 + 47) % 94 + 0x21))
        else:
            out.append(ch)
    return "".join(out)


_LETTER_RE = re.compile(r"[A-Za-z]")
_ROT47_PRINTABLE_RE = re.compile(r"[!-~]")


def _detect_rot13(text: str) -> list[EncodingDetection]:
    """Full-text ROT13 detection. Gate: text is 70%+ alphabetic AND decoded
    contains an injection keyword. Conservative -- avoids FP on arbitrary
    high-letter-density text.
    """
    if not text:
        return []
    letter_count = len(_LETTER_RE.findall(text))
    if letter_count / len(text) < 0.7:
        return []
    decoded = _rot13(text)
    if not _SUSPICIOUS_RE.search(decoded):
        return []
    return [
        EncodingDetection(
            type="rot13",
            original=text,
            decoded=decoded,
            position=0,
            length=len(text),
            suspicious=True,
        )
    ]


def _detect_rot47(text: str) -> list[EncodingDetection]:
    """Full-text ROT47 detection. Gate: 15+ printable non-space ASCII chars
    AND decoded contains injection keyword.
    """
    if not text:
        return []
    printable_count = len(_ROT47_PRINTABLE_RE.findall(text))
    if printable_count < 15:
        return []
    decoded = _rot47(text)
    if not _SUSPICIOUS_RE.search(decoded):
        return []
    return [
        EncodingDetection(
            type="rot47",
            original=text,
            decoded=decoded,
            position=0,
            length=len(text),
            suspicious=True,
        )
    ]


_BINARY_GATE = re.compile(r"\b[01]{8}(?:\s+[01]{8}){2,}\b")


def _detect_binary_strings(text: str) -> list[EncodingDetection]:
    """Binary string detection. Gate: 3+ consecutive 8-bit groups. Reject if
    any decoded char is outside the printable ASCII range ``0x20-0x7E``.
    """
    detections: list[EncodingDetection] = []
    for m in _BINARY_GATE.finditer(text):
        candidate = m.group(0)
        groups = candidate.strip().split()
        chars = [chr(int(g, 2)) for g in groups]
        if not all(0x20 <= ord(c) <= 0x7e for c in chars):
            continue
        decoded = "".join(chars)
        is_suspicious = bool(_SUSPICIOUS_RE.search(decoded))
        detections.append(
            EncodingDetection(
                type="binary",
                original=candidate,
                decoded=decoded,
                position=m.start(),
                length=len(candidate),
                suspicious=is_suspicious,
            )
        )
    return detections


#: Morse code table (A-Z + 0-9, lowercase decoded output).
MORSE_TABLE: dict[str, str] = {
    ".-": "a",
    "-...": "b",
    "-.-.": "c",
    "-..": "d",
    ".": "e",
    "..-.": "f",
    "--.": "g",
    "....": "h",
    "..": "i",
    ".---": "j",
    "-.-": "k",
    ".-..": "l",
    "--": "m",
    "-.": "n",
    "---": "o",
    ".--.": "p",
    "--.-": "q",
    ".-.": "r",
    "...": "s",
    "-": "t",
    "..-": "u",
    "...-": "v",
    ".--": "w",
    "-..-": "x",
    "-.--": "y",
    "--..": "z",
    "-----": "0",
    ".----": "1",
    "..---": "2",
    "...--": "3",
    "....-": "4",
    ".....": "5",
    "-....": "6",
    "--...": "7",
    "---..": "8",
    "----.": "9",
}

_MORSE_GATE = re.compile(r"(?:[.-]+ ){4,}[.-]+")


def _detect_morse(text: str) -> list[EncodingDetection]:
    """Morse code detection. Gate: 5+ dot/dash groups separated by spaces.
    Word separator is ``" / "`` (space-slash-space). Reject if >20% of
    symbols are unknown.
    """
    detections: list[EncodingDetection] = []
    for m in _MORSE_GATE.finditer(text):
        candidate = m.group(0).strip()
        words = candidate.split(" / ")
        chars: list[str] = []
        unknowns = 0
        for word in words:
            for sym in word.strip().split(" "):
                if not sym:
                    continue
                if sym in MORSE_TABLE:
                    chars.append(MORSE_TABLE[sym])
                else:
                    chars.append("?")
                    unknowns += 1
            chars.append(" ")
        total_symbols = sum(1 for c in chars if c != " ")
        if total_symbols == 0 or unknowns / total_symbols > 0.2:
            continue
        decoded = "".join(chars).strip()
        is_suspicious = bool(_SUSPICIOUS_RE.search(decoded))
        detections.append(
            EncodingDetection(
                type="morse",
                original=candidate,
                decoded=decoded,
                position=m.start(),
                length=len(candidate),
                suspicious=is_suspicious,
            )
        )
    return detections


def _process_encoded_content(
    text: str,
    detections: list[EncodingDetection],
    action: str,
    redact_replacement: str,
) -> str:
    """Process encoded content based on configuration action.

    Full-text detections (ROT13, ROT47) span ``position=0, length=len(text)``.
    If applied alongside positional detections via the normal reverse-position
    splice loop, they would overwrite partial decodes using the original
    length -- corrupting prior replacements and causing ``decode_all_levels``
    to oscillate rather than converge.

    Resolution: positional detections are applied first (end-to-start splice);
    full-text detections are only applied when there are no positional
    detections. Only the first full-text detection is used when multiple fire
    on the same string (e.g. ROT13 + ROT47).
    """

    def is_full_text(d: EncodingDetection) -> bool:
        return d.position == 0 and d.length == len(text)

    # HTML entities are commonly used for legitimate escaping (e.g.
    # ``&#49;&#48;&#37;`` = "10%"). In REDACT mode, drop benign HTML entity
    # runs so they survive sanitization. The "decode" path still processes
    # them so ``decode_all_levels`` can chain HTML->base64->plaintext correctly.
    if action == "redact":
        filtered = [d for d in detections if d.type != "html_entity" or d.suspicious]
    else:
        filtered = list(detections)

    positional = [d for d in filtered if not is_full_text(d)]
    full_text = [d for d in filtered if is_full_text(d)]

    if positional:
        result = text
        for det in sorted(positional, key=lambda d: d.position, reverse=True):
            replacement = redact_replacement if action == "redact" else (det.decoded or det.original)
            result = result[: det.position] + replacement + result[det.position + det.length :]
        return result

    if full_text:
        det = full_text[0]
        return redact_replacement if action == "redact" else (det.decoded or det.original)

    return text


def contains_encoded_content(text: str) -> bool:
    return detect_encoding(text).has_encoding


def contains_suspicious_encoding(text: str) -> bool:
    result = detect_encoding(text)
    return any(d.suspicious for d in result.detections)


def redact_all_encoding(text: str, replacement: str = "[ENCODED DATA DETECTED]") -> str:
    result = detect_encoding(text, action="redact", redact_replacement=replacement)
    return result.processed_text or text


def decode_all_encoding(text: str) -> str:
    """Decode all encoded content in *text* in a single pass.

    Only unwraps one layer. For chained encodings use ``decode_all_levels``.
    """
    result = detect_encoding(text, action="decode")
    return result.processed_text or text


def decode_all_levels(text: str, max_iterations: int = 5) -> tuple[str, int]:
    """Iteratively decode chained encodings until the output stabilises.

    A single call to ``decode_all_encoding`` only unwraps one layer. Chained
    encodings (e.g. base64 of hex-escaped content) require repeated passes.
    This function loops until the text stops changing or ``max_iterations``
    is reached.

    Safety guards:
    - Hard cap of ``max_iterations`` (default 5) to prevent CPU loops.
    - Aborts if decoded text exceeds 10x the original length (decompression
      bomb protection).

    Returns ``(decoded_text, levels)`` where ``levels`` is the number of
    decode passes applied (0 if the input had no detected encodings).
    """
    if not text:
        return text, 0

    max_length = len(text) * 10
    current = text
    levels = 0
    for _ in range(max_iterations):
        result = detect_encoding(current, action="decode")
        if not result.processed_text or result.processed_text == current:
            break
        if len(result.processed_text) > max_length:
            break
        current = result.processed_text
        levels += 1
    return current, levels


def contains_suspicious_encoding_deep(text: str) -> bool:
    """Check for suspicious encoded content at any nesting depth.

    Unlike ``contains_suspicious_encoding``, this fully unwraps chained
    encodings before checking for suspicious keywords, so double-encoded
    payloads are caught even if the intermediate form looks benign.
    """
    decoded, levels = decode_all_levels(text)
    if levels == 0:
        return contains_suspicious_encoding(text)
    # Also check encoded residue when ``decode_all_levels`` hit the iteration
    # cap before fully unwrapping.
    return bool(_SUSPICIOUS_RE.search(decoded)) or contains_suspicious_encoding(decoded)
