"""Make provider error text safe to put in ``skip_reason``, ``note`` and events.

Port of the TypeScript package's ``redactErrorText``. Patterns use JS semantics (ASCII ``\\w``
and ``\\b``, JS ``\\s``) so the output matches the shared conformance fixtures.
"""

from __future__ import annotations

import re

from .text import JS_WS, js_trim, truncate, utf16_len

_MAX_ERROR_TEXT = 200
_MAX_INPUT_TEXT = 2000
_S = f"[{JS_WS}]"
_NS = f"[^{JS_WS}]"
_DOT = "[^\n\r\u2028\u2029]"  # JS `.`
# Secret-looking field names. Bounded repeats keep matching linear on hostile input. `token`, `pass`
# and `pwd` must end the name, so `max_tokens` / `prompt_tokens` / `passed` stay readable.
_SECRET_NAME = (
    r"(?:[\w-]{0,40}(?:key|secret|password|passwd|passphrase|credentials?|authorization)[\w-]{0,40}"
    r"|[\w-]{0,40}(?:token|pass|pwd)\b)"
)
# Password-like names, whose values may contain spaces: redact to the next delimiter.
_SPACED_SECRET_NAME = r"(?:[\w-]{0,40}(?:password|passwd|passphrase)[\w-]{0,40}|pass|pwd)"
# A credential after an auth scheme: long enough and token-shaped, so prose like "Token limit" stays.
_CREDENTIAL = r"[A-Za-z0-9._~+/=-]{12,}"

_AI = re.ASCII | re.IGNORECASE
_A = re.ASCII

_RULES: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----[\s\S]*"), "[redacted-private-key]"),
    # URLs of any scheme: keep scheme + host only (drops userinfo, path and query). Userinfo runs to
    # the last `@`, so a password containing `@` doesn't leak into the host.
    (re.compile(rf"\b([a-z][a-z0-9+.-]*)://(?:[^/{JS_WS}?#]*@)?([^/@{JS_WS}?#]+){_NS}*", _AI), r"\1://\2"),
    # Bearer/Basic are always followed by a credential. "Token"/"Digest" are also ordinary words, so
    # they need a credential-shaped value.
    # A quoted value whole, else stop at a quote or comma so the next pair in a dict repr stays matchable.
    (re.compile(rf"\b(Bearer|Basic){_S}+(?:\"[^\"]*\"?|'[^']*'?|[^{JS_WS}'\",]+)", _AI), r"\1 [redacted]"),
    (re.compile(rf"\b(Token|Digest){_S}+{_CREDENTIAL}", _AI), r"\1 [redacted]"),
    (
        re.compile(rf"\b(authorization|proxy-authorization)({_S}*[:=]{_S}*)(?:[A-Za-z]+{_S}+)?{_CREDENTIAL}", _AI),
        r"\1\2[redacted]",
    ),
    (re.compile(rf"\b(cookie|set-cookie)({_S}*:{_S}*)(?:[\w.-]+=[^;{JS_WS}]*(?:;{_S}*)?)+", _AI), r"\1\2[redacted]"),
    # Quoted pairs, JSON or Python-repr. The closing quote is optional, so a cut-off value is removed too.
    (re.compile(rf'("{_SECRET_NAME}"{_S}*:{_S}*)"(?:[^"\\]|\\{_DOT})*"?', _AI), r'\1"[redacted]"'),
    (re.compile(rf"('{_SECRET_NAME}'{_S}*:{_S}*)'(?:[^'\\]|\\{_DOT})*'?", _AI), r"\1'[redacted]'"),
    # Unquoted name, quoted value: password: 'x' / token="x".
    (re.compile(rf"\b({_SECRET_NAME})({_S}*[:=]{_S}*)'(?:[^'\\]|\\{_DOT})*'?", _AI), r"\1\2'[redacted]'"),
    (re.compile(rf'\b({_SECRET_NAME})({_S}*[:=]{_S}*)"(?:[^"\\]|\\{_DOT})*"?', _AI), r'\1\2"[redacted]"'),
    # Values that may contain spaces: redact up to the next delimiter.
    (
        re.compile(rf"\b({_SPACED_SECRET_NAME})({_S}*[:=]{_S}*)[^{JS_WS}&,;\"'\[][^&,;\"'\n\[]*", _AI),
        r"\1\2[redacted]",
    ),
    (re.compile(rf"\b({_SECRET_NAME})({_S}*[:=]{_S}*)[^{JS_WS}&,;\"'\[]+", _AI), r"\1\2[redacted]"),
    # Token shapes from common providers.
    (re.compile(r"\b(sk|rk|pk)([-_])(?:live_|test_)?[A-Za-z0-9_-]{8,}", _A), r"\1\2[redacted]"),
    (re.compile(r"\bAIza[0-9A-Za-z_-]{20,}", _A), "AIza[redacted]"),
    (re.compile(r"\b(gh[pousr]_|github_pat_)[A-Za-z0-9_]{16,}", _A), r"\1[redacted]"),
    (re.compile(r"\b(AKIA|ASIA)[0-9A-Z]{16}\b", _A), r"\1[redacted]"),
    (re.compile(r"\bxox[abposr]-[A-Za-z0-9-]{8,}", _A), "xox-[redacted]"),
    (re.compile(r"\bhf_[A-Za-z0-9]{20,}", _A), "hf_[redacted]"),
    (re.compile(r"\bglpat-[A-Za-z0-9_-]{16,}", _A), "glpat-[redacted]"),
    (re.compile(r"\bnpm_[A-Za-z0-9]{30,}", _A), "npm_[redacted]"),
    (re.compile(r"\bSG\.[A-Za-z0-9_-]{16,}\.[A-Za-z0-9_-]{16,}", _A), "SG.[redacted]"),
    (re.compile(r"\beyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]+", _A), "[redacted-jwt]"),
    (re.compile(f"{_S}+"), " "),
]


def redact_error_text(text: str) -> str:
    """Strip URL paths and queries, redact token-shaped strings, and cap the length, so keys,
    echoed prompts or content can't ride along. Best effort: a custom provider should still
    raise short, fixed messages."""
    # Cap first: bounded input keeps every pattern below cheap.
    cleaned = truncate(text, _MAX_INPUT_TEXT)
    for pattern, repl in _RULES:
        cleaned = pattern.sub(repl, cleaned)
    cleaned = js_trim(cleaned)
    return f"{truncate(cleaned, _MAX_ERROR_TEXT - 1)}\u2026" if utf16_len(cleaned) > _MAX_ERROR_TEXT else cleaned


def describe_error(err: BaseException) -> str:
    """``str(err)`` that can't itself raise."""
    try:
        return str(err)
    except Exception:
        return "<unstringifiable exception>"
