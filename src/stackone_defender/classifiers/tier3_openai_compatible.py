"""A Tier 3 provider for any OpenAI-compatible ``/chat/completions`` endpoint (OpenAI, vLLM,
Ollama, OpenRouter, LiteLLM, compatible gateways). Needs ``httpx``: install the ``[llm]`` extra.

Port of the TypeScript package's ``tier3-openai-compatible``.
"""

from __future__ import annotations

import asyncio
import re
import time
from collections.abc import Mapping
from typing import Any, Literal
from urllib.parse import urlsplit, urlunsplit

from ..utils.text import JS_WS
from .tier3_llm_provider import (
    VERDICT_JSON_SCHEMA,
    LlmCompletionRequest,
    LlmTier3Provider,
    ParseVerdict,
    Prompt,
    create_llm_tier3_provider,
)

# Safety limit when defender sets no deadline (tier3.timeout_ms unset).
_DEFAULT_SAFETY_TIMEOUT_S = 30.0
_SAFE_CODE_RE = re.compile(r"[A-Za-z0-9_.-]{1,64}")
# Error codes that mean the provider refused the content (moderation), which the attacker can cause.
_REFUSAL_CODES = frozenset({"content_filter", "content_policy_violation", "invalid_prompt"})
_LOOPBACK_HOSTS = frozenset({"localhost", "127.0.0.1", "::1"})
_THINK_RE = re.compile(f"^[{JS_WS}]*<think>[\\s\\S]*?</think>[{JS_WS}]*", re.IGNORECASE)
_RESPONSE_FORMATS = ("json_schema", "json_object", "none")
_HEADER_BAD_RE = re.compile(r"[\r\n\x00]")


def _safe_code(value: Any) -> str | None:
    return value if isinstance(value, str) and _SAFE_CODE_RE.fullmatch(value) else None


def _error_code_of(body: Any) -> str | None:
    err = body.get("error") if isinstance(body, Mapping) else None
    if not isinstance(err, Mapping):
        return None
    return _safe_code(err.get("code")) or _safe_code(err.get("type"))


class _HttpError(Exception):
    def __init__(self, status: int, code: str | None):
        super().__init__(f"HTTP {status}{f' {code}' if code else ''}")
        self.status = status


def _chat_completions_url(base_url: str, allow_insecure_http: bool) -> str:
    """Append ``/chat/completions`` to the path, and require https except for loopback."""
    try:
        parts = urlsplit(base_url)
        host = (parts.hostname or "").lower()
        _ = parts.port  # raises on a bad port, which would otherwise fail on every call
    except ValueError:
        raise ValueError("create_openai_compatible_tier3_provider: `base_url` is not a valid URL") from None
    if not parts.scheme or not parts.netloc:
        raise ValueError("create_openai_compatible_tier3_provider: `base_url` is not a valid URL")
    if parts.username or parts.password:
        raise ValueError(
            "create_openai_compatible_tier3_provider: put credentials in `api_key` or `headers`, not in `base_url`"
        )
    local = host in _LOOPBACK_HOSTS or re.fullmatch(r"127(\.\d{1,3}){3}", host) is not None
    scheme = parts.scheme.lower()
    if scheme == "http" and not local and not allow_insecure_http:
        raise ValueError(
            "create_openai_compatible_tier3_provider: `base_url` must use https "
            "(http is allowed for localhost, or with allow_insecure_http)"
        )
    if scheme not in ("http", "https"):
        raise ValueError("create_openai_compatible_tier3_provider: `base_url` must be http(s)")
    path = parts.path.rstrip("/")
    if not path.endswith("/chat/completions"):
        path = f"{path}/chat/completions"
    return urlunsplit((scheme, parts.netloc, path, parts.query, ""))


def create_openai_compatible_tier3_provider(
    *,
    base_url: str,
    model: str,
    api_key: str | None = None,
    headers: Mapping[str, str] | None = None,
    response_format: Literal["json_schema", "json_object", "none"] = "json_schema",
    max_tokens: int = 300,
    legacy_max_tokens: bool = False,
    temperature: float | None = None,
    retries: int = 1,
    prompt: Prompt | None = None,
    parse: ParseVerdict | None = None,
    client: Any = None,
    allow_insecure_http: bool = False,
) -> LlmTier3Provider:
    """Build a Tier 3 provider for an OpenAI-compatible chat completions endpoint.

    - ``base_url``: e.g. ``"https://api.openai.com/v1"``; ``/chat/completions`` is appended.
    - ``api_key``: sent as ``Authorization: Bearer``. Omit for no header.
    - ``response_format``: ``"json_schema"`` (default), ``"json_object"``, or ``"none"`` for servers
      that support neither.
    - ``max_tokens``: default 300, sent as ``max_completion_tokens`` unless ``legacy_max_tokens``.
    - ``temperature``: not sent unless set (some reasoning models reject it).
    - ``retries``: extra attempts on 429, 5xx and network errors while the deadline allows. Default 1.
    - ``client``: an ``httpx.AsyncClient`` to reuse. Recommended in production: without it each call
      opens a new client (no connection pooling) with httpx defaults, including proxy and CA
      settings from the environment.
    - ``allow_insecure_http``: allow plain http to a non-local host. Tool content and the key then
      travel unencrypted.

    Raises ``ValueError`` on bad options and ``ImportError`` without ``httpx``.
    """
    try:
        import httpx
    except ImportError as e:  # pragma: no cover - depends on the environment
        raise ImportError(
            "create_openai_compatible_tier3_provider needs httpx: pip install 'stackone-defender[llm]'"
        ) from e
    if not base_url or not model:
        raise ValueError("create_openai_compatible_tier3_provider: `base_url` and `model` are required")
    url = _chat_completions_url(base_url, allow_insecure_http)
    if response_format not in _RESPONSE_FORMATS:
        raise ValueError(
            'create_openai_compatible_tier3_provider: `response_format` must be "json_schema", "json_object" or "none"'
        )
    if isinstance(retries, bool) or not isinstance(retries, int) or retries < 0:
        raise ValueError("create_openai_compatible_tier3_provider: `retries` must be an int >= 0")
    req_headers = {"Content-Type": "application/json", **(headers or {})}
    for name, value in req_headers.items():
        if not isinstance(name, str) or not isinstance(value, str) or _HEADER_BAD_RE.search(name + value):
            raise ValueError(
                "create_openai_compatible_tier3_provider: header names and values must be single-line strings"
            )
    if api_key:
        if not isinstance(api_key, str) or _HEADER_BAD_RE.search(api_key):
            raise ValueError("create_openai_compatible_tier3_provider: `api_key` must be a single-line string")
        # One Authorization header: a differently-cased duplicate would be sent too.
        req_headers = {k: v for k, v in req_headers.items() if k.lower() != "authorization"}
        req_headers["Authorization"] = f"Bearer {api_key}"

    async def complete(req: LlmCompletionRequest) -> Mapping[str, Any]:
        # Honor the deadline on its own too, so the provider works without defender's timeout.
        deadline = req.deadline if req.deadline is not None else time.time() + _DEFAULT_SAFETY_TIMEOUT_S
        body: dict[str, Any] = {
            "model": model,
            "messages": req.messages,
            ("max_tokens" if legacy_max_tokens else "max_completion_tokens"): max_tokens,
        }
        if temperature is not None:
            body["temperature"] = temperature
        if response_format == "json_schema":
            body["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "tier3_verdict", "strict": True, "schema": VERDICT_JSON_SCHEMA},
            }
        elif response_format == "json_object":
            body["response_format"] = {"type": "json_object"}

        def retry_delay(attempt: int) -> float | None:
            """The backoff before the next attempt, or None when it wouldn't fit the deadline."""
            delay = 0.25 * (attempt + 1)
            return delay if attempt < retries and time.time() + delay + 0.25 < deadline else None

        async def post(http: Any) -> Mapping[str, Any]:
            attempt = 0
            while True:
                remaining = deadline - time.time()
                if remaining <= 0:
                    raise TimeoutError("request timed out")
                try:
                    # Never follow redirects: they would carry the key and the tool content elsewhere.
                    # httpx's own default timeout is 5s; the deadline is the only limit.
                    resp = await asyncio.wait_for(
                        http.post(
                            url,
                            json=body,
                            headers=req_headers,
                            follow_redirects=False,
                            timeout=httpx.Timeout(remaining),
                        ),
                        remaining,
                    )
                except (TimeoutError, httpx.TimeoutException):
                    raise TimeoutError("request timed out") from None
                except httpx.HTTPError as err:
                    delay = retry_delay(attempt)
                    if delay is not None:
                        attempt += 1
                        await asyncio.sleep(delay)
                        continue
                    # Network error text can include the URL; report only the error type.
                    raise ConnectionError(f"network error {type(err).__name__}") from None
                try:
                    parsed = resp.json()
                except ValueError:
                    parsed = None
                if resp.status_code < 200 or resp.status_code >= 300:
                    code = _error_code_of(parsed)
                    if code in _REFUSAL_CODES:
                        return {"text": "", "refused": True}
                    delay = retry_delay(attempt) if resp.status_code == 429 or resp.status_code >= 500 else None
                    if delay is not None:
                        attempt += 1
                        await asyncio.sleep(delay)
                        continue
                    raise _HttpError(resp.status_code, code)
                return _read_choice(parsed)

        if client is not None:
            return await post(client)
        # httpx defaults: proxy and CA settings from the environment apply. Pass `client` to pool
        # connections or to control them.
        async with httpx.AsyncClient() as http:
            return await post(http)

    return create_llm_tier3_provider(complete=complete, prompt=prompt, parse=parse)


def _read_choice(parsed: Any) -> Mapping[str, Any]:
    choices = parsed.get("choices") if isinstance(parsed, Mapping) else None
    # An empty `choices` list is how some gateways report a safety block, which attacker content can
    # trigger, so it blocks. No `choices` at all is a transport or proxy fault, so on_error decides.
    if isinstance(choices, list) and not choices:
        return {"text": "", "refused": True}
    choice = choices[0] if isinstance(choices, list) else None
    if not isinstance(choice, Mapping):
        raise ValueError("no choice in reply")
    message = choice.get("message") if isinstance(choice.get("message"), Mapping) else {}
    content = message.get("content")
    content = _THINK_RE.sub("", content, count=1) if isinstance(content, str) else ""
    refusal = message.get("refusal")
    # Refusals, content filters, truncated and empty replies are unusable verdicts the attacker can
    # cause, so they block rather than fail open.
    if (
        choice.get("finish_reason") in ("content_filter", "length")
        or (isinstance(refusal, str) and refusal)
        or not content.strip()
    ):
        return {"text": "", "refused": True}
    out: dict[str, Any] = {"text": content}
    u = parsed.get("usage")
    if isinstance(u, Mapping) and all(
        isinstance(u.get(k), int) and not isinstance(u.get(k), bool)
        for k in ("prompt_tokens", "completion_tokens", "total_tokens")
    ):
        out["usage"] = {k: u[k] for k in ("prompt_tokens", "completion_tokens", "total_tokens")}
    return out
