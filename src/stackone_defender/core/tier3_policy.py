"""Tier 3 policy: option parsing, verdict validation, and one guarded provider call.

Port of the policy parts of the TypeScript ``PromptDefense`` (0.9.2): ``allow_override``,
``on_error``, ``decide``, ``timeout_ms``, ``max_concurrency``, ``stop_on_first_block``,
``on_event`` and ``include_raw``.
"""

from __future__ import annotations

import asyncio
import contextlib
import inspect
import logging
import math
import time
import weakref
from collections.abc import Callable, Mapping
from dataclasses import dataclass, fields
from typing import Any, Literal

from ..classifiers.tier3_llm_provider import GENERIC_LLM_PROVIDER_ATTR
from ..types import (
    DefenderMode,
    Tier3ChunkRef,
    Tier3DecisionInput,
    Tier3Event,
    Tier3Provider,
    Tier3Skip,
    Tier3TokenUsage,
    Tier3Verdict,
)
from ..utils.redact import describe_error, redact_error_text
from ..utils.text import from_utf16, js_typeof, truncate

_logger = logging.getLogger("stackone_defender.core.prompt_defense")

FAILURE_STREAK_WARN = 20
# A call that never settles keeps its concurrency slot for at most this many timeouts.
SLOT_CEILING_FACTOR = 10
_MAX_REASON_CHARS = 500
# Fields only the defender may set; never taken from a provider verdict.
_META_FIELDS = frozenset({"errored", "note", "override_applied", "override_suppressed", "decided_by"})
_VERDICT_FIELDS = tuple(f.name for f in fields(Tier3Verdict) if f.name not in _META_FIELDS)

OnError = Literal["tier2", "allow", "block"]
Decide = Callable[[Tier3DecisionInput], str]

# Module-scoped: hosts that build an instance per request would otherwise warn per request.
_generic_override_warned = False


class _MissingProvider:
    """Stands in for a missing provider when on_error="block" asks to fail closed."""

    def classify(self, text: str, *, ctx: Any = None) -> Any:
        raise RuntimeError("No Tier 3 provider registered")


MISSING_PROVIDER = _MissingProvider()


@dataclass
class CallOutcome:
    """One provider call. ``aborted`` = no longer needed (stop_on_first_block), not a failure."""

    kind: Literal["verdict", "failed", "aborted"]
    verdict: Tier3Verdict | None = None
    skip_reason: str | None = None


def _number(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _opt(opts: Mapping[str, Any], snake: str, camel: str) -> Any:
    v = opts.get(snake)
    return opts.get(camel) if v is None else v


class Tier3Policy:
    """The parsed ``tier3`` policy options plus per-instance call state."""

    def __init__(self, opts: Mapping[str, Any], mode: DefenderMode, block_threshold: float | None):
        self.mode = mode
        self.block_threshold = block_threshold
        self.allow_override: bool | float = True  # a float is max_tier2_score
        self.on_error: OnError | None = None
        self.decide: Decide | None = None
        self.timeout_ms: float | None = None
        self.max_concurrency: int | None = None
        self.stop_on_first_block = False
        self.on_event: Callable[[Tier3Event], Any] | None = None
        self.include_raw = True
        self._missing_score_warned = False
        self._decide_warned = False
        self._consecutive_failures = 0
        self._failure_streak_warned = False
        self._consecutive_refusals = 0
        self._refusal_streak_warned = False
        # asyncio primitives bind to one event loop; the sync API runs a new loop per call.
        self._semaphores: weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, asyncio.Semaphore] = (
            weakref.WeakKeyDictionary()
        )
        self._parse(opts)

    # --- options ---------------------------------------------------------------------------

    def _parse(self, opts: Mapping[str, Any]) -> None:
        ao = _opt(opts, "allow_override", "allowOverride")
        if ao is not None:
            if isinstance(ao, bool):
                self.allow_override = ao
            else:
                mx = _opt(ao, "max_tier2_score", "maxTier2Score") if isinstance(ao, Mapping) else None
                if _number(mx) and math.isfinite(mx) and 0 < mx <= 1:
                    self.allow_override = float(mx)
                else:
                    # Fall back to the restrictive setting, so a typo can't widen what an allow undoes.
                    self.allow_override = False
                    _logger.warning(
                        "[defender] invalid tier3.allow_override — must be a bool or "
                        "{max_tier2_score} with 0 < max_tier2_score <= 1. Falling back to False."
                    )
            if self.mode == "tier3_only":
                _logger.warning("[defender] tier3.allow_override has no effect in tier3_only mode.")
        oe = _opt(opts, "on_error", "onError")
        if oe is not None:
            if oe == "tier2" and self.mode == "tier3_only":
                _logger.warning('[defender] tier3.on_error "tier2" is not valid in tier3_only mode. Using "allow".')
                self.on_error = "allow"
            elif oe in ("tier2", "allow", "block"):
                self.on_error = oe
            else:
                _logger.warning(
                    '[defender] invalid tier3.on_error %r — must be "tier2", "allow" or "block". Using the default.',
                    oe,
                )
        decide = opts.get("decide")
        if decide is not None:
            if callable(decide):
                self.decide = decide
            else:
                _logger.warning("[defender] invalid tier3.decide — must be callable. Ignoring it.")
        ms = _opt(opts, "timeout_ms", "timeoutMs")
        if ms is not None:
            if _number(ms) and math.isfinite(ms) and ms > 0:
                self.timeout_ms = float(math.ceil(ms))
            else:
                _logger.warning("[defender] invalid tier3.timeout_ms %s — must be a finite number > 0. No timeout.", ms)
        mc = _opt(opts, "max_concurrency", "maxConcurrency")
        if mc is not None:
            if _number(mc) and (isinstance(mc, int) or math.isfinite(mc)) and math.floor(mc) >= 1:
                self.max_concurrency = math.floor(mc)
            else:
                _logger.warning(
                    "[defender] invalid tier3.max_concurrency %s — must be a finite number >= 1. Unbounded.", mc
                )
        if self.max_concurrency is not None and self.timeout_ms is None:
            _logger.warning(
                "[defender] tier3.max_concurrency is set without tier3.timeout_ms: one provider call that never "
                "settles holds its slot forever and queued calls wait with it. Set timeout_ms."
            )
        self.stop_on_first_block = _opt(opts, "stop_on_first_block", "stopOnFirstBlock") is True
        on_event = _opt(opts, "on_event", "onEvent")
        if callable(on_event):
            self.on_event = on_event
        self.include_raw = _opt(opts, "include_raw", "includeRaw") is not False

    # --- events and health ------------------------------------------------------------------

    def emit(self, event: Tier3Event) -> None:
        if self.on_event is None:
            return
        # A broken observability hook must never affect the verdict.
        with contextlib.suppress(Exception):
            self.on_event(event)

    def _note_failure(self, error_code: str) -> None:
        # A failure breaks a run of refusals, so the refusal warning stays "consecutive".
        self._consecutive_refusals = 0
        self._refusal_streak_warned = False
        self._consecutive_failures += 1
        if self._consecutive_failures >= FAILURE_STREAK_WARN and not self._failure_streak_warned:
            self._failure_streak_warned = True
            _logger.warning(
                "[defender] %s consecutive Tier 3 provider failures. Tier 3 may be misconfigured. Last error: %s",
                self._consecutive_failures,
                error_code,
            )

    def _note_success(self) -> None:
        self._consecutive_failures = 0
        self._failure_streak_warned = False
        self._consecutive_refusals = 0
        self._refusal_streak_warned = False

    def _note_refusal(self) -> None:
        # A refusal is a reply, so it breaks a run of provider failures.
        self._consecutive_failures = 0
        self._failure_streak_warned = False
        self._consecutive_refusals += 1
        if self._consecutive_refusals >= FAILURE_STREAK_WARN and not self._refusal_streak_warned:
            self._refusal_streak_warned = True
            _logger.warning(
                "[defender] %s consecutive Tier 3 refusals or unusable replies (all blocked). Check the model "
                "follows the JSON format and has enough max_tokens.",
                self._consecutive_refusals,
            )

    def warn_if_risky_override(self, provider: Tier3Provider) -> None:
        global _generic_override_warned
        if (
            self.allow_override is True
            and self.mode == "cascade"
            and self.decide is None
            and not _generic_override_warned
            and getattr(provider, GENERIC_LLM_PROVIDER_ATTR, False) is True
        ):
            _generic_override_warned = True
            _logger.warning(
                "[defender] tier3.allow_override is True with a general-purpose LLM provider: a judge talked into "
                "'allow' can undo Tier 2 blocks. Consider allow_override=False or {'max_tier2_score': ...}."
            )

    # --- verdicts ---------------------------------------------------------------------------

    @staticmethod
    def _usage(usage: Any) -> Tier3TokenUsage | None:
        if isinstance(usage, Tier3TokenUsage):
            return usage
        if not isinstance(usage, Mapping):
            return None
        vals = [
            _opt(usage, "prompt_tokens", "promptTokens"),
            _opt(usage, "completion_tokens", "completionTokens"),
            _opt(usage, "total_tokens", "totalTokens"),
        ]
        if not all(isinstance(v, int) and not isinstance(v, bool) for v in vals):
            return None
        return Tier3TokenUsage(prompt_tokens=vals[0], completion_tokens=vals[1], total_tokens=vals[2])

    def validate(self, verdict: Any) -> Tier3Verdict | Tier3Skip:
        """Normalize a provider verdict, or say why it's unusable. Meta fields are never copied."""
        if isinstance(verdict, Tier3Verdict):
            raw = {name: getattr(verdict, name) for name in _VERDICT_FIELDS}
        elif isinstance(verdict, Mapping):
            raw = dict(verdict)
        else:
            return Tier3Skip(skip_reason=f"Tier 3 provider returned non-object verdict: {js_typeof(verdict)}")
        decision = raw.get("decision")
        refused = raw.get("refused")
        refused = refused if isinstance(refused, bool) else None
        # A refusal is attacker-triggerable, so it blocks even without a valid decision.
        if refused is True:
            decision = "block"
        if decision not in ("block", "allow"):
            # Don't echo the value: it is provider output and may quote content.
            return Tier3Skip(
                skip_reason=(
                    f"Tier 3 provider returned invalid decision of type {js_typeof(decision)} "
                    '(expected "block" | "allow")'
                )
            )
        reason = raw.get("reason")
        score = raw.get("score")
        score_ok = _number(score) and math.isfinite(score) and 0 <= score <= 1
        prompt_version = _opt(raw, "prompt_version", "promptVersion")
        latency = _opt(raw, "latency_ms", "latencyMs")
        raw_out = raw.get("raw") if self.include_raw else None
        return Tier3Verdict(
            decision=decision,
            score=float(score) if score_ok else None,
            # Model text can echo a lone surrogate, which breaks UTF-8 logging downstream.
            raw=from_utf16(raw_out) if isinstance(raw_out, str) else raw_out,
            latency_ms=float(latency) if _number(latency) else None,
            usage=self._usage(raw.get("usage")),
            reason=from_utf16(truncate(reason, _MAX_REASON_CHARS)) if isinstance(reason, str) else None,
            refused=refused,
            prompt_version=prompt_version if isinstance(prompt_version, str) else None,
        )

    def is_block(self, verdict: Tier3Verdict) -> bool:
        """Resolve a validated verdict to block/allow (a refusal always blocks)."""
        if verdict.refused is True:
            return True
        if self.block_threshold is None:
            return verdict.decision == "block"
        if verdict.score is not None:
            return verdict.score >= self.block_threshold
        if not self._missing_score_warned:
            self._missing_score_warned = True
            _logger.warning(
                "[defender] tier3.block_threshold=%s is set but the provider did not report a usable score "
                "(missing, or not a number in [0, 1]). Falling back to the provider's decision — the threshold "
                "is not being applied.",
                self.block_threshold,
            )
        return verdict.decision == "block"

    def override_permitted(self, tier2_undamped_score: float | None) -> bool:
        """Whether a Tier-3 allow may undo a Tier-2 block, given Tier 2's undamped top-chunk score."""
        ao = self.allow_override
        if isinstance(ao, bool):
            return ao
        return tier2_undamped_score is not None and tier2_undamped_score < ao

    def run_decide(self, decision_input: Tier3DecisionInput) -> str:
        """Run ``tier3.decide``; a raise or an invalid return means "default"."""
        if self.decide is None:
            return "default"
        try:
            out = self.decide(decision_input)
            if out in ("block", "allow", "default"):
                return out
            raise ValueError(f"invalid return value {type(out).__name__}")
        except Exception as e:
            code = redact_error_text(describe_error(e))
            if not self._decide_warned:
                self._decide_warned = True
                _logger.warning("[defender] tier3.decide failed; using the typed options. %s", code)
            self.emit(Tier3Event(kind="policy_error", mode=decision_input.mode, error_code=code))
            return "default"

    # --- one provider call ------------------------------------------------------------------

    def _semaphore(self) -> asyncio.Semaphore | None:
        if self.max_concurrency is None:
            return None
        loop = asyncio.get_running_loop()
        sem = self._semaphores.get(loop)
        if sem is None:
            sem = self._semaphores[loop] = asyncio.Semaphore(self.max_concurrency)
        return sem

    def _failed(
        self, mode: DefenderMode, chunk: Tier3ChunkRef | None, kind: Literal["error", "timeout"], reason: str
    ) -> CallOutcome:
        self._note_failure(reason)
        self.emit(Tier3Event(kind=kind, mode=mode, chunk=chunk, error_code=reason))
        return CallOutcome(kind="failed", skip_reason=reason)

    async def call(
        self,
        provider: Tier3Provider,
        text: str,
        ctx: dict[str, Any],
        stop: asyncio.Event | None = None,
        on_verdict: Callable[[Tier3Verdict], None] | None = None,
    ) -> CallOutcome:
        """One provider call with the policy applied: concurrency slot, timeout, verdict validation,
        redacted error text, events, failure streaks. Never raises, except when the caller's own task
        is cancelled."""
        mode: DefenderMode = ctx["mode"]
        chunk_ctx = ctx.get("chunk")
        chunk = Tier3ChunkRef(index=chunk_ctx["index"], total=chunk_ctx["total"]) if chunk_ctx else None
        timeout_s = self.timeout_ms / 1000 if self.timeout_ms is not None else None
        sem = self._semaphore()
        stop_task = asyncio.ensure_future(stop.wait()) if stop is not None else None
        # True while this call holds a concurrency slot.
        held = False
        # Held until the provider settled AND this call's outcome is handled, so a stop_on_first_block
        # stop lands before a queued call can take the slot.
        holds = 2

        def release() -> None:
            nonlocal held
            if sem is not None and held:
                held = False
                sem.release()

        def release_one() -> None:
            nonlocal holds
            holds -= 1
            if holds == 0:
                release()

        call: asyncio.Future[Any] | None = None
        try:
            if sem is not None:
                # Waiting for a slot is bounded at the same ceiling a hung call holds its slot for.
                wait_s = timeout_s * SLOT_CEILING_FACTOR if timeout_s is not None else None
                acquire = asyncio.ensure_future(sem.acquire())
                waiters = {acquire} | ({stop_task} if stop_task is not None else set())
                try:
                    await asyncio.wait(waiters, timeout=wait_s, return_when=asyncio.FIRST_COMPLETED)
                finally:
                    # Also on our own cancellation: only a slot actually acquired is released later.
                    if not acquire.done():
                        acquire.cancel()
                    held = acquire.done() and not acquire.cancelled() and acquire.exception() is None
                if held and stop is not None and stop.is_set():
                    release()
                    return CallOutcome(kind="aborted")
                if not held:
                    if stop is not None and stop.is_set():
                        return CallOutcome(kind="aborted")
                    return self._failed(
                        mode, chunk, "timeout", f"Tier 3 call timed out after {wait_s * 1000:g}ms waiting for a slot"
                    )
            if stop is not None and stop.is_set():
                release()
                return CallOutcome(kind="aborted")

            full_ctx = dict(ctx)
            if timeout_s is not None:
                full_ctx["deadline"] = time.time() + timeout_s
            t0 = time.perf_counter()

            async def invoke() -> Any:
                result = provider.classify(text, ctx=full_ctx)
                if inspect.isawaitable(result):
                    result = await result
                return result

            call = asyncio.ensure_future(invoke())
            # Hold the slot until the provider settles, even after we stop waiting, so a slow provider
            # can't push in-flight calls past max_concurrency. A provider that never settles gets its
            # slot back after a ceiling.
            if sem is not None:
                ceiling = (
                    asyncio.get_running_loop().call_later(timeout_s * SLOT_CEILING_FACTOR, release)
                    if timeout_s is not None
                    else None
                )

                def on_settled(_: Any) -> None:
                    if ceiling is not None:
                        ceiling.cancel()
                    release_one()

                call.add_done_callback(on_settled)
            waiters = {call} | ({stop_task} if stop_task is not None else set())
            done, _ = await asyncio.wait(waiters, timeout=timeout_s, return_when=asyncio.FIRST_COMPLETED)
            if call not in done:
                call.cancel()
                call.add_done_callback(_swallow)
                if stop_task is not None and stop_task in done:
                    return CallOutcome(kind="aborted")
                return self._failed(mode, chunk, "timeout", f"Tier 3 provider timed out after {self.timeout_ms:g}ms")
            if call.cancelled():
                return self._failed(mode, chunk, "error", "Tier 3 provider error: cancelled")
            err = call.exception()
            if err is not None:
                return self._failed(
                    mode, chunk, "error", f"Tier 3 provider error: {redact_error_text(describe_error(err))}"
                )
            try:
                validated = self.validate(call.result())
            except Exception as e:
                return self._failed(
                    mode,
                    chunk,
                    "error",
                    f"Tier 3 provider returned an unreadable verdict: {redact_error_text(describe_error(e))}",
                )
            if isinstance(validated, Tier3Skip):
                return self._failed(mode, chunk, "error", validated.skip_reason)
            if on_verdict is not None:
                on_verdict(validated)
            if validated.refused:
                self._note_refusal()
            else:
                self._note_success()
            self.emit(
                Tier3Event(
                    kind="refusal" if validated.refused else "verdict",
                    mode=mode,
                    chunk=chunk,
                    latency_ms=(time.perf_counter() - t0) * 1000,
                    usage=validated.usage,
                )
            )
            return CallOutcome(kind="verdict", verdict=validated)
        finally:
            if stop_task is not None and not stop_task.done():
                stop_task.cancel()
            if call is not None and not call.done():
                # Our own task was cancelled: stop the provider call too.
                call.cancel()
                call.add_done_callback(_swallow)
            if call is None:
                release()
            else:
                release_one()


def _swallow(fut: asyncio.Future[Any]) -> None:
    """Retrieve a late result or exception so asyncio doesn't log it as never retrieved."""
    if not fut.cancelled():
        fut.exception()
