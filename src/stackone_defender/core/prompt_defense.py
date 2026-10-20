"""PromptDefense - Main Entry Point.

The primary class for using the prompt defense framework.
Provides a simple API for defending tool results against prompt injection.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
import math
import time
from dataclasses import dataclass, field, replace
from typing import Any

from ..classifiers.onnx_classifier import BatchTokenStats
from ..classifiers.pattern_detector import PatternDetector, create_pattern_detector
from ..classifiers.tier2_classifier import Tier2Classifier, create_tier2_classifier
from ..classifiers.tier3_orchestrator import get_default_tier3_provider
from ..config import MAX_TRAVERSAL_DEPTH, create_config
from ..sfe.preprocess import SfePredictor, get_default_predictor, sfe_preprocess
from ..types import (
    DefenderMode,
    DefenseResult,
    MultiheadConfig,
    PhaseTimings,
    PromptDefenseConfig,
    RiskLevel,
    Tier1Result,
    Tier2Stats,
    Tier3ChunkSummary,
    Tier3EscalationBand,
    Tier3Provider,
    Tier3Result,
    Tier3Skip,
    Tier3TokenUsage,
    Tier3Verdict,
)
from ..utils.boundary import generate_data_boundary
from .sentence_cleaner import clean_high_risk_content
from .tier3_input import DEFAULT_MAX_CHUNKS, chunk_tier3_input, content_budget, format_records_for_tier3
from .tool_result_sanitizer import ToolResultSanitizer, create_tool_result_sanitizer

_logger = logging.getLogger(__name__)

# Module-scoped (not per-instance): PromptDefense is constructed per-request in
# some hosts, so an instance flag would warn at full request volume.
_tier2_unavailable_warned = False

_DEFAULT_TIER3_BAND = Tier3EscalationBand(lower=0.3, upper=0.85)
_DEFAULT_TIER3_MAX_TEXT_LENGTH = 10000
_TIER3_ON_OVERSIZE = ("skip", "block", "scan_anyway")


@dataclass
class _Tier2Aggregate:
    """Aggregated chunk-level scoring state before the decision branch.

    Populated by :meth:`PromptDefense._tier2_score_strings`. Three pieces of
    state coexist because the decision branch needs them all: per-string
    max scores (for density damping), a global max-main pointer with its
    aux companion (for the aux-veto branch's reported scores), and the
    strongest rule-triggering chunk (for the rule-fire branch's reported
    scores).
    """

    per_string_scores: list[float] = field(default_factory=list)
    # Global max main score across all chunks, pre-density / pre-rule.
    max_main: float | None = None
    max_main_sentence: str | None = None
    # Aux of the chunk that has the global-max main. Surfaced as
    # ``tier2_aux_score`` on the aux-veto branch.
    aux_of_max_main: float | None = None
    # Strongest chunk satisfying the multi-head block rule, if any.
    rule_any_fired: bool = False
    rule_top_main: float = -1.0
    rule_top_aux: float | None = None
    rule_top_chunk: str = ""


@dataclass
class _Tier2Outcome:
    """Final Tier 2 decision surface returned to ``defend_tool_result``."""

    risk: RiskLevel = "low"
    effective_score: float | None = None
    raw_score: float | None = None
    aux_score: float | None = None
    multihead_blocked: bool | None = None
    max_sentence: str | None = None
    skip_reason: str | None = None
    # Cost telemetry — set only when the batched classifier ran on this call.
    phase_timings: PhaseTimings | None = None
    tier2_stats: Tier2Stats | None = None
    cold_load: bool | None = None
    # False when Tier 2 was enabled but the model/runtime failed to load.
    tier2_available: bool | None = None
    # Leaf string values that scored high — the fields the cleaner rewrites.
    high_risk_values: set[str] = field(default_factory=set)


def _extract_strings(
    obj: Any,
    fields: list[str] | None = None,
    depth_flag: dict[str, bool] | None = None,
) -> list[str]:
    """Recursively extract string values from an object for Tier 2.

    If ``fields`` is None or empty, all strings are collected. Otherwise only
    strings under matching dict keys are collected (via full-depth ``collect_all``);
    non-matching keys are traversed recursively without collecting string leaves
    under them (matches post-ENG-12518 TypeScript behavior).
    """
    strings: list[str] = []

    def collect_all(value: Any, depth: int) -> None:
        if depth > MAX_TRAVERSAL_DEPTH:
            if depth_flag is not None:
                depth_flag["hit"] = True
            return
        if isinstance(value, str):
            strings.append(value)
        elif isinstance(value, list):
            for item in value:
                collect_all(item, depth + 1)
        elif isinstance(value, dict):
            for v in value.values():
                collect_all(v, depth + 1)

    if fields is None or len(fields) == 0:
        collect_all(obj, 0)
        return strings

    if isinstance(obj, str):
        strings.append(obj)
        return strings

    field_set = set(fields)

    def traverse(value: Any, depth: int) -> None:
        if depth > MAX_TRAVERSAL_DEPTH:
            if depth_flag is not None:
                depth_flag["hit"] = True
            return
        if isinstance(value, list):
            for item in value:
                traverse(item, depth + 1)
        elif isinstance(value, dict):
            for k, v in value.items():
                if k in field_set:
                    collect_all(v, depth + 1)
                else:
                    traverse(v, depth + 1)

    traverse(obj, 0)
    return strings


def _finite_at_least_one(v: Any) -> bool:
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        return False
    if isinstance(v, int):
        return v >= 1  # math.isfinite overflows on huge ints
    return math.isfinite(v) and math.floor(v) >= 1


_RISK_LEVELS: list[RiskLevel] = ["low", "medium", "high", "critical"]


class PromptDefense:
    """Main API for prompt injection defense."""

    def __init__(
        self,
        *,
        config: dict | None = None,
        enable_tier1: bool = True,
        enable_tier2: bool = True,
        tier2_config: dict | None = None,
        tier2_fields: list[str] | None = None,
        use_sfe: bool | dict[str, Any] = False,
        block_high_risk: bool = False,
        default_risk_level: RiskLevel = "low",
        annotate_boundary: bool = False,
        sanitize_content: bool = True,
        require_tier2: bool = False,
        enable_tier3: bool = False,
        defender_mode: DefenderMode = "cascade",
        tier3: dict[str, Any] | None = None,
    ):
        self._config: PromptDefenseConfig = create_config(config)
        if block_high_risk:
            self._config.block_high_risk = True
        self._tier2_required = require_tier2

        self._tier2_fields = tier2_fields
        self._sfe_enabled = False
        self._sfe_threshold = 0.5
        self._sfe_custom_predictor: SfePredictor | None = None
        if use_sfe is True:
            self._sfe_enabled = True
        elif isinstance(use_sfe, dict):
            self._sfe_enabled = True
            if isinstance(use_sfe.get("threshold"), (int, float)):
                self._sfe_threshold = float(use_sfe["threshold"])
            if use_sfe.get("predictor") is not None:
                self._sfe_custom_predictor = use_sfe["predictor"]

        self._tool_sanitizer: ToolResultSanitizer = create_tool_result_sanitizer(
            risky_fields=self._config.risky_fields,
            traversal=self._config.traversal,
            default_risk_level=default_risk_level,
            use_tier1_classification=enable_tier1,
            cumulative_risk_thresholds=self._config.cumulative_risk_thresholds,
            annotate_boundary=annotate_boundary,
        )
        self._sanitize_content = sanitize_content
        self._annotate_boundary = annotate_boundary

        self._pattern_detector: PatternDetector = create_pattern_detector()
        self._tier2: Tier2Classifier | None = None

        if enable_tier2:
            self._tier2 = create_tier2_classifier(tier2_config)
            # Bug 1 fix: sync the gate's threshold copy with whatever
            # Tier2Classifier resolved. ``Tier2Classifier`` merges hardcoded
            # defaults < model ``classifier_config.json`` < caller-provided
            # ``tier2_config``. Reading back here ensures the block gate at
            # ``self._config.tier2.high_risk_threshold`` matches the
            # ``get_risk_level`` thresholds used inside Tier 2. Without this
            # readback, a model shipping calibrated defaults (e.g. v5 with
            # ``high_risk_threshold = 0.64``) lands ``risk_level = "high"``
            # at score 0.7 but ``allowed = True`` because the gate is still
            # on the library default of 0.8.
            if self._config.tier2 is not None:
                effective = self._tier2.get_config()
                self._config.tier2.high_risk_threshold = float(effective["high_risk_threshold"])
                self._config.tier2.medium_risk_threshold = float(effective["medium_risk_threshold"])

        self._tier3_enabled = enable_tier3
        if defender_mode not in ("cascade", "tier3_only"):
            _logger.warning(
                '[defender] invalid defender_mode %r — must be "cascade" or "tier3_only". '
                'Falling back to "cascade".',
                defender_mode,
            )
            defender_mode = "cascade"
        self._defender_mode: DefenderMode = defender_mode
        self._tier3_custom_provider: Tier3Provider | None = None
        self._tier3_band = _DEFAULT_TIER3_BAND
        self._tier3_max_text_length = _DEFAULT_TIER3_MAX_TEXT_LENGTH
        self._tier3_missing_provider_warned = False
        self._tier3_block_threshold: float | None = None
        self._tier3_missing_score_warned = False
        self._tier3_on_oversize = "skip"
        self._tier3_max_chunks = DEFAULT_MAX_CHUNKS
        tier3_opts = tier3 or {}
        if tier3_opts.get("provider") is not None:
            self._tier3_custom_provider = tier3_opts["provider"]
        max_text_length = tier3_opts.get("max_text_length", tier3_opts.get("maxTextLength"))
        if max_text_length is not None:
            # Validate the floored value: a cap in (0, 1) floors to 0, which would disable Tier 3.
            if _finite_at_least_one(max_text_length):
                self._tier3_max_text_length = math.floor(max_text_length)
            else:
                _logger.warning(
                    "[defender] invalid tier3.max_text_length %s — must be a finite number >= 1. "
                    "Falling back to default %s.",
                    max_text_length,
                    _DEFAULT_TIER3_MAX_TEXT_LENGTH,
                )
        on_oversize = tier3_opts.get("on_oversize", tier3_opts.get("onOversize"))
        if on_oversize is not None:
            if on_oversize in _TIER3_ON_OVERSIZE:
                self._tier3_on_oversize = on_oversize
            else:
                _logger.warning(
                    '[defender] invalid tier3.on_oversize %r — must be "skip", "block", or "scan_anyway". '
                    'Falling back to default "skip".',
                    on_oversize,
                )
        max_chunks = tier3_opts.get("max_chunks", tier3_opts.get("maxChunks"))
        if max_chunks is not None:
            if _finite_at_least_one(max_chunks):
                self._tier3_max_chunks = math.floor(max_chunks)
            else:
                _logger.warning(
                    "[defender] invalid tier3.max_chunks %s — must be a finite number >= 1. "
                    "Falling back to default %s.",
                    max_chunks,
                    DEFAULT_MAX_CHUNKS,
                )
        escalation_band = tier3_opts.get("escalation_band", tier3_opts.get("escalationBand"))
        if escalation_band is not None:
            lower = escalation_band.get("lower")
            upper = escalation_band.get("upper")
            if (
                isinstance(lower, (int, float))
                and isinstance(upper, (int, float))
                and math.isfinite(lower)
                and math.isfinite(upper)
                and 0 <= lower < upper <= 1
            ):
                self._tier3_band = Tier3EscalationBand(lower=float(lower), upper=float(upper))
            else:
                _logger.warning(
                    "[defender] invalid tier3.escalation_band { lower: %s, upper: %s } — "
                    "must satisfy 0 <= lower < upper <= 1. Falling back to default { lower: 0.3, upper: 0.85 }.",
                    lower,
                    upper,
                )
        block_threshold = tier3_opts.get("block_threshold", tier3_opts.get("blockThreshold"))
        if block_threshold is not None:
            if (
                isinstance(block_threshold, (int, float))
                and not isinstance(block_threshold, bool)
                and math.isfinite(block_threshold)
                and 0 <= block_threshold <= 1
            ):
                self._tier3_block_threshold = float(block_threshold)
            else:
                _logger.warning(
                    "[defender] invalid tier3.block_threshold %s — must be a finite number in [0, 1]. "
                    "Falling back to the provider's decision.",
                    block_threshold,
                )

    def warmup_tier2(self) -> None:
        if self._tier2:
            try:
                self._tier2.warmup()
            except Exception as e:
                self._handle_tier2_unavailable(e)
        if self._sfe_enabled and self._sfe_custom_predictor is None:
            predictor = get_default_predictor()
            if predictor is None:
                _logger.warning(
                    "[defender] SFE predictor unavailable at warmup; "
                    "calls with use_sfe enabled will pass payloads through unfiltered."
                )

    def is_tier2_ready(self) -> bool:
        return self._tier2.is_ready() if self._tier2 else False

    def _handle_tier2_unavailable(self, err: Exception) -> None:
        """Tier 2 enabled but the model/runtime failed to load. Fail closed when
        require_tier2 is set; otherwise warn once per process and continue
        Tier-1-only (fail open)."""
        if self._tier2_required:
            raise RuntimeError(
                f"[defender] Tier 2 is required (require_tier2=True) but the model/runtime "
                f"failed to load: {err}. Install the optional dependencies with "
                f"`pip install stackone-defender[onnx]`."
            )
        global _tier2_unavailable_warned
        if not _tier2_unavailable_warned:
            _tier2_unavailable_warned = True
            _logger.warning(
                "[defender] Tier 2 unavailable (model/runtime failed to load); "
                "continuing Tier-1-only. Reason: %s",
                err,
            )

    @staticmethod
    def _coverage_degraded(metadata: Any, depth_flag: dict[str, bool]) -> bool | None:
        """True when Tier 1 detection coverage was reduced (depth/size limit hit,
        or a wide payload's analysis was capped). Content is still returned in full."""
        sm = metadata.size_metrics
        if depth_flag.get("hit") or metadata.analysis_truncated or sm.size_limit_hit or sm.depth_limit_hit:
            return True
        return None

    def _resolve_tier3_provider(self) -> Tier3Provider | None:
        return self._tier3_custom_provider or get_default_tier3_provider()

    @staticmethod
    def _parse_tier3_usage(usage: Any) -> Tier3TokenUsage | None:
        if usage is None or not isinstance(usage, dict):
            return None
        prompt_tokens = usage.get("prompt_tokens", usage.get("promptTokens"))
        completion_tokens = usage.get("completion_tokens", usage.get("completionTokens"))
        total_tokens = usage.get("total_tokens", usage.get("totalTokens"))
        if not all(isinstance(value, int) for value in (prompt_tokens, completion_tokens, total_tokens)):
            return None
        return Tier3TokenUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
        )

    @staticmethod
    def _normalize_tier3_score(score: Any) -> float | None:
        # Public contract is score in [0, 1]; drop anything else (wrong type,
        # non-finite, out of range) rather than leak it to consumers or the
        # block gate. bool is excluded since it is not a numeric score.
        if isinstance(score, bool) or not isinstance(score, (int, float)):
            return None
        if not math.isfinite(score) or score < 0 or score > 1:
            return None
        return float(score)

    @staticmethod
    def _validate_tier3_verdict(verdict: Any) -> Tier3Verdict | Tier3Skip:
        if isinstance(verdict, Tier3Verdict):
            if verdict.decision in ("block", "allow"):
                # errored/note are set by the defender, never by a provider.
                return replace(
                    verdict, score=PromptDefense._normalize_tier3_score(verdict.score), errored=None, note=None
                )
            return Tier3Skip(
                skip_reason=(
                    f'Tier 3 provider returned invalid decision: {verdict.decision!r} '
                    '(expected "block" | "allow")'
                )
            )
        if verdict is None or not isinstance(verdict, dict):
            return Tier3Skip(
                skip_reason=f"Tier 3 provider returned non-object verdict: {type(verdict).__name__}"
            )
        decision = verdict.get("decision")
        if decision not in ("block", "allow"):
            return Tier3Skip(
                skip_reason=f'Tier 3 provider returned invalid decision: {decision!r} (expected "block" | "allow")'
            )
        return Tier3Verdict(
            decision=decision,
            score=PromptDefense._normalize_tier3_score(verdict.get("score")),
            raw=verdict.get("raw"),
            latency_ms=verdict.get("latency_ms", verdict.get("latencyMs")),
            usage=PromptDefense._parse_tier3_usage(verdict.get("usage")),
        )

    def _is_tier3_block(self, verdict: Tier3Verdict) -> bool:
        """Resolve a validated verdict to block/allow.

        With ``block_threshold`` unset this is the model's ``decision`` (its
        argmax, an implicit 0.5 cut). With a threshold set we decide on ``score``
        (P(block)) instead, which is what makes any other operating point
        reachable. A threshold with no usable score falls back to ``decision``
        (warned once) rather than taking the tier offline.
        """
        if self._tier3_block_threshold is None:
            return verdict.decision == "block"
        # Validation already dropped any score outside [0, 1], so a non-None
        # score here is usable as P(block).
        if verdict.score is not None:
            return verdict.score >= self._tier3_block_threshold
        if not self._tier3_missing_score_warned:
            self._tier3_missing_score_warned = True
            _logger.warning(
                "[defender] tier3.block_threshold=%s is set but the provider did not report a usable "
                "score (missing, or not a number in [0, 1]). Falling back to the provider's decision — "
                "the threshold is not being applied.",
                self._tier3_block_threshold,
            )
        return verdict.decision == "block"

    @staticmethod
    async def _invoke_tier3_classify(provider: Tier3Provider, text: str, tool_name: str) -> Any:
        ctx = {"toolName": tool_name}
        result = provider.classify(text, ctx=ctx)
        if inspect.isawaitable(result):
            return await result
        return result

    @staticmethod
    def _tier1_metadata(sanitized) -> tuple[list[str], list[str], dict]:
        prm = sanitized.metadata.patterns_removed_by_field
        mbf = sanitized.metadata.methods_by_field
        detections = list(dict.fromkeys(p for patterns in prm.values() for p in patterns))
        active_methods = {"role_stripping", "pattern_removal", "encoding_detection"}
        fields_sanitized = [
            field_name for field_name, methods in mbf.items()
            if any(m in active_methods for m in methods)
        ]
        return detections, fields_sanitized, prm

    async def _maybe_tier3_cascade(
        self,
        tier2: _Tier2Outcome,
        tool_name: str,
    ) -> tuple[Tier3Result | None, bool | None]:
        """Run Tier 3 cascade escalation when Tier 2 score is in the gray band."""
        if not (self._tier3_enabled and self._defender_mode == "cascade"):
            return None, None
        eff = tier2.effective_score
        if eff is None or not tier2.max_sentence:
            return None, None
        if eff < self._tier3_band.lower or eff >= self._tier3_band.upper:
            return None, None

        provider = self._resolve_tier3_provider()
        if provider is None:
            if not self._tier3_missing_provider_warned:
                self._tier3_missing_provider_warned = True
                _logger.warning(
                    "[defender] enable_tier3=true but no Tier 3 provider is registered. "
                    "Cascade will skip Tier 3 escalation. Call set_default_tier3_provider() at app startup."
                )
            return Tier3Skip(skip_reason="No Tier 3 provider registered"), None

        max_sentence = tier2.max_sentence
        bounded = (
            max_sentence[: self._tier3_max_text_length]
            if len(max_sentence) > self._tier3_max_text_length
            else max_sentence
        )
        try:
            raw = await self._invoke_tier3_classify(provider, bounded, tool_name)
            validated = self._validate_tier3_verdict(raw)
            if isinstance(validated, Tier3Skip):
                return validated, None
            return validated, self._is_tier3_block(validated)
        except Exception as e:
            return Tier3Skip(skip_reason=f"Tier 3 provider error: {e}"), None

    @staticmethod
    def _finalize_allowed_and_risk(
        *,
        detections: list[str],
        tier1_flagged: list[str],
        tier2_has_threat: bool,
        tier2_idx: int,
        tier1_idx: int,
        risk_level: RiskLevel,
        block_high_risk: bool,
        tier3_override_block: bool | None,
    ) -> tuple[RiskLevel, bool]:
        tier3_overrode_to_allow = tier3_override_block is False
        tier3_overrode_to_block = tier3_override_block is True

        if tier3_overrode_to_block and _RISK_LEVELS.index(risk_level) < _RISK_LEVELS.index("high"):
            risk_level = "high"
        elif tier3_overrode_to_allow and tier2_idx > tier1_idx:
            risk_level = _RISK_LEVELS[tier1_idx]

        has_threats = (
            bool(detections)
            or bool(tier1_flagged)
            or (tier2_has_threat and not tier3_overrode_to_allow)
            or tier3_overrode_to_block
        )
        allowed = (
            not block_high_risk
            or not has_threats
            or risk_level not in ("high", "critical")
        )
        return risk_level, allowed

    async def _review_tier3_chunk(self, provider: Tier3Provider, text: str, tool_name: str) -> Tier3Result:
        try:
            raw = await self._invoke_tier3_classify(provider, text, tool_name)
        except asyncio.CancelledError as e:
            # Our own task being cancelled must propagate; a provider's internal cancel is a chunk failure.
            task = asyncio.current_task()
            if task is not None and task.cancelling():
                raise
            return Tier3Skip(skip_reason=f"Tier 3 provider error: cancelled {e}".rstrip())
        except Exception as e:
            return Tier3Skip(skip_reason=f"Tier 3 provider error: {e}")
        return self._validate_tier3_verdict(raw)

    async def _run_tier3_only(
        self,
        value: Any,
        provider: Tier3Provider,
        tool_name: str,
        depth_flag: dict[str, Any],
        start_time: float,
    ) -> DefenseResult:
        # ENG-2455: record-oriented `field: value` input, so bare values keep field context.
        # ENG-1339: serialize up to a coverage ceiling, chunk it, and review every chunk in parallel
        # (union of blocks) instead of truncating: a fixed truncation point is easy to evade.
        verdict: Tier3Verdict | None = None
        skip_reason: str | None = None
        # A policy block (on_oversize), as opposed to a chunk verdict block.
        oversize_block = False
        chunk_summary: Tier3ChunkSummary | None = None
        # The payload couldn't be serialized: un-analyzable input fails closed in strict mode.
        payload_error = False
        chunk_failures = 0
        chunk_total = 0
        first_chunk_failure: str | None = None

        per_chunk = self._tier3_max_text_length
        # The ceiling uses the per-chunk content budget (minus the carried overlap), so the content
        # typically fits in max_chunks chunks; the reviewer still sees <= per_chunk chars per call.
        ceiling = self._tier3_max_chunks * content_budget(per_chunk)
        joined = ""
        try:
            joined = format_records_for_tier3(value, depth_flag, ceiling, self._config.traversal.max_size)
        except Exception as e:
            payload_error = True
            skip_reason = f"Tier 3 serialization error: {e}"
        # A depth cut also hides content from the reviewer, so it counts as oversize.
        oversize = depth_flag.get("budget_exceeded") is True or depth_flag["hit"] is True

        if payload_error:
            pass
        elif oversize and self._tier3_on_oversize == "block":
            # Checked before the empty case so "block" fails closed even when nothing fit.
            oversize_block = True
            depth_flag["coverage_degraded"] = True
            skip_reason = (
                f"Tool result too large to fully review (> {self._tier3_max_chunks} chunks) — blocked by onOversize"
            )
        elif not joined:
            if oversize:
                depth_flag["coverage_degraded"] = True
            if oversize and self._tier3_on_oversize == "scan_anyway":
                oversize_block = True
                skip_reason = "Tool result too large to review — no content fit within the ceiling"
            elif skip_reason is None:
                skip_reason = (
                    "Tool result too large to review — no content fit within the ceiling"
                    if oversize
                    else "No reviewable string content emitted from tool result"
                )
        else:
            # Always review what fit, so a fitting injection is never dropped; on_oversize governs
            # only the overflow.
            chunks = chunk_tier3_input(joined, per_chunk)
            results = await asyncio.gather(*(self._review_tier3_chunk(provider, c, tool_name) for c in chunks))
            blocked_count = 0
            blocking_chunk: int | None = None
            reviewed = 0
            for idx, res in enumerate(results):
                if isinstance(res, Tier3Skip):
                    chunk_failures += 1
                    if first_chunk_failure is None:
                        first_chunk_failure = res.skip_reason
                    continue
                reviewed += 1
                is_block = self._is_tier3_block(res)
                if is_block:
                    blocked_count += 1
                    if blocking_chunk is None:
                        blocking_chunk = idx
                # Representative: a blocker if any, else the highest-score allow.
                if verdict is None:
                    verdict = res
                else:
                    rep_block = self._is_tier3_block(verdict)
                    if (is_block and not rep_block) or (
                        is_block == rep_block and (res.score or 0) > (verdict.score or 0)
                    ):
                        verdict = res
            chunk_total = len(chunks)
            chunk_summary = Tier3ChunkSummary(
                chunks=chunk_total,
                blocked=blocked_count,
                blocking_chunk=blocking_chunk,
                oversize=True if oversize else None,
            )
            if reviewed == 0:
                # Every chunk failed: provider outage or bad verdicts. Fails open.
                if skip_reason is None:
                    skip_reason = first_chunk_failure or "Tier 3 skipped: no chunk produced a verdict"
                depth_flag["coverage_degraded"] = True
            elif chunk_failures > 0:
                depth_flag["coverage_degraded"] = True
            if oversize:
                depth_flag["coverage_degraded"] = True
                # The overflow went unreviewed: scan_anyway fails closed on it unless a chunk blocked.
                if not (verdict is not None and self._is_tier3_block(verdict)) and self._tier3_on_oversize == "scan_anyway":
                    oversize_block = True
                    skip_reason = f"Tool result exceeds review ceiling; {chunk_total} chunks scanned, overflow unreviewed"

        # Tier 1 runs for `detections` metadata only (the LLM decides in tier3_only). Its walk is
        # uncapped, so a failure here degrades coverage but doesn't override a verdict.
        patterns_by_field: dict[str, list[str]] = {}
        detections: list[str] = []
        sanitized_content: Any = value
        analysis_degraded = False
        try:
            sanitized = self._tool_sanitizer.sanitize(value, tool_name=tool_name)
            detections, _tier1_flagged, patterns_by_field = self._tier1_metadata(sanitized)
            sanitized_content = sanitized.sanitized
            analysis_degraded = bool(self._coverage_degraded(sanitized.metadata, {"hit": False}))
        except Exception as e:
            analysis_degraded = True
            if skip_reason is None:
                skip_reason = f"Tier 1 metadata error: {e}"

        blocked = oversize_block or (verdict is not None and self._is_tier3_block(verdict))
        risk_level: RiskLevel = "high" if blocked or payload_error else "low"
        # block_high_risk=False always allows; strict mode fails closed on a payload error.
        allowed = not self._config.block_high_risk or (not blocked and not payload_error)

        errored = True if chunk_failures > 0 else None
        tier3_result: Tier3Result
        if verdict is not None:
            # skip_reason isn't reported with a verdict, so say why the outcome isn't the verdict's alone.
            notes: list[str] = []
            if oversize_block and skip_reason:
                notes.append(skip_reason)
            if chunk_failures > 0:
                failed = f"{chunk_failures} of {chunk_total} chunks failed"
                notes.append(f"{failed}: {first_chunk_failure}" if first_chunk_failure else failed)
            tier3_result = replace(verdict, errored=errored, note="; ".join(notes) or None)
        else:
            tier3_result = Tier3Skip(skip_reason=skip_reason or "Tier 3 skipped", errored=errored)

        return DefenseResult(
            allowed=allowed,
            risk_level=risk_level,
            sanitized=sanitized_content,
            detections=detections,
            fields_sanitized=[],
            patterns_by_field=patterns_by_field,
            detected_field_count=len(patterns_by_field),
            tier3=tier3_result,
            fields_dropped=[],
            # The depth-limit signal only; budget drops set coverage_degraded.
            truncated_at_depth=depth_flag["hit"] or None,
            latency_ms=(time.perf_counter() - start_time) * 1000,
            coverage_degraded=(
                True
                if depth_flag["hit"] or depth_flag.get("coverage_degraded") or analysis_degraded or payload_error
                else None
            ),
            tier3_chunk_summary=chunk_summary,
        )

    def defend_tool_result(self, value: Any, tool_name: str) -> DefenseResult:
        """Defend a tool result using Tier 1 and optionally Tier 2 / Tier 3 classification.

        When SFE is enabled, ``fields_dropped`` lists paths excluded from **Tier 2**
        string extraction only; the returned ``sanitized`` payload is still Tier 1 output
        from the **original** tool value (SFE does not remove fields from the returned object).

        When ``enable_tier3`` is on, this delegates to :meth:`defend_tool_result_async`
        via ``asyncio.run``. Call that method directly from async code (e.g. FastAPI).
        """
        if self._tier3_enabled:
            try:
                asyncio.get_running_loop()
            except RuntimeError:
                return asyncio.run(self.defend_tool_result_async(value, tool_name))
            raise RuntimeError(
                "defend_tool_result() cannot call Tier 3 from a running event loop; "
                "use: await defense.defend_tool_result_async(value, tool_name)"
            )
        return self._defend_tool_result_sync(value, tool_name)

    async def defend_tool_result_async(self, value: Any, tool_name: str) -> DefenseResult:
        """Async defense path — required when Tier 3 is enabled inside a running event loop."""
        start_time = time.perf_counter()
        depth_flag = {"hit": False}

        if self._tier3_enabled and self._defender_mode == "tier3_only":
            provider = self._resolve_tier3_provider()
            if provider is not None:
                return await self._run_tier3_only(value, provider, tool_name, depth_flag, start_time)
            if not self._tier3_missing_provider_warned:
                self._tier3_missing_provider_warned = True
                _logger.warning(
                    "[defender] defender_mode=tier3_only but no Tier 3 provider is registered. "
                    "Falling back to Tier 1 + Tier 2. Call set_default_tier3_provider() at app startup."
                )

        return await self._defend_tool_result_async_impl(
            value, tool_name, start_time=start_time, depth_flag=depth_flag
        )

    async def _defend_tool_result_async_impl(
        self,
        value: Any,
        tool_name: str,
        *,
        start_time: float,
        depth_flag: dict[str, bool],
    ) -> DefenseResult:
        sfe_filtered_value: Any = value
        fields_dropped: list[str] = []
        if self._sfe_enabled:
            try:
                predictor = self._sfe_custom_predictor or get_default_predictor()
                if predictor is not None:
                    pre = sfe_preprocess(value, {"predictor": predictor, "threshold": self._sfe_threshold})
                    sfe_filtered_value = pre.filtered
                    fields_dropped = pre.dropped
                    if pre.truncated_at_depth:
                        depth_flag["hit"] = True
            except Exception as e:
                _logger.warning(
                    "[defender] SFE preprocessing failed; continuing without filtering. Reason: %s",
                    e,
                )

        # Own the boundary here so the sentence-cleaner can wrap its cleaned fields
        # with the same markers the sanitizer used.
        boundary = generate_data_boundary() if self._annotate_boundary else None

        t_tier1_start = time.perf_counter()
        sanitized = self._tool_sanitizer.sanitize(value, tool_name=tool_name, boundary=boundary)
        detections, tier1_flagged, prm = self._tier1_metadata(sanitized)
        tier1_ms = (time.perf_counter() - t_tier1_start) * 1000

        tier2 = (
            self._evaluate_tier2(self._tier2, sfe_filtered_value, depth_flag)
            if self._tier2 is not None
            else _Tier2Outcome()
        )

        tier3_result, tier3_override_block = await self._maybe_tier3_cascade(tier2, tool_name)

        tier1_idx = _RISK_LEVELS.index(sanitized.metadata.overall_risk_level)
        tier2_idx = _RISK_LEVELS.index(tier2.risk)
        risk_level = _RISK_LEVELS[max(tier1_idx, tier2_idx)]

        if tier2.multihead_blocked is True:
            tier2_has_threat = True
        elif tier2.multihead_blocked is False:
            tier2_has_threat = False
        else:
            tier2_has_threat = (
                tier2.effective_score is not None
                and tier2.effective_score >= self._config.tier2.high_risk_threshold
            )

        risk_level, allowed = self._finalize_allowed_and_risk(
            detections=detections,
            tier1_flagged=tier1_flagged,
            tier2_has_threat=tier2_has_threat,
            tier2_idx=tier2_idx,
            tier1_idx=tier1_idx,
            risk_level=risk_level,
            block_high_risk=self._config.block_high_risk,
            tier3_override_block=tier3_override_block,
        )

        # sanitized is a sentence-cleaned copy of the detect-only payload's high-risk
        # fields (unless sanitize_content is off). fields_sanitized reports the fields
        # the cleaner actually changed.
        original = sanitized.sanitized
        if (
            self._sanitize_content
            and self._tier2 is not None
            and risk_level in ("high", "critical")
            and tier2.high_risk_values
        ):
            cleaned, cleaned_fields = clean_high_risk_content(
                original,
                tier2.high_risk_values,
                self._tier2,
                self._config.tier2.high_risk_threshold,
                boundary,
            )
        else:
            cleaned, cleaned_fields = original, []

        return DefenseResult(
            allowed=allowed,
            risk_level=risk_level,
            sanitized=cleaned,
            detections=detections,
            fields_sanitized=cleaned_fields,
            patterns_by_field=prm,
            detected_field_count=len(prm),
            tier2_score=tier2.effective_score,
            tier2_raw_score=tier2.raw_score,
            tier2_aux_score=tier2.aux_score,
            tier2_multihead_blocked=tier2.multihead_blocked,
            tier2_skip_reason=tier2.skip_reason,
            max_sentence=tier2.max_sentence,
            tier3=tier3_result,
            fields_dropped=fields_dropped,
            truncated_at_depth=depth_flag["hit"] or None,
            latency_ms=(time.perf_counter() - start_time) * 1000,
            phase_timings=tier2.phase_timings,
            tier2_stats=tier2.tier2_stats,
            tier1_ms=tier1_ms,
            cold_load=tier2.cold_load,
            tier2_available=tier2.tier2_available,
            coverage_degraded=self._coverage_degraded(sanitized.metadata, depth_flag),
        )

    def _defend_tool_result_sync(
        self,
        value: Any,
        tool_name: str,
        *,
        start_time: float | None = None,
        depth_flag: dict[str, bool] | None = None,
    ) -> DefenseResult:
        if start_time is None:
            start_time = time.perf_counter()
        if depth_flag is None:
            depth_flag = {"hit": False}

        sfe_filtered_value: Any = value
        fields_dropped: list[str] = []
        if self._sfe_enabled:
            try:
                predictor = self._sfe_custom_predictor or get_default_predictor()
                if predictor is not None:
                    pre = sfe_preprocess(value, {"predictor": predictor, "threshold": self._sfe_threshold})
                    sfe_filtered_value = pre.filtered
                    fields_dropped = pre.dropped
                    if pre.truncated_at_depth:
                        depth_flag["hit"] = True
            except Exception as e:
                _logger.warning(
                    "[defender] SFE preprocessing failed; continuing without filtering. Reason: %s",
                    e,
                )

        # Own the boundary so the sentence-cleaner wraps cleaned fields the same way.
        boundary = generate_data_boundary() if self._annotate_boundary else None

        # Tier 1: pattern-based sanitization on the original payload (matches TS 0.6.3).
        t_tier1_start = time.perf_counter()
        sanitized = self._tool_sanitizer.sanitize(value, tool_name=tool_name, boundary=boundary)

        # Collect Tier 1 metadata
        prm = sanitized.metadata.patterns_removed_by_field
        mbf = sanitized.metadata.methods_by_field
        detections = list(dict.fromkeys(p for patterns in prm.values() for p in patterns))

        active_methods = {"role_stripping", "pattern_removal", "encoding_detection"}
        tier1_flagged = [
            field for field, methods in mbf.items()
            if any(m in active_methods for m in methods)
        ]
        tier1_ms = (time.perf_counter() - t_tier1_start) * 1000

        # Tier 2 runs on the SFE-filtered view (full value if SFE off) and
        # produces a self-contained outcome -- skip reason or the
        # (effective, raw, aux, blocked, max_sentence, risk) tuple. The
        # decision branches (single-head density, multi-head rule fire,
        # multi-head aux veto) live in :meth:`_tier2_finalize`; see
        # :class:`_Tier2Aggregate` for the score variable taxonomy.
        # The classifier is passed in explicitly so the helpers don't
        # need to ``assert self._tier2 is not None`` -- those asserts
        # would be stripped under PYTHONOPTIMIZE.
        tier2 = (
            self._evaluate_tier2(self._tier2, sfe_filtered_value, depth_flag)
            if self._tier2 is not None
            else _Tier2Outcome()
        )

        # Combine risk levels (take the higher of Tier 1 and Tier 2)
        tier1_idx = _RISK_LEVELS.index(sanitized.metadata.overall_risk_level)
        tier2_idx = _RISK_LEVELS.index(tier2.risk)
        risk_level = _RISK_LEVELS[max(tier1_idx, tier2_idx)]

        # Three-way Tier 2 threat derivation. In multi-head mode the rule
        # replaces the threshold check: a flagged chunk under
        # ``main >= main_thr AND aux < aux_thr`` is a Tier 2 threat; aux veto
        # suppresses the threshold-based Tier 2 signal entirely.
        if tier2.multihead_blocked is True:
            tier2_has_threat = True
        elif tier2.multihead_blocked is False:
            tier2_has_threat = False
        else:
            tier2_has_threat = (
                tier2.effective_score is not None
                and tier2.effective_score >= self._config.tier2.high_risk_threshold
            )

        # Threat signals: Tier 1 detections, Tier 1 sanitization methods, or
        # Tier 2 above-threshold (subject to multi-head veto).
        risk_level, allowed = self._finalize_allowed_and_risk(
            detections=detections,
            tier1_flagged=tier1_flagged,
            tier2_has_threat=tier2_has_threat,
            tier2_idx=tier2_idx,
            tier1_idx=tier1_idx,
            risk_level=risk_level,
            block_high_risk=self._config.block_high_risk,
            tier3_override_block=None,
        )

        # sanitized is the sentence-cleaned copy of the detect-only payload.
        # fields_sanitized reports the fields the cleaner actually changed.
        original = sanitized.sanitized
        if (
            self._sanitize_content
            and self._tier2 is not None
            and risk_level in ("high", "critical")
            and tier2.high_risk_values
        ):
            cleaned, cleaned_fields = clean_high_risk_content(
                original,
                tier2.high_risk_values,
                self._tier2,
                self._config.tier2.high_risk_threshold,
                boundary,
            )
        else:
            cleaned, cleaned_fields = original, []

        return DefenseResult(
            allowed=allowed,
            risk_level=risk_level,
            sanitized=cleaned,
            detections=detections,
            fields_sanitized=cleaned_fields,
            patterns_by_field=prm,
            detected_field_count=len(prm),
            tier2_score=tier2.effective_score,
            tier2_raw_score=tier2.raw_score,
            tier2_aux_score=tier2.aux_score,
            tier2_multihead_blocked=tier2.multihead_blocked,
            tier2_skip_reason=tier2.skip_reason,
            max_sentence=tier2.max_sentence,
            fields_dropped=fields_dropped,
            truncated_at_depth=depth_flag["hit"] or None,
            latency_ms=(time.perf_counter() - start_time) * 1000,
            phase_timings=tier2.phase_timings,
            tier2_stats=tier2.tier2_stats,
            tier1_ms=tier1_ms,
            cold_load=tier2.cold_load,
            tier2_available=tier2.tier2_available,
            coverage_degraded=self._coverage_degraded(sanitized.metadata, depth_flag),
        )

    # ------------------------------------------------------------------
    # Tier 2 helpers
    # ------------------------------------------------------------------
    #
    # ``defend_tool_result`` orchestrates Tier 1 + Tier 2. The Tier 2 path
    # is split into five focused steps below to keep that orchestration
    # readable and to let each branch be unit-tested in isolation.

    def _evaluate_tier2(
        self,
        tier2: Tier2Classifier,
        sfe_filtered_value: Any,
        depth_flag: dict[str, bool],
    ) -> _Tier2Outcome:
        """Run the Tier 2 pipeline end to end for one tool result.

        ``tier2`` is passed explicitly (rather than read from
        ``self._tier2``) so the type system can prove non-``None``
        without runtime asserts; asserts would be stripped under
        ``PYTHONOPTIMIZE``.
        """
        out = _Tier2Outcome()

        # Sample cold-start BEFORE warmup, then load the model. A load failure
        # (missing optional deps) is a hard "Tier 2 unavailable" — fail closed when
        # require_tier2, else warn once and continue Tier-1-only.
        was_cold = not tier2.is_ready()
        try:
            tier2.warmup()
        except Exception as e:
            out.tier2_available = False
            out.skip_reason = f"Tier 2 unavailable (model/runtime failed to load): {e}"
            self._handle_tier2_unavailable(e)
            return out

        fields_for_tier2 = (
            self._tier2_fields
            if self._tier2_fields is not None
            else self._config.tier2.tier2_fields
        )
        strings = [
            s
            for s in _extract_strings(sfe_filtered_value, fields_for_tier2, depth_flag)
            if len(s) > 0
        ]
        if not strings:
            scoped = fields_for_tier2 is not None and len(fields_for_tier2) > 0
            out.skip_reason = (
                "No strings found in tier2_fields"
                if scoped
                else "No strings extracted from tool result"
            )
            return out

        # Phase 1: prepare chunks (warmup + tokenize + pack).
        t_prep_start = time.perf_counter()
        all_chunks, string_ranges, skip_reasons = self._tier2_build_chunks(tier2, strings)
        if not all_chunks:
            out.skip_reason = (
                "All strings skipped by classifier"
                if not skip_reasons
                else f"All strings skipped by classifier: {'; '.join(sorted(skip_reasons))}"
            )
            return out

        # Phase 2: one batched (deduped) inference over all chunks.
        t_infer_start = time.perf_counter()
        # Set now (before the failure early-return) so cold_load is a bool
        # whenever inference was attempted — success or failure (TS 0.7.4 parity).
        out.cold_load = was_cold
        stats = BatchTokenStats()
        multihead_cfg = tier2.get_multihead_config()
        all_scores, all_pairs, infer_skip, unique_count = self._tier2_run_inference(
            tier2, all_chunks, multihead_cfg, stats
        )
        if infer_skip is not None:
            out.skip_reason = infer_skip
        if all_scores is None:
            return out
        t_agg_start = time.perf_counter()

        # Phase 3: per-string aggregation + verdict.
        agg = self._tier2_score_strings(
            all_chunks, all_scores, all_pairs, string_ranges, multihead_cfg
        )
        out.raw_score = agg.max_main
        out.max_sentence = agg.max_main_sentence
        self._tier2_finalize(tier2, out, agg, multihead_cfg)

        # Collect the leaf values that scored high (per-string scores align with
        # non-skipped strings, in string_ranges order) for the sentence-cleaner.
        high_threshold = self._config.tier2.high_risk_threshold
        scores_iter = iter(agg.per_string_scores)
        for i, (start, _end) in enumerate(string_ranges):
            if start < 0:
                continue
            if next(scores_iter, 0.0) >= high_threshold:
                out.high_risk_values.add(strings[i])

        now = time.perf_counter()
        out.phase_timings = PhaseTimings(
            prepare_ms=(t_infer_start - t_prep_start) * 1000,
            infer_ms=(t_agg_start - t_infer_start) * 1000,
            aggregate_ms=(now - t_agg_start) * 1000,
        )
        out.tier2_stats = Tier2Stats(
            string_count=len(strings),
            chunk_count=len(all_chunks),
            unique_chunk_count=unique_count,
            real_tokens=stats.real_tokens,
            padded_tokens=stats.padded_tokens,
        )
        return out

    @staticmethod
    def _tier2_build_chunks(
        tier2: Tier2Classifier,
        strings: list[str],
    ) -> tuple[list[str], list[tuple[int, int]], set[str]]:
        """Run ``prepare_chunks`` per string and flatten into one chunk list.

        ``string_ranges[i] = (start, end)`` is the half-open slice of
        ``all_chunks`` belonging to ``strings[i]``. A range of ``(-1, -1)``
        marks a string that the classifier skipped (too short, etc.).
        """
        all_chunks: list[str] = []
        string_ranges: list[tuple[int, int]] = []
        skip_reasons: set[str] = set()
        for prep in (tier2.prepare_chunks(s) for s in strings):
            if prep.get("skipped", True):
                if prep.get("skip_reason"):
                    skip_reasons.add(str(prep["skip_reason"]))
                string_ranges.append((-1, -1))
                continue
            chunks = prep.get("chunks", [])
            start_idx = len(all_chunks)
            all_chunks.extend(chunks)
            string_ranges.append((start_idx, len(all_chunks)))
        return all_chunks, string_ranges, skip_reasons

    @staticmethod
    def _tier2_run_inference(
        tier2: Tier2Classifier,
        all_chunks: list[str],
        multihead_cfg: MultiheadConfig | None,
        stats: BatchTokenStats,
    ) -> tuple[list[float] | None, list[tuple[float, float | None]] | None, str | None, int]:
        """Run classifier inference with the multi-head misconfig guard.

        Returns ``(all_scores, all_pairs, skip_reason, unique_chunk_count)``.
        ``skip_reason`` is non-``None`` either on inference failure or when
        ``multihead`` is configured but the model emits only single-head logits
        -- in that case ``all_pairs`` is forced to ``None`` so the caller falls
        back to the single-head decision path instead of silently disabling
        Tier 2 via the aux-veto branch. ``stats`` accumulates padding counts.
        """
        # Dedupe before inference: identical chunk strings score identically
        # (fixed-width buckets make scores neighbour-independent), so classify
        # each unique chunk once and fan the score back out. List responses
        # repeat enum/status/boolean values heavily — most of the saving here.
        unique_chunks: list[str] = []
        unique_index: dict[str, int] = {}
        dedupe_index: list[int] = []
        for chunk in all_chunks:
            u = unique_index.get(chunk)
            if u is None:
                u = len(unique_chunks)
                unique_index[chunk] = u
                unique_chunks.append(chunk)
            dedupe_index.append(u)
        unique_count = len(unique_chunks)

        try:
            if multihead_cfg is not None:
                unique_pairs = tier2.classify_chunks_batch_pair(unique_chunks, stats)
                if unique_pairs and all(p[1] is None for p in unique_pairs):
                    return (
                        None,
                        None,
                        "multihead configured but model emits single-head logits"
                        " -- remove `multihead` config or use a dual-head model",
                        unique_count,
                    )
                all_pairs = [unique_pairs[u] for u in dedupe_index]
                return [p[0] for p in all_pairs], all_pairs, None, unique_count
            unique_scores = tier2.classify_chunks_batch(unique_chunks, stats)
            return [unique_scores[u] for u in dedupe_index], None, None, unique_count
        except Exception as e:
            return None, None, f"Inference error: {e}", unique_count

    @staticmethod
    def _coerce_safe_score(raw: Any) -> float:
        """Coerce a classifier output to a finite float, treating NaN as 0.0."""
        if isinstance(raw, (float, int)) and raw == raw:  # NaN != NaN
            return float(raw)
        return 0.0

    @staticmethod
    def _apply_multihead_rule(
        agg: _Tier2Aggregate,
        chunk: str,
        main_score: float,
        aux_raw: float | None,
        multihead_cfg: MultiheadConfig | None,
        all_pairs_present: bool,
    ) -> None:
        """Update ``agg`` with this chunk's contribution to the multi-head rule.

        Guard-clause style early returns keep the call site flat. The rule
        fires when ``main >= main_threshold AND aux < aux_threshold``;
        when it does, we track the strongest such chunk so the rule-fire
        branch can surface it.
        """
        if multihead_cfg is None or not all_pairs_present or aux_raw is None:
            return
        if main_score < multihead_cfg.main_threshold:
            return
        if aux_raw >= multihead_cfg.aux_threshold:
            return
        agg.rule_any_fired = True
        if main_score > agg.rule_top_main:
            agg.rule_top_main = main_score
            agg.rule_top_aux = aux_raw
            agg.rule_top_chunk = chunk

    @classmethod
    def _tier2_score_strings(
        cls,
        all_chunks: list[str],
        all_scores: list[float],
        all_pairs: list[tuple[float, float | None]] | None,
        string_ranges: list[tuple[int, int]],
        multihead_cfg: MultiheadConfig | None,
    ) -> _Tier2Aggregate:
        """Aggregate chunk-level scores into per-string + multi-head state.

        Walks each non-skipped string's chunk slice once, tracking three
        independent maxima: the per-string max (for density damping), the
        global max-main pointer with its aux companion (used by the aux-
        veto branch for reporting), and the strongest chunk satisfying the
        multi-head block rule (used by the rule-fire branch for reporting).
        """
        agg = _Tier2Aggregate()
        pairs_present = all_pairs is not None
        for start_idx, end_idx in string_ranges:
            if start_idx < 0:
                continue
            s_max = 0.0
            s_max_chunk = ""
            s_max_aux: float | None = None
            for j in range(start_idx, end_idx):
                chunk = all_chunks[j]
                safe_score = cls._coerce_safe_score(all_scores[j])
                aux_raw = all_pairs[j][1] if all_pairs is not None else None
                if safe_score > s_max:
                    s_max = safe_score
                    s_max_chunk = chunk
                    s_max_aux = aux_raw if pairs_present else None
                cls._apply_multihead_rule(
                    agg, chunk, safe_score, aux_raw, multihead_cfg, pairs_present
                )
            agg.per_string_scores.append(s_max)
            if agg.max_main is None or s_max > agg.max_main:
                agg.max_main = s_max
                agg.max_main_sentence = s_max_chunk
                agg.aux_of_max_main = s_max_aux
        return agg

    @staticmethod
    def _tier2_finalize(
        tier2: Tier2Classifier,
        out: _Tier2Outcome,
        agg: _Tier2Aggregate,
        multihead_cfg: MultiheadConfig | None,
    ) -> None:
        """Resolve aggregated scores into the public outcome surface.

        Exactly one of three branches runs:

        * **Multi-head rule fire** -- the rule's chunk-level signal is
          already the decision; density damping is intentionally skipped.
        * **Multi-head aux veto** -- the rule rescued the content;
          ``effective_score`` collapses to ``0.0`` so the operator triple
          ``(tier2_score, risk_level, allowed)`` reads as zero / low /
          true. The high main score is still available via
          ``raw_score`` and the rescuing aux via ``aux_score``.
        * **Single-head** -- cross-string density damping (only when 3+
          strings); then bucket into a risk level. Density sub-threshold
          rescales with temperature (Bug 2 fix in TS 0.7.0).
        """
        if multihead_cfg is not None:
            out.multihead_blocked = agg.rule_any_fired
            if agg.rule_any_fired:
                out.risk = "high"
                out.max_sentence = agg.rule_top_chunk
                out.effective_score = agg.rule_top_main
                out.aux_score = agg.rule_top_aux
            else:
                out.risk = "low"
                out.effective_score = 0.0
                out.aux_score = agg.aux_of_max_main
            return

        if agg.max_main is None:
            return

        # Single-head density damping. Fires only on 3+ strings -- a 1-
        # or 2-string payload is mathematically indistinguishable from a
        # real short attack and damping there would create false
        # negatives. Factor ``pow(high_count/total, 0.1)`` is gentle:
        # 1/100 -> 0.63x, 1/10 -> 0.79x, 5/10 -> 0.93x.
        #
        # Bug 2 fix (TS 0.7.0 parity): rescale the "high" cutoff with
        # temperature. Raw 0.75 corresponds to logit log(3); under
        # ``T > 1`` the cutoff is ``sigmoid(log(3)/T)``. At T=1 this is
        # 0.75 (no-op); at T=2.41 it is ~0.612.
        out.effective_score = agg.max_main
        t = tier2.get_temperature() or 1.0
        density_sub_threshold = (
            0.75 if t == 1.0 else 1.0 / (1.0 + math.exp(-math.log(3.0) / t))
        )
        if len(agg.per_string_scores) > 2:
            high_count = sum(
                1 for s in agg.per_string_scores if s >= density_sub_threshold
            )
            if high_count > 0:
                factor = (high_count / len(agg.per_string_scores)) ** 0.1
                out.effective_score = agg.max_main * factor
        out.risk = tier2.get_risk_level(out.effective_score)

    def defend_tool_results(self, items: list[dict[str, Any]]) -> list[DefenseResult]:
        """Defend multiple tool results (sequential when Tier 3 is off).

        When ``enable_tier3`` is on, delegates to :meth:`defend_tool_results_async`
        via ``asyncio.run`` (parallel per item, matching npm ``defendToolResults``).
        Use the async method directly inside a running event loop.
        """
        if self._tier3_enabled:
            try:
                asyncio.get_running_loop()
            except RuntimeError:
                return asyncio.run(self.defend_tool_results_async(items))
            raise RuntimeError(
                "defend_tool_results() cannot call Tier 3 from a running event loop; "
                "use: await defense.defend_tool_results_async(items)"
            )
        return [self.defend_tool_result(item["value"], item["tool_name"]) for item in items]

    async def defend_tool_results_async(self, items: list[dict[str, Any]]) -> list[DefenseResult]:
        """Defend multiple tool results concurrently (npm ``defendToolResults`` parity).

        Runs :meth:`defend_tool_result_async` per item in parallel via ``asyncio.gather``.
        Result order matches ``items``.
        """
        if not items:
            return []
        return list(
            await asyncio.gather(
                *(self.defend_tool_result_async(item["value"], item["tool_name"]) for item in items)
            )
        )

    def analyze(self, text: str) -> Tier1Result:
        """Analyze text for injection patterns (Tier 1 only)."""
        return self._pattern_detector.analyze(text)

    def get_config(self) -> PromptDefenseConfig:
        return self._config


def create_prompt_defense(**kwargs) -> PromptDefense:
    return PromptDefense(**kwargs)
