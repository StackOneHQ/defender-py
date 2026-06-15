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
from dataclasses import dataclass, field
from typing import Any

from ..classifiers.pattern_detector import PatternDetector, create_pattern_detector
from ..classifiers.tier2_classifier import Tier2Classifier, create_tier2_classifier
from ..classifiers.tier3_orchestrator import get_default_tier3_provider
from ..config import MAX_TRAVERSAL_DEPTH, create_config
from ..sfe.preprocess import SfePredictor, get_default_predictor, sfe_preprocess
from ..types import (
    DefenderMode,
    DefenseResult,
    MultiheadConfig,
    PromptDefenseConfig,
    RiskLevel,
    Tier1Result,
    Tier3EscalationBand,
    Tier3Provider,
    Tier3Result,
    Tier3Skip,
    Tier3Verdict,
)
from .tool_result_sanitizer import ToolResultSanitizer, create_tool_result_sanitizer

_logger = logging.getLogger(__name__)

_DEFAULT_TIER3_BAND = Tier3EscalationBand(lower=0.3, upper=0.85)
_DEFAULT_TIER3_MAX_TEXT_LENGTH = 10000


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


def _bounded_join_strings(strings: list[str], max_len: int, sep: str = "\n") -> str:
    """Join strings with ``sep``, capping total length at ``max_len`` without building the full join first."""
    if max_len <= 0:
        return ""
    parts: list[str] = []
    used = 0
    sep_len = len(sep)
    for s in strings:
        if not s:
            continue
        prefix = sep_len if parts else 0
        if used + prefix >= max_len:
            break
        remaining = max_len - used - prefix
        if len(s) <= remaining:
            parts.append(s)
            used += prefix + len(s)
        else:
            parts.append(s[:remaining])
            break
    return sep.join(parts)


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
        default_risk_level: RiskLevel = "medium",
        annotate_boundary: bool = False,
        enable_tier3: bool = False,
        defender_mode: DefenderMode = "cascade",
        tier3: dict[str, Any] | None = None,
    ):
        self._config: PromptDefenseConfig = create_config(config)
        if block_high_risk:
            self._config.block_high_risk = True

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
            block_high_risk=block_high_risk,
            cumulative_risk_thresholds=self._config.cumulative_risk_thresholds,
            annotate_boundary=annotate_boundary,
        )

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
        tier3_opts = tier3 or {}
        if tier3_opts.get("provider") is not None:
            self._tier3_custom_provider = tier3_opts["provider"]
        max_text_length = tier3_opts.get("max_text_length", tier3_opts.get("maxTextLength"))
        if max_text_length is not None:
            if isinstance(max_text_length, (int, float)) and math.isfinite(max_text_length) and max_text_length > 0:
                self._tier3_max_text_length = int(max_text_length)
            else:
                _logger.warning(
                    "[defender] invalid tier3.max_text_length %s — must be a positive finite number. "
                    "Falling back to default %s.",
                    max_text_length,
                    _DEFAULT_TIER3_MAX_TEXT_LENGTH,
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

    def warmup_tier2(self) -> None:
        if self._tier2:
            self._tier2.warmup()
        if self._sfe_enabled and self._sfe_custom_predictor is None:
            predictor = get_default_predictor()
            if predictor is None:
                _logger.warning(
                    "[defender] SFE predictor unavailable at warmup; "
                    "calls with use_sfe enabled will pass payloads through unfiltered."
                )

    def is_tier2_ready(self) -> bool:
        return self._tier2.is_ready() if self._tier2 else False

    def _resolve_tier3_provider(self) -> Tier3Provider | None:
        return self._tier3_custom_provider or get_default_tier3_provider()

    @staticmethod
    def _validate_tier3_verdict(verdict: Any) -> Tier3Verdict | Tier3Skip:
        if isinstance(verdict, Tier3Verdict):
            if verdict.decision in ("block", "allow"):
                return verdict
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
            score=verdict.get("score"),
            raw=verdict.get("raw"),
            latency_ms=verdict.get("latency_ms", verdict.get("latencyMs")),
        )

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
            return validated, validated.decision == "block"
        except Exception as e:
            return Tier3Skip(skip_reason=f"Tier 3 provider error: {e}"), None

    @staticmethod
    def _finalize_allowed_and_risk(
        *,
        detections: list[str],
        fields_sanitized: list[str],
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
            or bool(fields_sanitized)
            or (tier2_has_threat and not tier3_overrode_to_allow)
            or tier3_overrode_to_block
        )
        allowed = (
            not block_high_risk
            or not has_threats
            or risk_level not in ("high", "critical")
        )
        return risk_level, allowed

    async def _run_tier3_only(
        self,
        value: Any,
        provider: Tier3Provider,
        tool_name: str,
        depth_flag: dict[str, bool],
        start_time: float,
    ) -> DefenseResult:
        strings = [s for s in _extract_strings(value, None, depth_flag) if len(s) > 0]
        bounded = _bounded_join_strings(strings, self._tier3_max_text_length)

        verdict: Tier3Verdict | None = None
        skip_reason: str | None = None
        if len(bounded) == 0:
            skip_reason = "No strings extracted from tool result"
        else:
            try:
                raw = await self._invoke_tier3_classify(provider, bounded, tool_name)
                validated = self._validate_tier3_verdict(raw)
                if isinstance(validated, Tier3Skip):
                    skip_reason = validated.skip_reason
                else:
                    verdict = validated
            except Exception as e:
                skip_reason = f"Tier 3 provider error: {e}"

        sanitized = self._tool_sanitizer.sanitize(value, tool_name=tool_name)
        detections, fields_sanitized, prm = self._tier1_metadata(sanitized)

        blocked = verdict is not None and verdict.decision == "block"
        risk_level: RiskLevel = "high" if blocked else "low"
        allowed = not self._config.block_high_risk or not blocked
        tier3_result: Tier3Result = (
            verdict if verdict is not None else Tier3Skip(skip_reason=skip_reason or "Tier 3 skipped")
        )

        return DefenseResult(
            allowed=allowed,
            risk_level=risk_level,
            sanitized=sanitized.sanitized,
            detections=detections,
            fields_sanitized=fields_sanitized,
            patterns_by_field=prm,
            tier3=tier3_result,
            fields_dropped=[],
            truncated_at_depth=depth_flag["hit"] or None,
            latency_ms=(time.perf_counter() - start_time) * 1000,
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

        sanitized = self._tool_sanitizer.sanitize(value, tool_name=tool_name)
        detections, fields_sanitized, prm = self._tier1_metadata(sanitized)

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
            fields_sanitized=fields_sanitized,
            tier2_has_threat=tier2_has_threat,
            tier2_idx=tier2_idx,
            tier1_idx=tier1_idx,
            risk_level=risk_level,
            block_high_risk=self._config.block_high_risk,
            tier3_override_block=tier3_override_block,
        )

        return DefenseResult(
            allowed=allowed,
            risk_level=risk_level,
            sanitized=sanitized.sanitized,
            detections=detections,
            fields_sanitized=fields_sanitized,
            patterns_by_field=prm,
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

        # Tier 1: pattern-based sanitization on the original payload (matches TS 0.6.3).
        sanitized = self._tool_sanitizer.sanitize(value, tool_name=tool_name)

        # Collect Tier 1 metadata
        prm = sanitized.metadata.patterns_removed_by_field
        mbf = sanitized.metadata.methods_by_field
        detections = list(dict.fromkeys(p for patterns in prm.values() for p in patterns))

        active_methods = {"role_stripping", "pattern_removal", "encoding_detection"}
        fields_sanitized = [
            field for field, methods in mbf.items()
            if any(m in active_methods for m in methods)
        ]

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
            fields_sanitized=fields_sanitized,
            tier2_has_threat=tier2_has_threat,
            tier2_idx=tier2_idx,
            tier1_idx=tier1_idx,
            risk_level=risk_level,
            block_high_risk=self._config.block_high_risk,
            tier3_override_block=None,
        )

        return DefenseResult(
            allowed=allowed,
            risk_level=risk_level,
            sanitized=sanitized.sanitized,
            detections=detections,
            fields_sanitized=fields_sanitized,
            patterns_by_field=prm,
            tier2_score=tier2.effective_score,
            tier2_raw_score=tier2.raw_score,
            tier2_aux_score=tier2.aux_score,
            tier2_multihead_blocked=tier2.multihead_blocked,
            tier2_skip_reason=tier2.skip_reason,
            max_sentence=tier2.max_sentence,
            fields_dropped=fields_dropped,
            truncated_at_depth=depth_flag["hit"] or None,
            latency_ms=(time.perf_counter() - start_time) * 1000,
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

        all_chunks, string_ranges, skip_reasons = self._tier2_build_chunks(tier2, strings)
        if not all_chunks:
            out.skip_reason = (
                "All strings skipped by classifier"
                if not skip_reasons
                else f"All strings skipped by classifier: {'; '.join(sorted(skip_reasons))}"
            )
            return out

        multihead_cfg = tier2.get_multihead_config()
        all_scores, all_pairs, infer_skip = self._tier2_run_inference(
            tier2, all_chunks, multihead_cfg
        )
        if infer_skip is not None:
            out.skip_reason = infer_skip
        if all_scores is None:
            return out

        agg = self._tier2_score_strings(
            all_chunks, all_scores, all_pairs, string_ranges, multihead_cfg
        )
        out.raw_score = agg.max_main
        out.max_sentence = agg.max_main_sentence
        self._tier2_finalize(tier2, out, agg, multihead_cfg)
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
    ) -> tuple[list[float] | None, list[tuple[float, float | None]] | None, str | None]:
        """Run classifier inference with the multi-head misconfig guard.

        Returns ``(all_scores, all_pairs, skip_reason)``. ``skip_reason``
        is non-``None`` either on inference failure or when ``multihead``
        is configured but the model emits only single-head logits -- in
        that case ``all_pairs`` is forced to ``None`` so the caller falls
        back to the single-head decision path instead of silently
        disabling Tier 2 via the aux-veto branch.
        """
        try:
            if multihead_cfg is not None:
                pairs = tier2.classify_chunks_batch_pair(all_chunks)
                if pairs and all(p[1] is None for p in pairs):
                    return (
                        None,
                        None,
                        "multihead configured but model emits single-head logits"
                        " -- remove `multihead` config or use a dual-head model",
                    )
                return [p[0] for p in pairs], pairs, None
            scores = tier2.classify_chunks_batch(all_chunks)
            return scores, None, None
        except Exception as e:
            return None, None, f"Inference error: {e}"

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
