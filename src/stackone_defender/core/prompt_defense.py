"""PromptDefense - Main Entry Point.

The primary class for using the prompt defense framework.
Provides a simple API for defending tool results against prompt injection.
"""

from __future__ import annotations

import logging
import time
from typing import Any

from ..classifiers.pattern_detector import PatternDetector, create_pattern_detector
from ..classifiers.tier2_classifier import Tier2Classifier, create_tier2_classifier
from ..config import MAX_TRAVERSAL_DEPTH, create_config
from ..sfe.preprocess import SfePredictor, get_default_predictor, sfe_preprocess
from ..types import DefenseResult, PromptDefenseConfig, RiskLevel, Tier1Result
from .tool_result_sanitizer import ToolResultSanitizer, create_tool_result_sanitizer

_logger = logging.getLogger(__name__)


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

    def defend_tool_result(self, value: Any, tool_name: str) -> DefenseResult:
        """Defend a tool result using Tier 1 and optionally Tier 2 classification.

        When SFE is enabled, ``fields_dropped`` lists paths excluded from **Tier 2**
        string extraction only; the returned ``sanitized`` payload is still Tier 1 output
        from the **original** tool value (SFE does not remove fields from the returned object).
        """
        start_time = time.perf_counter()
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

        # Tier 2: ML classification on strings from the SFE-filtered view (or full value if SFE off).
        tier2_score: float | None = None
        tier2_effective_score: float | None = None
        max_sentence: str | None = None
        tier2_risk: RiskLevel = "low"
        tier2_skip_reason: str | None = None

        if self._tier2:
            fields_for_tier2 = (
                self._tier2_fields if self._tier2_fields is not None else self._config.tier2.tier2_fields
            )
            strings = [
                s
                for s in _extract_strings(sfe_filtered_value, fields_for_tier2, depth_flag)
                if len(s) > 0
            ]
            if not strings:
                scoped = fields_for_tier2 is not None and len(fields_for_tier2) > 0
                if scoped:
                    tier2_skip_reason = "No strings found in tier2_fields"
                else:
                    tier2_skip_reason = "No strings extracted from tool result"
            else:
                preps = [self._tier2.prepare_chunks(s) for s in strings]
                all_chunks: list[str] = []
                string_ranges: list[tuple[int, int]] = []
                skip_reasons: set[str] = set()
                for prep in preps:
                    if prep.get("skipped", True):
                        if prep.get("skip_reason"):
                            skip_reasons.add(str(prep["skip_reason"]))
                        string_ranges.append((-1, -1))
                        continue
                    chunks = prep.get("chunks", [])
                    start_idx = len(all_chunks)
                    all_chunks.extend(chunks)
                    string_ranges.append((start_idx, len(all_chunks)))

                if not all_chunks:
                    tier2_skip_reason = (
                        "All strings skipped by classifier"
                        if not skip_reasons
                        else f"All strings skipped by classifier: {'; '.join(sorted(skip_reasons))}"
                    )
                else:
                    all_scores: list[float] | None = None
                    try:
                        all_scores = self._tier2.classify_chunks_batch(all_chunks)
                    except Exception as e:
                        tier2_skip_reason = f"Inference error: {e}"

                    if all_scores is not None:
                        per_string_scores: list[float] = []
                        for start_idx, end_idx in string_ranges:
                            if start_idx < 0:
                                continue
                            string_max = 0.0
                            string_max_chunk = ""
                            for chunk_idx in range(start_idx, end_idx):
                                raw = all_scores[chunk_idx]
                                safe_score = raw if isinstance(raw, (float, int)) and raw == raw else 0.0
                                if safe_score > string_max:
                                    string_max = safe_score
                                    string_max_chunk = all_chunks[chunk_idx]
                            per_string_scores.append(string_max)
                            if tier2_score is None or string_max > tier2_score:
                                tier2_score = string_max
                                max_sentence = string_max_chunk

                        tier2_effective_score = tier2_score
                        density_sub_threshold = 0.75
                        if tier2_score is not None and len(per_string_scores) > 2:
                            high_count = len([s for s in per_string_scores if s >= density_sub_threshold])
                            if high_count > 0:
                                factor = (high_count / len(per_string_scores)) ** 0.1
                                tier2_effective_score = tier2_score * factor

                        if tier2_effective_score is not None:
                            tier2_risk = self._tier2.get_risk_level(tier2_effective_score)

        # Combine risk levels
        tier1_idx = _RISK_LEVELS.index(sanitized.metadata.overall_risk_level)
        tier2_idx = _RISK_LEVELS.index(tier2_risk)
        risk_level = _RISK_LEVELS[max(tier1_idx, tier2_idx)]

        # Determine whether any threat signals were found (Tier 1 or Tier 2).
        # fields_sanitized captures sanitization methods (role stripping, encoding detection, etc.)
        # that may fire without adding named pattern detections, so we include it here.
        has_threats = (
            len(detections) > 0
            or len(fields_sanitized) > 0
            or (tier2_effective_score is not None and tier2_effective_score >= self._config.tier2.high_risk_threshold)
        )

        # Three cases for allowed:
        # 1. block_high_risk is off -> always allow
        # 2. No threat signals found -> allow (base risk from tool rules alone does not block)
        # 3. Risk did not reach high/critical -> allow
        allowed = not self._config.block_high_risk or not has_threats or risk_level not in ("high", "critical")

        return DefenseResult(
            allowed=allowed,
            risk_level=risk_level,
            sanitized=sanitized.sanitized,
            detections=detections,
            fields_sanitized=fields_sanitized,
            patterns_by_field=prm,
            tier2_score=tier2_score,
            tier2_skip_reason=tier2_skip_reason,
            max_sentence=max_sentence,
            fields_dropped=fields_dropped,
            truncated_at_depth=depth_flag["hit"] or None,
            latency_ms=(time.perf_counter() - start_time) * 1000,
        )

    def defend_tool_results(self, items: list[dict[str, Any]]) -> list[DefenseResult]:
        """Defend multiple tool results."""
        return [self.defend_tool_result(item["value"], item["tool_name"]) for item in items]

    def analyze(self, text: str) -> Tier1Result:
        """Analyze text for injection patterns (Tier 1 only)."""
        return self._pattern_detector.analyze(text)

    def get_config(self) -> PromptDefenseConfig:
        return self._config


def create_prompt_defense(**kwargs) -> PromptDefense:
    return PromptDefense(**kwargs)
