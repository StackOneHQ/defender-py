"""PromptDefense - Main Entry Point.

The primary class for using the prompt defense framework.
Provides a simple API for defending tool results against prompt injection.
"""

from __future__ import annotations

import logging
import math
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

        # Tier 2: ML classification on strings from the SFE-filtered view
        # (or full value if SFE off).
        #
        # Three score variables track different stages of the same signal:
        #   - tier2_score: local intermediate. Starts as max-chunk main, gets
        #     reassigned to the rule-trigger chunk's main under multi-head
        #     rule fire. NOT surfaced directly on the result.
        #   - tier2_raw_score: max-chunk main pre-density, pre-rule-reassignment.
        #     Surfaced as ``result.tier2_raw_score`` for forensics.
        #   - tier2_effective_score: the score that drives the block decision.
        #     Under single-head this is post-density ``tier2_score``. Under
        #     multi-head rule fire this is the rule-trigger chunk's main.
        #     Under aux veto this is explicitly ``0.0``. Surfaced as
        #     ``result.tier2_score``.
        tier2_score: float | None = None
        tier2_raw_score: float | None = None
        tier2_aux_score: float | None = None
        tier2_multihead_blocked: bool | None = None
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
                    multihead_cfg = self._tier2.get_multihead_config()
                    all_scores: list[float] | None = None
                    all_pairs: list[tuple[float, float | None]] | None = None
                    try:
                        if multihead_cfg is not None:
                            all_pairs = self._tier2.classify_chunks_batch_pair(all_chunks)
                            # Multi-head misconfig guard: single-head model
                            # under a multi-head config. Every aux is None.
                            # Without this guard the rule path sees no aux
                            # signal, treats no chunk as a multihead block,
                            # fires the aux-veto branch, and collapses
                            # ``tier2_effective_score`` to 0 -- Tier 2 is
                            # silently disabled. Surface the misconfig instead.
                            if all_pairs and all(p[1] is None for p in all_pairs):
                                tier2_skip_reason = (
                                    "multihead configured but model emits single-head logits"
                                    " -- remove `multihead` config or use a dual-head model"
                                )
                                all_pairs = None
                            else:
                                all_scores = [p[0] for p in all_pairs]
                        else:
                            all_scores = self._tier2.classify_chunks_batch(all_chunks)
                    except Exception as e:
                        tier2_skip_reason = f"Inference error: {e}"

                    if all_scores is not None:
                        per_string_scores: list[float] = []
                        # Multi-head: track whether any chunk independently
                        # triggers the (main >= main_thr AND aux < aux_thr)
                        # rule, and remember the strongest such chunk so the
                        # result surfaces it.
                        mh_any_block = False
                        mh_top_block_chunk = ""
                        mh_top_block_main = -1.0
                        mh_top_block_aux: float | None = None
                        # Aux score of the chunk with the global-max main
                        # score. Only populated under multi-head config; used
                        # by the aux-veto branch so the reported
                        # ``tier2_aux_score`` points at the chunk that came
                        # closest to blocking.
                        aux_of_max_main: float | None = None
                        for i, (start_idx, end_idx) in enumerate(string_ranges):
                            if start_idx < 0:
                                continue
                            s_max = 0.0
                            s_max_chunk = ""
                            s_max_aux: float | None = None
                            for j in range(start_idx, end_idx):
                                raw = all_scores[j]
                                safe_score = (
                                    float(raw)
                                    if isinstance(raw, (float, int)) and raw == raw
                                    else 0.0
                                )
                                if safe_score > s_max:
                                    s_max = safe_score
                                    s_max_chunk = all_chunks[j]
                                    if all_pairs is not None:
                                        aux_raw = all_pairs[j][1]
                                        s_max_aux = aux_raw
                                if multihead_cfg is not None and all_pairs is not None:
                                    aux_raw = all_pairs[j][1]
                                    if aux_raw is not None:
                                        chunk_blocks = (
                                            safe_score >= multihead_cfg.main_threshold
                                            and aux_raw < multihead_cfg.aux_threshold
                                        )
                                        if chunk_blocks:
                                            mh_any_block = True
                                            if safe_score > mh_top_block_main:
                                                mh_top_block_main = safe_score
                                                mh_top_block_aux = aux_raw
                                                mh_top_block_chunk = all_chunks[j]
                            per_string_scores.append(s_max)
                            if tier2_score is None or s_max > tier2_score:
                                tier2_score = s_max
                                max_sentence = s_max_chunk
                                aux_of_max_main = s_max_aux

                        # Bug 3 fix: capture the raw max-chunk main score
                        # before any density adjustment or multi-head rule
                        # reassignment. Surfaced as ``tier2_raw_score`` on
                        # the result for forensics / threshold tuning.
                        tier2_raw_score = tier2_score

                        if multihead_cfg is not None and all_pairs is not None:
                            # Multi-head decision rule: report the rule-
                            # triggering chunk when the rule fires, otherwise
                            # report the (rescued) global max-main chunk for
                            # debugging. Density damping is intentionally not
                            # applied here -- the rule's chunk-level main
                            # scores are already the decision signal.
                            tier2_multihead_blocked = mh_any_block
                            if mh_any_block:
                                tier2_risk = "high"
                                max_sentence = mh_top_block_chunk
                                tier2_score = mh_top_block_main
                                tier2_effective_score = mh_top_block_main
                                tier2_aux_score = mh_top_block_aux
                            else:
                                # Aux veto fired -- the rule rescued this
                                # content, so Tier 2 contributed nothing to a
                                # block. Set ``tier2_effective_score = 0`` so
                                # the operator triple (``tier2_score``,
                                # ``risk_level``, ``allowed``) reads coherently
                                # as zero / low / true. The model's actual
                                # main signal is on ``tier2_raw_score``; the
                                # aux that did the rescuing is reported via
                                # ``tier2_aux_score``.
                                tier2_risk = "low"
                                tier2_effective_score = 0.0
                                tier2_aux_score = aux_of_max_main
                        elif tier2_score is not None:
                            # Single-head path: cross-string density
                            # adjustment (mild), then bucket into risk level.
                            #
                            # Density damping fires only on 3+ strings -- a
                            # 1- or 2-string payload is mathematically
                            # indistinguishable from a real short attack, and
                            # damping would create false negatives. For
                            # larger payloads, a lone high-scoring string
                            # surrounded by many benign strings is typical
                            # of benign connector responses. Factor
                            # ``pow(high_count/total, 0.1)`` is gentle:
                            # 1/100 -> 0.63x, 1/10 -> 0.79x, 5/10 -> 0.93x.
                            #
                            # Bug 2 fix: the "high" cutoff was originally
                            # hardcoded at 0.75 (raw sigmoid space). Under
                            # ``temperature_t > 1`` every score is
                            # ``sigmoid(logit / T)`` -- compressed toward
                            # 0.5 -- so a literal 0.75 cutoff stops counting
                            # events that were "high" under raw scoring.
                            # Rescale in logit-space: raw 0.75 corresponds
                            # to logit ``log(3) ~ 1.0986``; calibrated
                            # cutoff is ``sigmoid(log(3)/T)``. At T=1 this is
                            # 0.75 (no-op); at T=2.41 it's ~0.612.
                            tier2_effective_score = tier2_score
                            t = self._tier2.get_temperature() or 1.0
                            density_sub_threshold = (
                                0.75 if t == 1.0 else 1.0 / (1.0 + math.exp(-math.log(3.0) / t))
                            )
                            if len(per_string_scores) > 2:
                                high_count = sum(
                                    1 for s in per_string_scores if s >= density_sub_threshold
                                )
                                if high_count > 0:
                                    factor = (high_count / len(per_string_scores)) ** 0.1
                                    tier2_effective_score = tier2_score * factor
                            tier2_risk = self._tier2.get_risk_level(tier2_effective_score)

        # Combine risk levels (take the higher of Tier 1 and Tier 2)
        tier1_idx = _RISK_LEVELS.index(sanitized.metadata.overall_risk_level)
        tier2_idx = _RISK_LEVELS.index(tier2_risk)
        risk_level = _RISK_LEVELS[max(tier1_idx, tier2_idx)]

        # Three-way Tier 2 threat derivation. In multi-head mode the rule
        # replaces the threshold check: a flagged chunk under
        # ``main >= main_thr AND aux < aux_thr`` is a Tier 2 threat; aux veto
        # suppresses the threshold-based Tier 2 signal entirely.
        if tier2_multihead_blocked is True:
            tier2_has_threat = True
        elif tier2_multihead_blocked is False:
            tier2_has_threat = False
        else:
            tier2_has_threat = (
                tier2_effective_score is not None
                and tier2_effective_score >= self._config.tier2.high_risk_threshold
            )

        # Threat signals: Tier 1 detections, Tier 1 sanitization methods, or
        # Tier 2 above-threshold (subject to multi-head veto).
        has_threats = bool(detections) or bool(fields_sanitized) or tier2_has_threat

        # Three cases for ``allowed``:
        # 1. ``block_high_risk`` is off -> always allow.
        # 2. No threat signals found -> allow (base risk from tool rules
        #    alone does not block).
        # 3. Risk did not reach high/critical -> allow.
        allowed = (
            not self._config.block_high_risk
            or not has_threats
            or risk_level not in ("high", "critical")
        )

        # ``tier2_score`` reports ``tier2_effective_score`` -- the value that
        # drove the block decision. The multi-head aux veto path sets
        # ``tier2_effective_score = 0.0`` (not ``None``), keeping the triple
        # coherent: tier2_score=0 / risk_level low / allowed=true.
        # ``tier2_raw_score`` is the pre-density / pre-rule max-chunk main
        # score for forensics -- never use it to make decisions.
        return DefenseResult(
            allowed=allowed,
            risk_level=risk_level,
            sanitized=sanitized.sanitized,
            detections=detections,
            fields_sanitized=fields_sanitized,
            patterns_by_field=prm,
            tier2_score=tier2_effective_score,
            tier2_raw_score=tier2_raw_score,
            tier2_aux_score=tier2_aux_score,
            tier2_multihead_blocked=tier2_multihead_blocked,
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
