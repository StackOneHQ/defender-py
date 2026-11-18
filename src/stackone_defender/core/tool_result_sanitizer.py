"""Tool Result Sanitizer (detect-and-gate).

Traverses a tool result, runs Tier 1 pattern detection, and records DETECTION
evidence — it does NOT mutate or rewrite content. The returned ``sanitized``
value is the original payload (optionally boundary-wrapped when
``annotate_boundary`` is enabled). Blocking is expressed upstream via the
``allowed`` / ``risk_level`` decision, never by redacting content.
"""

from __future__ import annotations

import time
from collections.abc import Mapping
from typing import Any

from ..classifiers.pattern_detector import PatternDetector, create_pattern_detector, tier1_analysis_copy
from ..config import (
    DANGEROUS_KEYS,
    DEFAULT_CUMULATIVE_RISK_THRESHOLDS,
    DEFAULT_MAX_FIELD_ANALYSIS_LENGTH,
    DEFAULT_RISKY_FIELDS,
    DEFAULT_TRAVERSAL_CONFIG,
)
from ..sanitizers.encoding_detector import decode_all_levels
from ..types import (
    CumulativeRiskTracker,
    DataBoundary,
    PatternMatch,
    RiskLevel,
    RiskyFieldConfig,
    SanitizationContext,
    SanitizationMetadata,
    SanitizationResult,
    TraversalConfig,
)
from ..utils.boundary import generate_data_boundary, wrap_with_boundary
from ..utils.field_detection import is_risky_field
from ..utils.structure import (
    create_size_metrics,
    detect_structure_type,
    get_wrapped_data,
    is_paginated_response,
    should_continue_traversal,
    update_size_metrics,
)
from ..utils.text import phrase_key, to_utf16, truncate

# Risk levels in ascending order, for monotonic max comparison.
RISK_ORDER: list[RiskLevel] = ["low", "medium", "high", "critical"]


def _risk_ladder(current: RiskLevel, suggested: RiskLevel) -> RiskLevel:
    """Escalate ``current`` toward ``suggested`` without downgrading."""
    if suggested == "critical":
        return "critical"
    if suggested == "high" and current != "critical":
        return "high"
    if suggested == "medium" and current == "low":
        return "medium"
    return current


# Characters of context on each side that make an occurrence in a copy the same as one in the raw text.
_OCCURRENCE_CONTEXT = 16


def _has_occurrence_not_in(copy: str, raw: str, k: str) -> bool:
    """Whether some occurrence of ``k`` in ``copy``, with its context, isn't in ``raw``: the copy holds
    that occurrence-with-context more often than the raw text, so a repeated context can't vouch for
    it. Context is counted in UTF-16 units, as in TS."""
    copy, raw, k = to_utf16(copy), to_utf16(raw), to_utf16(k)
    i = copy.find(k)
    while i >= 0:
        around = copy[max(0, i - _OCCURRENCE_CONTEXT) : i + len(k) + _OCCURRENCE_CONTEXT]
        if copy.count(around) > raw.count(around):
            return True
        i = copy.find(k, i + len(k))
    return False

class ToolResultSanitizer:
    """Detect threats in a tool result and return the original content unchanged."""

    def __init__(
        self,
        *,
        risky_fields: RiskyFieldConfig | None = None,
        traversal: TraversalConfig | None = None,
        default_risk_level: RiskLevel = "low",
        use_tier1_classification: bool = True,
        cumulative_risk_thresholds: dict[str, int | float] | None = None,
        annotate_boundary: bool = False,
        max_field_analysis_length: int = DEFAULT_MAX_FIELD_ANALYSIS_LENGTH,
    ):
        self._risky_fields = risky_fields or DEFAULT_RISKY_FIELDS
        self._traversal = traversal or DEFAULT_TRAVERSAL_CONFIG
        self._default_risk_level = default_risk_level
        self._use_tier1 = use_tier1_classification
        self._annotate_boundary = annotate_boundary
        self._max_field_analysis_length = max_field_analysis_length
        merged = dict(DEFAULT_CUMULATIVE_RISK_THRESHOLDS)
        if cumulative_risk_thresholds:
            merged.update(cumulative_risk_thresholds)
        self._cumulative_thresholds = merged

        self._pattern_detector: PatternDetector = create_pattern_detector()

    def sanitize(
        self,
        value: Any,
        *,
        tool_name: str,
        vertical: str | None = None,
        resource: str | None = None,
        risk_level: RiskLevel | None = None,
        boundary: DataBoundary | None = None,
    ) -> SanitizationResult:
        start_time = time.perf_counter()
        if self._annotate_boundary:
            boundary = boundary or generate_data_boundary()
        else:
            boundary = None
        cumulative_risk = self._create_cumulative_tracker()
        size_metrics = create_size_metrics()

        context = SanitizationContext(
            path="",
            field_name="",
            tool_name=tool_name,
            vertical=vertical or self._extract_vertical(tool_name),
            resource=resource or self._extract_resource(tool_name),
            risk_level=risk_level or self._default_risk_level,
            boundary=boundary,
            cumulative_risk=cumulative_risk,
        )

        metadata = SanitizationMetadata(
            overall_risk_level=context.risk_level,
            size_metrics=size_metrics,
            risky_field_names=[],
        )

        # A top-level string IS the entire tool result — run Tier 1 on it directly.
        # The recursion only scans strings under risky object fields, so a bare
        # string would otherwise skip Tier 1 (a gap when Tier 2 is off/unavailable).
        if isinstance(value, str):
            sanitized: Any = self._sanitize_string_field(value, context, metadata, True)
        else:
            sanitized = self._sanitize_value(value, context, metadata, 0)

        # Cumulative fragmented-attack escalation — raise (max), never downgrade.
        if self._should_escalate(cumulative_risk):
            metadata.cumulative_risk_escalated = True
            self._raise_overall_risk(metadata, "high")

        metadata.total_latency_ms = (time.perf_counter() - start_time) * 1000
        metadata.size_metrics = size_metrics
        metadata.risky_field_names = list(dict.fromkeys(metadata.risky_field_names))
        return SanitizationResult(sanitized=sanitized, metadata=metadata)

    # ------------------------------------------------------------------
    # Recursive traversal
    # ------------------------------------------------------------------

    def _detection_allowed(self, index: int, size: int, metadata: SanitizationMetadata) -> bool:
        """Whether Tier 1 detection may run for entry ``index`` of a container of
        ``size``. Primary bound is the call-scoped byte budget (``max_size``). The
        deprecated ``skip_large_arrays`` per-container cap is honored only when
        explicitly enabled. Flags ``analysis_truncated`` when detection is skipped;
        no data is ever dropped."""
        if metadata.size_metrics.estimated_bytes >= self._traversal.max_size:
            metadata.analysis_truncated = True
            return False
        if self._traversal.skip_large_arrays and size > self._traversal.large_array_threshold and index >= 100:
            metadata.analysis_truncated = True
            return False
        return True

    def _sanitize_value(
        self,
        value: Any,
        context: SanitizationContext,
        metadata: SanitizationMetadata,
        depth: int,
        detect: bool = True,
    ) -> Any:
        # Strings inside arrays/nesting reach here (object fields go via _sanitize_object).
        # Scan risky ones so {"name": [INJ]} is covered like {"name": INJ}.
        if isinstance(value, str):
            if self._is_field_risky(context.field_name, context.tool_name):
                return self._sanitize_string_field(value, context, metadata, detect)
            update_size_metrics(metadata.size_metrics, value)
            self._scan_encoded_field(value, context, metadata, detect)
            return value
        update_size_metrics(metadata.size_metrics, value)
        if not should_continue_traversal(metadata.size_metrics, depth, self._traversal.max_size, self._traversal.max_depth):
            return value
        if value is None:
            return value
        if isinstance(value, list):
            return self._sanitize_array(value, context, metadata, depth, detect)
        # Any dict (dict, OrderedDict, bson.SON, ...) is traversed so key stripping + detection apply.
        if isinstance(value, dict):
            return self._sanitize_object(value, context, metadata, depth, detect)
        # Tuples, sets and other Mappings (UserDict, MappingProxyType) are scanned for detection but
        # returned as given. Other objects (datetime, class instances) pass through.
        if isinstance(value, (tuple, set, frozenset)):
            self._sanitize_array(list(value), context, metadata, depth, detect)
        elif isinstance(value, Mapping):
            self._sanitize_object(dict(value), context, metadata, depth, detect)
        return value

    def _child_context(self, context: SanitizationContext, path: str, field_name: str) -> SanitizationContext:
        return SanitizationContext(
            path=path,
            field_name=field_name,
            tool_name=context.tool_name,
            vertical=context.vertical,
            resource=context.resource,
            risk_level=context.risk_level,
            boundary=context.boundary,
            cumulative_risk=context.cumulative_risk,
        )

    def _sanitize_array(
        self,
        arr: list,
        context: SanitizationContext,
        metadata: SanitizationMetadata,
        depth: int,
        detect: bool = True,
    ) -> list:
        # array_count is incremented in update_size_metrics (via _sanitize_value,
        # and at the direct call sites below that bypass it).
        result = []
        for i, item in enumerate(arr):
            ctx = self._child_context(context, f"{context.path}[{i}]", context.field_name)
            result.append(self._sanitize_value(item, ctx, metadata, depth + 1, detect and self._detection_allowed(i, len(arr), metadata)))
        return result

    def _sanitize_object(
        self,
        obj: dict,
        context: SanitizationContext,
        metadata: SanitizationMetadata,
        depth: int,
        detect: bool = True,
    ) -> dict:
        # object_count is incremented once in update_size_metrics (via _sanitize_value).

        if is_paginated_response(obj):
            return self._sanitize_paginated(obj, context, metadata, depth, detect)
        if detect_structure_type(obj) == "wrapped":
            return self._sanitize_wrapped(obj, context, metadata, depth, detect)

        result: dict = {}
        for i, (key, val) in enumerate(obj.items()):
            entry_detect = detect and self._detection_allowed(i, len(obj), metadata)
            if key in DANGEROUS_KEYS:
                self._record_dangerous_key(metadata, context.path, key)
                continue
            field_path = f"{context.path}.{key}" if context.path else key
            if entry_detect:
                self._detect_in_key(key, field_path, context, metadata)
            field_ctx = self._child_context(context, field_path, key)
            if self._is_field_risky(key, context.tool_name) and isinstance(val, str):
                if entry_detect:
                    metadata.risky_field_names.append(key)
                result[key] = self._sanitize_string_field(val, field_ctx, metadata, entry_detect)
            else:
                result[key] = self._sanitize_value(val, field_ctx, metadata, depth + 1, entry_detect)
        return result

    def _sanitize_paginated(
        self,
        obj: dict,
        context: SanitizationContext,
        metadata: SanitizationMetadata,
        depth: int,
        detect: bool = True,
    ) -> dict:
        result: dict = {}
        data_keys = {"data", "results", "items", "records"}
        for i, (key, val) in enumerate(obj.items()):
            entry_detect = detect and self._detection_allowed(i, len(obj), metadata)
            if key in DANGEROUS_KEYS:
                self._record_dangerous_key(metadata, context.path, key)
                continue
            field_path = f"{context.path}.{key}" if context.path else key
            if entry_detect:
                self._detect_in_key(key, field_path, context, metadata)
            field_ctx = self._child_context(context, field_path, key)
            if key in data_keys and isinstance(val, list):
                # Direct _sanitize_array bypasses _sanitize_value, so count it here.
                update_size_metrics(metadata.size_metrics, val)
                result[key] = self._sanitize_array(val, field_ctx, metadata, depth + 1, entry_detect)
            else:
                result[key] = self._sanitize_value(val, field_ctx, metadata, depth + 1, entry_detect)
        return result

    def _sanitize_wrapped(
        self,
        obj: dict,
        context: SanitizationContext,
        metadata: SanitizationMetadata,
        depth: int,
        detect: bool = True,
    ) -> dict:
        result: dict = {}
        for i, (key, val) in enumerate(obj.items()):
            entry_detect = detect and self._detection_allowed(i, len(obj), metadata)
            if key in DANGEROUS_KEYS:
                self._record_dangerous_key(metadata, context.path, key)
                continue
            field_path = f"{context.path}.{key}" if context.path else key
            if entry_detect:
                self._detect_in_key(key, field_path, context, metadata)
            field_ctx = self._child_context(context, field_path, key)
            if get_wrapped_data({key: val}) is not None:
                # Direct _sanitize_array bypasses _sanitize_value, so count it here.
                update_size_metrics(metadata.size_metrics, val)
                result[key] = self._sanitize_array(val, field_ctx, metadata, depth + 1, entry_detect)
            else:
                result[key] = self._sanitize_value(val, field_ctx, metadata, depth + 1, entry_detect)
        return result

    # ------------------------------------------------------------------
    # String / key detection (never mutates content)
    # ------------------------------------------------------------------

    def _sanitize_string_field(
        self,
        value: str,
        context: SanitizationContext,
        metadata: SanitizationMetadata,
        detect: bool = True,
    ) -> str:
        # Structural accounting runs even when detection is skipped.
        update_size_metrics(metadata.size_metrics, value)
        if not detect:
            return self._maybe_wrap(value, context)

        risk_level = context.risk_level
        tier1_patterns: list[str] = []
        escalated_from_encoding = False
        if context.cumulative_risk:
            context.cumulative_risk.total_fields_processed += 1

        if self._use_tier1:
            cap = self._max_field_analysis_length
            # The cap counts UTF-16 units, as in TS.
            analysis_value = truncate(value, cap)
            if analysis_value != value:
                metadata.analysis_truncated = True

            result = self._pattern_detector.analyze(analysis_value)
            if result.has_detections:
                tier1_patterns = list(dict.fromkeys(m.pattern for m in result.matches))
                if tier1_patterns:
                    self._record_tier1_match(metadata, analysis_value, result.matches)
                risk_level = _risk_ladder(risk_level, result.suggested_risk)
                if context.cumulative_risk and result.matches:
                    self._update_cumulative_risk(context.cumulative_risk, result.suggested_risk, tier1_patterns)

            enc = self._scan_decoded(analysis_value, context, metadata)
            if enc is not None:
                enc_patterns, enc_risk = enc
                tier1_patterns = list(dict.fromkeys(tier1_patterns + enc_patterns))
                escalated_from_encoding = True
                risk_level = _risk_ladder(risk_level, enc_risk)

        self._record_field_detections(metadata, context, risk_level, tier1_patterns, escalated_from_encoding)
        return self._maybe_wrap(value, context)

    def _scan_decoded(
        self,
        analysis_value: str,
        context: SanitizationContext,
        metadata: SanitizationMetadata,
        cumulative: bool = True,
    ) -> tuple[list[str], RiskLevel] | None:
        """Decode an encoded value (base64, hex, ROT13..., chained) and run the pattern detector on the
        decoded text. Evidence-driven: a decode counts only when it trips a real attack pattern, so a
        benign base64 body that merely contains the word "ignore" isn't flagged."""
        decoded, levels = decode_all_levels(analysis_value)
        if levels == 0 or decoded == analysis_value:
            return None
        enc = self._pattern_detector.analyze(decoded)
        if not enc.has_detections or not enc.matches:
            return None
        patterns = list(dict.fromkeys(m.pattern for m in enc.matches))
        self._record_tier1_match(metadata, decoded, enc.matches)
        if cumulative and context.cumulative_risk:
            self._update_cumulative_risk(context.cumulative_risk, enc.suggested_risk, patterns)
        return patterns, enc.suggested_risk

    def _record_field_detections(
        self,
        metadata: SanitizationMetadata,
        context: SanitizationContext,
        risk_level: RiskLevel,
        patterns: list[str],
        from_encoding: bool,
    ) -> None:
        """Raise the overall risk and record which patterns and methods fired on this field."""
        self._raise_overall_risk(metadata, risk_level)
        if not patterns and not from_encoding:
            return
        metadata.fields_sanitized.append(context.path)
        methods: list = []
        if patterns:
            methods.append("pattern_removal")
        if from_encoding:
            methods.append("encoding_detection")
        metadata.methods_by_field[context.path] = methods
        if patterns:
            metadata.patterns_removed_by_field[context.path] = patterns

    def _scan_encoded_field(
        self, value: str, context: SanitizationContext, metadata: SanitizationMetadata, detect: bool
    ) -> None:
        """A field that isn't risky by name isn't pattern-scanned, but an encoded value in it is decoded
        and scanned (ENG-4342): an attack can hide in any field as base64, e.g. a Gmail body in
        ``payload.body.data``."""
        if not detect or not self._use_tier1:
            return
        cap = self._max_field_analysis_length
        # Only this field escalates: these fields aren't counted as processed in the cumulative tracker,
        # so feeding it their matches would skew its fractions.
        enc = self._scan_decoded(truncate(value, cap), context, metadata, cumulative=False)
        if enc is not None:
            patterns, risk = enc
            self._record_field_detections(metadata, context, _risk_ladder(context.risk_level, risk), patterns, True)

    @staticmethod
    def _record_tier1_match(metadata: SanitizationMetadata, text: str, matches: list[PatternMatch]) -> None:
        """Record what Tier 1 matched, for the Tier 2 gate. Tier 1's normalised copy (leet, split or
        joined words) can hold occurrences of a phrase the raw text doesn't. When any occurrence in the
        copy, with its context, isn't in the raw text, Tier 2 also gets that copy."""
        metadata.tier1_match_texts.append(text)
        keys: list[str] = []
        for phrase in dict.fromkeys(m.matched for m in matches):
            metadata.tier1_match_phrases.append((text, phrase))
            k = phrase_key(phrase)
            if k:
                keys.append(k)
        if not keys or not any(m.normalised for m in matches):
            return
        raw = phrase_key(text)
        for copy in dict.fromkeys(tier1_analysis_copy(text, alternate) for alternate in (False, True)):
            key = phrase_key(copy)
            if copy != text and any(_has_occurrence_not_in(key, raw, k) for k in keys):
                metadata.tier1_normalised_copies.append((text, copy))

    def _detect_in_key(self, key: str, key_path: str, context: SanitizationContext, metadata: SanitizationMetadata) -> None:
        """Detect injection hidden in an object key. Keys are never rewritten."""
        if not self._use_tier1 or len(key) < 3:
            return
        cap = self._max_field_analysis_length
        analysis_key = truncate(key, cap)
        if analysis_key != key:
            metadata.analysis_truncated = True
        result = self._pattern_detector.analyze(analysis_key)
        if not result.has_detections or not result.matches:
            return
        patterns = [m.pattern for m in result.matches]
        metadata.tier1_match_texts.append(analysis_key)
        path = f"{key_path} (key)"
        metadata.fields_sanitized.append(path)
        metadata.methods_by_field[path] = ["pattern_removal"]
        metadata.patterns_removed_by_field[path] = patterns
        self._raise_overall_risk(metadata, result.suggested_risk)
        if context.cumulative_risk:
            context.cumulative_risk.total_fields_processed += 1
            self._update_cumulative_risk(context.cumulative_risk, result.suggested_risk, patterns)

    def _maybe_wrap(self, value: str, context: SanitizationContext) -> str:
        if self._annotate_boundary and context.boundary is not None:
            return wrap_with_boundary(value, context.boundary)
        return value

    def _raise_overall_risk(self, metadata: SanitizationMetadata, level: RiskLevel) -> None:
        if RISK_ORDER.index(level) > RISK_ORDER.index(metadata.overall_risk_level):
            metadata.overall_risk_level = level

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _is_field_risky(self, field_name: str, tool_name: str) -> bool:
        return is_risky_field(field_name, self._risky_fields, tool_name)

    def _create_cumulative_tracker(self) -> CumulativeRiskTracker:
        return CumulativeRiskTracker(escalation_threshold=dict(self._cumulative_thresholds))

    @staticmethod
    def _update_cumulative_risk(tracker: CumulativeRiskTracker, risk_level: RiskLevel, patterns: list[str]) -> None:
        if risk_level in ("high", "critical"):
            tracker.high_risk_count += 1
        elif risk_level == "medium":
            tracker.medium_risk_count += 1
        if patterns:
            tracker.suspicious_patterns.extend(patterns)

    @staticmethod
    def _should_escalate(tracker: CumulativeRiskTracker) -> bool:
        thresholds = tracker.escalation_threshold
        high_threshold = int(thresholds.get("high", DEFAULT_CUMULATIVE_RISK_THRESHOLDS["high"]))
        medium_threshold = int(thresholds.get("medium", DEFAULT_CUMULATIVE_RISK_THRESHOLDS["medium"]))
        patterns_threshold = int(thresholds.get("patterns", DEFAULT_CUMULATIVE_RISK_THRESHOLDS["patterns"]))
        medium_fraction = float(thresholds.get("medium_fraction", DEFAULT_CUMULATIVE_RISK_THRESHOLDS["medium_fraction"]))
        patterns_fraction = float(thresholds.get("patterns_fraction", DEFAULT_CUMULATIVE_RISK_THRESHOLDS["patterns_fraction"]))
        if tracker.high_risk_count >= high_threshold:
            return True
        total = max(tracker.total_fields_processed, 1)
        if tracker.medium_risk_count >= medium_threshold and (tracker.medium_risk_count / total) >= medium_fraction:
            return True
        if len(tracker.suspicious_patterns) >= patterns_threshold and (len(tracker.suspicious_patterns) / total) >= patterns_fraction:
            return True
        return False

    @staticmethod
    def _record_dangerous_key(metadata: SanitizationMetadata, parent_path: str, key: str) -> None:
        key_path = f"{parent_path}.{key}" if parent_path else key
        metadata.dangerous_keys_removed.append(key_path)

    @staticmethod
    def _extract_vertical(tool_name: str) -> str:
        parts = tool_name.split("_")
        if len(parts) >= 2:
            return parts[1] if parts[0] == "unified" else parts[0]
        return "unknown"

    @staticmethod
    def _extract_resource(tool_name: str) -> str:
        parts = tool_name.split("_")
        if len(parts) >= 3:
            return parts[-1]
        return "unknown"


def create_tool_result_sanitizer(**kwargs) -> ToolResultSanitizer:
    return ToolResultSanitizer(**kwargs)


def sanitize_tool_result(value: Any, tool_name: str, **kwargs) -> SanitizationResult:
    """Convenience: create a sanitizer and run it in one call."""
    sanitizer = create_tool_result_sanitizer(**kwargs)
    return sanitizer.sanitize(value, tool_name=tool_name)
