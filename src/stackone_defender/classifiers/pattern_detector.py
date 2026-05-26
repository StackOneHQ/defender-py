"""Tier 1: Pattern Detection.

Fast, regex-based detection of known injection patterns.
Target latency: < 1-2ms per field.
"""

from __future__ import annotations

import math
import re
import time
import unicodedata

from ..sanitizers.leet_normalizer import normalize_leet_speak
from ..sanitizers.normalizer import normalize_unicode, normalize_whitespace, strip_combining_marks
from ..types import PatternDefinition, PatternMatch, RiskLevel, StructuralFlag, Tier1Result
from .patterns import ALL_PATTERNS, contains_filter_keywords

DEFAULT_DETECTOR_CONFIG = {
    "use_fast_filter": True,
    "max_analysis_length": 50000,
    "entropy_threshold": 4.5,
    "entropy_min_length": 50,
    "max_field_length": 100000,
}


class PatternDetector:
    """Pattern Detector for Tier 1 classification."""

    def __init__(self, config: dict | None = None, custom_patterns: list[PatternDefinition] | None = None):
        cfg = dict(DEFAULT_DETECTOR_CONFIG)
        if config:
            cfg.update(config)
        self._use_fast_filter = cfg["use_fast_filter"]
        self._max_analysis_length = cfg["max_analysis_length"]
        self._entropy_threshold = cfg["entropy_threshold"]
        self._entropy_min_length = cfg["entropy_min_length"]
        self._max_field_length = cfg["max_field_length"]
        self._patterns: list[PatternDefinition] = list(ALL_PATTERNS)
        self._has_custom = False
        if custom_patterns:
            self._patterns.extend(custom_patterns)
            self._has_custom = True

    def analyze(self, text: str) -> Tier1Result:
        start = time.perf_counter()

        if not text or len(text) < 3:
            return self._empty_result(start)

        original_length = len(text)
        raw_text = text[: self._max_analysis_length] if len(text) > self._max_analysis_length else text

        # Normalisation chain: collapse obfuscation before injection pattern
        # matching. Order matters:
        # 1. NFD-decompose: precomposed accents become base + combining mark.
        # 2. strip_combining_marks: Zalgo defense + accent stripping.
        # 3. normalize_unicode: homoglyphs/fullwidth -> ASCII.
        # 4. normalize_whitespace: collapse spaced letters + embedded newlines.
        # 5. normalize_leet_speak: 1gn0r3 -> ignore.
        # NFD-decomposition lives here (not in normalize_unicode) because it
        # strips legitimate accents like ``café`` -> ``cafe`` -- fine for
        # analysis but would be data loss if returned to callers. The result
        # is analysis-only and never returned.
        analysis_text = normalize_leet_speak(
            normalize_whitespace(
                normalize_unicode(strip_combining_marks(unicodedata.normalize("NFD", raw_text)))
            )
        )

        # Fast filter: short-circuit if neither raw nor normalised text
        # contains keywords. Raw text is checked to preserve detection of
        # obfuscation patterns (e.g. invisible unicode, leet-speak variants)
        # that are normalised away before injection patterns run. Disable the
        # fast filter when custom patterns are provided -- callers may add
        # patterns whose keywords aren't in the static list.
        should_use_fast_filter = self._use_fast_filter and not self._has_custom
        raw_has_keywords = not should_use_fast_filter or contains_filter_keywords(raw_text)
        norm_has_keywords = not should_use_fast_filter or contains_filter_keywords(analysis_text)

        if not raw_has_keywords and not norm_has_keywords:
            flags = self._detect_structural_issues(raw_text, original_length)
            return self._create_result([], flags, start)

        # Short-circuit: if normalisation produced no change, a single pass
        # is sufficient and avoids doubling pattern work for plain-text input.
        if raw_text == analysis_text:
            matches = self._detect_patterns(raw_text) if raw_has_keywords else []
            flags = self._detect_structural_issues(raw_text, original_length)
            return self._create_result(matches, flags, start)

        # Run patterns on raw text -- catches obfuscation-specific patterns
        # (e.g. invisible_unicode, leetspeak_injection) that normalisation
        # removes. Run whenever EITHER pass has keywords: if only the
        # normalised text has keywords (pure leet-speak with no other
        # fast-filter hits), we still want the raw pass to fire obfuscation
        # patterns like leetspeak_injection.
        raw_matches = (
            self._detect_patterns(raw_text) if (raw_has_keywords or norm_has_keywords) else []
        )

        # Run patterns on normalised text -- catches injection patterns
        # hidden behind leet-speak, whitespace, or homoglyph obfuscation.
        # Matches are tagged ``normalised=True`` because their
        # position/matched values reference the transformed text.
        norm_matches_raw = self._detect_patterns(analysis_text) if norm_has_keywords else []
        norm_matches = [
            PatternMatch(
                pattern=m.pattern,
                matched=m.matched,
                position=m.position,
                category=m.category,
                severity=m.severity,
                normalised=True,
            )
            for m in norm_matches_raw
        ]

        # Merge: normalised matches take priority. Raw-only matches are
        # appended for patterns that fired on the original text but not the
        # normalised form (e.g. obfuscation-detection patterns that match the
        # raw encoding characters).
        seen_patterns = {m.pattern for m in norm_matches}
        merged_matches: list[PatternMatch] = [*norm_matches]
        merged_matches.extend(m for m in raw_matches if m.pattern not in seen_patterns)

        flags = self._detect_structural_issues(raw_text, original_length)
        return self._create_result(merged_matches, flags, start)

    # ------------------------------------------------------------------
    # Pattern detection
    # ------------------------------------------------------------------

    def _detect_patterns(self, text: str) -> list[PatternMatch]:
        matches: list[PatternMatch] = []
        for defn in self._patterns:
            for m in defn.pattern.finditer(text):
                matches.append(
                    PatternMatch(
                        pattern=defn.id,
                        matched=m.group(0),
                        position=m.start(),
                        category=defn.category,
                        severity=defn.severity,
                    )
                )
        return matches

    # ------------------------------------------------------------------
    # Structural analysis
    # ------------------------------------------------------------------

    def _detect_structural_issues(self, text: str, original_length: int | None = None) -> list[StructuralFlag]:
        flags: list[StructuralFlag] = []
        length_to_check = original_length if original_length is not None else len(text)

        if length_to_check > self._max_field_length:
            flags.append(
                StructuralFlag(
                    type="excessive_length",
                    details=f"Field length {length_to_check} exceeds maximum {self._max_field_length}",
                    severity="medium",
                )
            )

        if len(text) >= self._entropy_min_length:
            entropy = self._calculate_entropy(text)
            if entropy > self._entropy_threshold:
                flags.append(
                    StructuralFlag(
                        type="high_entropy",
                        details=f"Entropy {entropy:.2f} exceeds threshold {self._entropy_threshold}",
                        severity="medium",
                    )
                )

        if self._has_nested_markers(text):
            flags.append(
                StructuralFlag(
                    type="nested_markers",
                    details="Suspicious nested XML tags or bracket patterns detected",
                    severity="medium",
                )
            )

        if self._has_suspicious_formatting(text):
            flags.append(
                StructuralFlag(
                    type="suspicious_formatting",
                    details="Unusual formatting patterns detected",
                    severity="low",
                )
            )

        return flags

    @staticmethod
    def _calculate_entropy(text: str) -> float:
        freq: dict[str, int] = {}
        for ch in text:
            freq[ch] = freq.get(ch, 0) + 1
        length = len(text)
        entropy = 0.0
        for count in freq.values():
            p = count / length
            entropy -= p * math.log2(p)
        return entropy

    @staticmethod
    def _has_nested_markers(text: str) -> bool:
        suspicious_xml = re.compile(
            r"</?(?:system|user|assistant|instruction|prompt|admin|developer)[^>]*>", re.I
        )
        tags = suspicious_xml.findall(text)
        if len(tags) >= 2:
            return True

        xml_tags = re.findall(r"<[a-zA-Z][^>]*>", text)
        if len(xml_tags) > 4:
            marker_tags = [t for t in xml_tags if re.search(r"system|user|assistant|instruction|prompt", t, re.I)]
            if marker_tags:
                return True

        if re.search(r"\[\[.*?(?:system|instruction|ignore).*?\]\]", text, re.I):
            return True

        return False

    @staticmethod
    def _has_suspicious_formatting(text: str) -> bool:
        if re.search(r"\n{3,}(?:system|instruction|ignore|forget)", text, re.I):
            return True
        if re.search(r"^#{1,3}\s*(?:system|instruction|new rules)", text, re.I | re.M):
            return True
        if re.search(r"[-=]{3,}\s*\n\s*(?:system|instruction|ignore)", text, re.I):
            return True
        return False

    # ------------------------------------------------------------------
    # Risk calculation
    # ------------------------------------------------------------------

    @staticmethod
    def _calculate_suggested_risk(matches: list[PatternMatch], flags: list[StructuralFlag]) -> RiskLevel:
        high_matches = sum(1 for m in matches if m.severity == "high")
        medium_matches = sum(1 for m in matches if m.severity == "medium")
        high_flags = sum(1 for f in flags if f.severity == "high")
        medium_flags = sum(1 for f in flags if f.severity == "medium")

        if high_matches >= 2 or (high_matches >= 1 and high_flags >= 1):
            return "critical"
        if high_matches >= 1 or medium_matches >= 3 or (medium_matches >= 2 and medium_flags >= 1):
            return "high"
        if medium_matches >= 1 or high_flags >= 1 or medium_flags >= 2:
            return "medium"
        if matches or flags:
            return "low"
        return "low"

    # ------------------------------------------------------------------
    # Result helpers
    # ------------------------------------------------------------------

    def _create_result(self, matches: list[PatternMatch], flags: list[StructuralFlag], start: float) -> Tier1Result:
        return Tier1Result(
            matches=matches,
            structural_flags=flags,
            has_detections=bool(matches) or bool(flags),
            suggested_risk=self._calculate_suggested_risk(matches, flags),
            latency_ms=(time.perf_counter() - start) * 1000,
        )

    @staticmethod
    def _empty_result(start: float) -> Tier1Result:
        return Tier1Result(
            matches=[],
            structural_flags=[],
            has_detections=False,
            suggested_risk="low",
            latency_ms=(time.perf_counter() - start) * 1000,
        )

    def add_pattern(self, pattern: PatternDefinition) -> None:
        self._patterns.append(pattern)

    def get_patterns(self) -> list[PatternDefinition]:
        return list(self._patterns)


def create_pattern_detector(config: dict | None = None, custom_patterns: list[PatternDefinition] | None = None) -> PatternDetector:
    return PatternDetector(config=config, custom_patterns=custom_patterns)
