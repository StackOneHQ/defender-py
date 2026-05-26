"""Composite Sanitizer.

Risk-based sanitization that combines multiple methods based on risk level.
"""

from __future__ import annotations

from ..types import DataBoundary, FieldSanitizationResult, RiskLevel, SanitizationMethod
from ..utils.boundary import generate_data_boundary, wrap_with_boundary
from .encoding_detector import contains_suspicious_encoding, redact_all_encoding
from .normalizer import contains_suspicious_unicode, normalize_unicode
from .pattern_remover import remove_patterns
from .role_stripper import contains_role_markers, strip_role_markers


class Sanitizer:
    """Composite Sanitizer.

    Applies sanitization methods based on risk level:
    - Low: Unicode normalization; boundary wrapping only if ``annotate_boundary``
    - Medium: + Role stripping + pattern removal
    - High: + Encoding detection and redaction
    - Critical: Block (returns empty or error indicator)

    Boundary ``[UD-*]`` wrapping is off by default. Pass ``annotate_boundary=True``
    or use explicit ``methods`` including ``boundary_annotation`` (escape hatch).
    """

    def __init__(
        self,
        *,
        always_normalize: bool = True,
        annotate_boundary: bool = False,
        default_boundary: DataBoundary | None = None,
        redaction_text: str = "[REDACTED]",
        encoding_redaction_text: str = "[ENCODED DATA]",
        include_original: bool = False,
    ):
        self._always_normalize = always_normalize
        self._annotate_boundary = annotate_boundary
        self._default_boundary = default_boundary
        self._redaction_text = redaction_text
        self._encoding_redaction_text = encoding_redaction_text
        self._include_original = include_original

    def sanitize(
        self,
        text: str,
        *,
        risk_level: RiskLevel,
        boundary: DataBoundary | None = None,
        methods: list[SanitizationMethod] | None = None,
        field_name: str | None = None,
    ) -> FieldSanitizationResult:
        if not text:
            return FieldSanitizationResult(
                original=text if self._include_original else "",
                sanitized=text or "",
                methods_applied=[],
                patterns_removed=[],
                risk_level=risk_level,
            )

        if risk_level == "critical":
            return self._block_content(text, risk_level)

        if methods:
            return self._apply_specific_methods(text, methods, boundary, risk_level)

        return self._apply_risk_based_methods(text, risk_level, boundary)

    def _apply_risk_based_methods(
        self, text: str, risk_level: RiskLevel, boundary: DataBoundary | None
    ) -> FieldSanitizationResult:
        result = text
        methods_applied: list[SanitizationMethod] = []
        patterns_removed: list[str] = []

        # Step 1: Unicode normalization
        if self._always_normalize or risk_level != "low":
            result = normalize_unicode(result)
            methods_applied.append("unicode_normalization")

        # Step 2: Role stripping (medium+)
        if risk_level in ("medium", "high"):
            if contains_role_markers(result):
                result = strip_role_markers(result)
                methods_applied.append("role_stripping")

        # Step 3: Pattern removal (medium+)
        if risk_level in ("medium", "high"):
            pr = remove_patterns(
                result,
                replacement=self._redaction_text,
                high_severity_only=(risk_level == "medium"),
            )
            if pr.replacement_count > 0:
                result = pr.text
                patterns_removed.extend(pr.patterns_removed)
                methods_applied.append("pattern_removal")

        # Step 4: Encoding detection (high only)
        if risk_level == "high":
            if contains_suspicious_encoding(result):
                result = redact_all_encoding(result, self._encoding_redaction_text)
                methods_applied.append("encoding_detection")

        # Step 5: Boundary annotation (opt-in; off by default)
        if self._annotate_boundary:
            b = boundary or self._default_boundary or generate_data_boundary()
            result = wrap_with_boundary(result, b)
            methods_applied.append("boundary_annotation")

        return FieldSanitizationResult(
            original=text if self._include_original else "",
            sanitized=result,
            methods_applied=methods_applied,
            patterns_removed=patterns_removed,
            risk_level=risk_level,
        )

    def _apply_specific_methods(
        self, text: str, methods: list[SanitizationMethod], boundary: DataBoundary | None, risk_level: RiskLevel
    ) -> FieldSanitizationResult:
        result = text
        methods_applied: list[SanitizationMethod] = []
        patterns_removed: list[str] = []

        for method in methods:
            if method == "unicode_normalization":
                result = normalize_unicode(result)
                methods_applied.append(method)
            elif method == "role_stripping":
                result = strip_role_markers(result)
                methods_applied.append(method)
            elif method == "pattern_removal":
                pr = remove_patterns(result, replacement=self._redaction_text)
                result = pr.text
                patterns_removed.extend(pr.patterns_removed)
                methods_applied.append(method)
            elif method == "encoding_detection":
                result = redact_all_encoding(result, self._encoding_redaction_text)
                methods_applied.append(method)
            elif method == "boundary_annotation":
                # Explicit method list — honored even when annotate_boundary is False.
                b = boundary or self._default_boundary or generate_data_boundary()
                result = wrap_with_boundary(result, b)
                methods_applied.append(method)

        return FieldSanitizationResult(
            original=text if self._include_original else "",
            sanitized=result,
            methods_applied=methods_applied,
            patterns_removed=patterns_removed,
            risk_level=risk_level,
        )

    def sanitize_default(self, text: str, boundary: DataBoundary | None = None) -> FieldSanitizationResult:
        """Convenience: sanitize with medium risk."""
        return self.sanitize(text, risk_level="medium", boundary=boundary)

    def sanitize_light(self, text: str, boundary: DataBoundary | None = None) -> FieldSanitizationResult:
        """Convenience: sanitize with low risk."""
        return self.sanitize(text, risk_level="low", boundary=boundary)

    def sanitize_aggressive(self, text: str, boundary: DataBoundary | None = None) -> FieldSanitizationResult:
        """Convenience: sanitize with high risk."""
        return self.sanitize(text, risk_level="high", boundary=boundary)

    def _block_content(self, text: str, risk_level: RiskLevel) -> FieldSanitizationResult:
        return FieldSanitizationResult(
            original=text if self._include_original else "",
            sanitized="[CONTENT BLOCKED FOR SECURITY]",
            methods_applied=[],
            patterns_removed=[],
            risk_level=risk_level,
        )


def create_sanitizer(**kwargs) -> Sanitizer:
    return Sanitizer(**kwargs)


def sanitize_text(text: str, risk_level: RiskLevel = "medium", boundary: DataBoundary | None = None) -> str:
    s = create_sanitizer()
    result = s.sanitize(text, risk_level=risk_level, boundary=boundary)
    return result.sanitized


def suggest_risk_level(text: str) -> RiskLevel:
    if not text:
        return "low"
    risk_score = 0
    if contains_suspicious_unicode(text):
        risk_score += 1
    if contains_role_markers(text):
        risk_score += 2
    if contains_suspicious_encoding(text):
        risk_score += 2
    keywords = ["ignore previous", "forget instructions", "you are now", "system:", "bypass", "jailbreak"]
    lower = text.lower()
    for kw in keywords:
        if kw in lower:
            risk_score += 2
    if risk_score >= 6:
        return "critical"
    if risk_score >= 4:
        return "high"
    if risk_score >= 2:
        return "medium"
    return "low"
