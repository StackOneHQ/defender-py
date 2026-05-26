"""Core types for the Prompt Defense Framework."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Literal, Union

RiskLevel = Literal["low", "medium", "high", "critical"]

PatternCategory = Literal[
    "role_marker",
    "instruction_override",
    "role_assumption",
    "security_bypass",
    "command_execution",
    "encoding_suspicious",
    "structural",
]

SanitizationMethod = Literal[
    "unicode_normalization",
    "boundary_annotation",
    "role_stripping",
    "pattern_removal",
    "encoding_detection",
]

StructureType = Literal["array", "object", "wrapped", "primitive", "null"]

SanitizableValue = Union[str, int, float, bool, None, list, dict]


@dataclass
class PatternMatch:
    pattern: str
    matched: str
    position: int
    category: PatternCategory
    severity: Literal["low", "medium", "high"]


@dataclass
class StructuralFlag:
    type: Literal["high_entropy", "excessive_length", "suspicious_formatting", "nested_markers"]
    details: str
    severity: Literal["low", "medium", "high"]


@dataclass
class Tier1Result:
    matches: list[PatternMatch]
    structural_flags: list[StructuralFlag]
    has_detections: bool
    suggested_risk: RiskLevel
    latency_ms: float


@dataclass
class Tier2Result:
    score: float
    confidence: float
    skipped: bool
    skip_reason: str | None = None
    latency_ms: float = 0.0


@dataclass
class DataBoundary:
    id: str
    start_tag: str
    end_tag: str


@dataclass
class CumulativeRiskTracker:
    medium_risk_count: int = 0
    high_risk_count: int = 0
    suspicious_patterns: list[str] = field(default_factory=list)
    total_fields_processed: int = 0
    escalation_threshold: dict[str, int | float] = field(
        default_factory=lambda: {
            "medium": 3,
            "high": 1,
            "patterns": 3,
            "medium_fraction": 0.25,
            "patterns_fraction": 0.25,
        }
    )


@dataclass
class SanitizationContext:
    path: str
    field_name: str
    tool_name: str
    vertical: str
    resource: str
    risk_level: RiskLevel
    boundary: DataBoundary | None = None
    cumulative_risk: CumulativeRiskTracker | None = None


@dataclass
class FieldSanitizationResult:
    original: str
    sanitized: str
    methods_applied: list[SanitizationMethod]
    patterns_removed: list[str]
    risk_level: RiskLevel


@dataclass
class SizeMetrics:
    estimated_bytes: int = 0
    string_count: int = 0
    object_count: int = 0
    array_count: int = 0
    size_limit_hit: bool = False
    depth_limit_hit: bool = False


@dataclass
class SanitizationMetadata:
    fields_sanitized: list[str] = field(default_factory=list)
    methods_by_field: dict[str, list[SanitizationMethod]] = field(default_factory=dict)
    patterns_removed_by_field: dict[str, list[str]] = field(default_factory=dict)
    overall_risk_level: RiskLevel = "medium"
    cumulative_risk_escalated: bool = False
    total_latency_ms: float = 0.0
    size_metrics: SizeMetrics = field(default_factory=SizeMetrics)
    # Leaf dict keys Tier 1 identified as risky string fields (telemetry / diagnostics).
    risky_field_names: list[str] = field(default_factory=list)
    # Paths of keys removed due to prototype-pollution risk.
    dangerous_keys_removed: list[str] = field(default_factory=list)


@dataclass
class SanitizationResult:
    sanitized: Any
    metadata: SanitizationMetadata


@dataclass
class PatternDefinition:
    id: str
    pattern: re.Pattern
    category: PatternCategory
    severity: Literal["low", "medium", "high"]
    description: str


@dataclass
class RiskyFieldConfig:
    field_names: list[str] = field(default_factory=list)
    field_patterns: list[re.Pattern] = field(default_factory=list)
    tool_overrides: dict[str, list[str]] | None = None


@dataclass
class TraversalConfig:
    max_depth: int = 10
    max_size: int = 10 * 1024 * 1024  # 10MB
    large_array_threshold: int = 1000
    skip_large_arrays: bool = True


@dataclass
class Tier2Config:
    high_risk_threshold: float = 0.8
    medium_risk_threshold: float = 0.5
    skip_below_size: int = 50
    min_text_length: int = 10
    max_text_length: int = 10000
    onnx_model_path: str | None = None
    # Tier 2 extraction scope (SFE-filtered payload when SFE is on).
    # ``None`` or empty list: all strings (matches TypeScript when ``tier2Fields`` is unset).
    # Non-empty list: only strings under those dict keys (full-depth collect).
    tier2_fields: list[str] | None = None


@dataclass
class PromptDefenseConfig:
    risky_fields: RiskyFieldConfig = field(default_factory=RiskyFieldConfig)
    traversal: TraversalConfig = field(default_factory=TraversalConfig)
    cumulative_risk_thresholds: dict[str, int | float] = field(
        default_factory=lambda: {
            "medium": 3,
            "high": 1,
            "patterns": 3,
            "medium_fraction": 0.25,
            "patterns_fraction": 0.25,
        }
    )
    tier2: Tier2Config = field(default_factory=Tier2Config)
    block_high_risk: bool = False


@dataclass
class DefenseResult:
    """Outcome of ``defend_tool_result`` (Tier 1 sanitize + optional Tier 2 + SFE metadata).

    ``fields_dropped`` (when SFE is enabled) lists field paths removed from the **Tier 2**
    classifier input only; they are **not** stripped from ``sanitized``.
    """

    allowed: bool
    risk_level: RiskLevel
    sanitized: Any
    detections: list[str]
    fields_sanitized: list[str]
    patterns_by_field: dict[str, list[str]]
    tier2_score: float | None = None
    tier2_skip_reason: str | None = None
    max_sentence: str | None = None
    fields_dropped: list[str] = field(default_factory=list)
    truncated_at_depth: bool | None = None
    latency_ms: float = 0.0
