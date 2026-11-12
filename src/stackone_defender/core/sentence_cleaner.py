"""Sentence-level cleaning for the ``sanitized`` copy.

Within a high-risk field, replace each contiguous run of high-scoring sentences
with a marker and keep the rest, so a mid-content cut stays visible to consumers
that read only ``sanitized``. Best-effort only (capped by detection) — callers
still gate on ``allowed``. Runs after Tier 2 so per-sentence scores are available.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ..classifiers.tier2_classifier import Tier2Classifier, _split_into_sentences
from ..sanitizers.role_stripper import strip_role_markers
from ..types import DataBoundary
from ..utils.boundary import strip_boundary_patterns, wrap_with_boundary

#: Inline marker left where a contiguous high-risk run was dropped.
CONTENT_SANITISED_MARKER = "[CONTENT SANITISED]"


def _clean_field(raw: str, tier2: Tier2Classifier, high_threshold: float) -> str:
    sentences = _split_into_sentences(raw)
    # A single sentence can't be isolated to a bad part, and benign opaque tokens read
    # as one sentence — leave it untouched; the verdict/``allowed`` still gates it.
    if len(sentences) <= 1:
        return raw
    # Score what Tier 2 scored: the normalised text.
    scores = tier2.classify_chunks_batch([tier2.normalize(x) for x in sentences])
    flagged = [(scores[i] if i < len(scores) else 0.0) >= high_threshold for i in range(len(sentences))]
    # Nothing dropped — return the field verbatim, never a reconstruction (a
    # rebuilt join can differ from the original and report a spurious change).
    if not any(flagged):
        return raw
    # Collapse each contiguous high-risk run into one marker, keeping surviving
    # sentences in place. All-high field → just the marker.
    parts: list[str] = []
    in_run = False
    for sentence, is_high in zip(sentences, flagged, strict=False):
        if is_high:
            if not in_run:
                parts.append(CONTENT_SANITISED_MARKER)
            in_run = True
            continue
        # Strip role markers from survivors as defense-in-depth against a sub-threshold marker.
        parts.append(strip_role_markers(sentence).strip())
        in_run = False
    return " ".join(p for p in parts if p).strip()


def clean_high_risk_content(
    content: Any,
    high_risk_values: set[str],
    tier2: Tier2Classifier,
    high_threshold: float,
    boundary: DataBoundary | None = None,
) -> tuple[Any, list[str]]:
    """Clone ``content`` (already structurally protected, optionally boundary-wrapped)
    and replace only the leaf strings whose unwrapped value is in ``high_risk_values``
    with a sentence-cleaned version. Returns ``(content, changed_fields)`` — the paths
    whose content actually changed (a single-sentence field left as-is reports none).
    Paths follow the sanitizer's convention: ``parent.key`` / ``parent[i]``."""
    if not high_risk_values:
        return content, []

    changed_fields: list[str] = []

    def walk(value: Any, path: str) -> Any:
        if isinstance(value, str):
            raw = strip_boundary_patterns(value) if boundary else value
            if raw not in high_risk_values:
                return value
            cleaned = _clean_field(raw, tier2, high_threshold)
            if cleaned != raw:
                changed_fields.append(path)
            return wrap_with_boundary(cleaned, boundary) if boundary else cleaned
        if isinstance(value, (list, tuple)):
            out = [walk(v, f"{path}[{i}]") for i, v in enumerate(value)]
            return out if isinstance(value, list) else tuple(out)
        if isinstance(value, (set, frozenset)):
            return type(value)(walk(v, f"{path}[{i}]") for i, v in enumerate(value))
        if isinstance(value, Mapping):
            return {k: walk(v, f"{path}.{k}" if path else k) for k, v in value.items()}
        return value

    return walk(content, ""), changed_fields
