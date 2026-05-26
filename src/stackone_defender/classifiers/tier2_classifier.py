"""Tier 2 Classifier: ML-based prompt injection detection (ONNX).

Supports single-head and multi-head (``[batch, 2]``) ONNX models, with
post-hoc temperature scaling and optional model-bundled calibration
defaults read from ``<model_dir>/classifier_config.json``.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from dataclasses import dataclass
from typing import Any

from ..types import MultiheadConfig, RiskLevel, Tier2Result
from ..utils.boundary import strip_boundary_patterns
from .onnx_classifier import OnnxClassifier, get_default_model_path

_logger = logging.getLogger(__name__)

DEFAULT_TIER2_CLASSIFIER_CONFIG: dict[str, Any] = {
    "high_risk_threshold": 0.8,
    "medium_risk_threshold": 0.5,
    "min_text_length": 10,
    "max_text_length": 10000,
}


@dataclass
class _ModelCalibrationDefaults:
    """Subset of ``classifier_config.json`` defender consumes at runtime."""

    temperature_t: float | None = None
    high_risk_threshold: float | None = None
    medium_risk_threshold: float | None = None


# Module-level memo of ``classifier_config.json`` per resolved model dir.
# Bundled model assets are immutable at runtime, so the sync FS read +
# ``json.loads`` is amortized to once per process per model dir. ``None`` is
# a valid cached value ("no calibration block for this model"), so we probe
# with ``in`` not by sentinel comparison.
_calibration_cache: dict[str, _ModelCalibrationDefaults | None] = {}


def _read_calibration_defaults(model_dir: str) -> _ModelCalibrationDefaults | None:
    """Read calibration defaults from ``<model_dir>/classifier_config.json``.

    Returns ``None`` for a missing file (legacy models) or an absent
    ``calibration`` key. Other read or parse failures emit a warning so they
    don't silently fall back to library defaults -- a typo in a shipped
    calibration block would otherwise be invisible until someone digs into
    decision divergence. Memoized per ``model_dir`` (resolved path).
    """
    key = os.path.realpath(model_dir)
    if key in _calibration_cache:
        return _calibration_cache[key]

    config_path = os.path.join(key, "classifier_config.json")
    try:
        with open(config_path, encoding="utf-8") as f:
            raw = f.read()
    except FileNotFoundError:
        _calibration_cache[key] = None
        return None
    except OSError as e:
        _logger.warning("[defender] failed to read %s: %s", config_path, e)
        _calibration_cache[key] = None
        return None

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        _logger.warning("[defender] malformed classifier_config.json at %s: %s", config_path, e)
        _calibration_cache[key] = None
        return None

    calibration = data.get("calibration") if isinstance(data, dict) else None
    if not isinstance(calibration, dict):
        _calibration_cache[key] = None
        return None

    result = _ModelCalibrationDefaults(
        temperature_t=_coerce_float(calibration.get("temperatureT")),
        high_risk_threshold=_coerce_float(calibration.get("highRiskThreshold")),
        medium_risk_threshold=_coerce_float(calibration.get("mediumRiskThreshold")),
    )
    _calibration_cache[key] = result
    return result


def _coerce_float(value: Any) -> float | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return None


class Tier2Classifier:
    """Tier 2 Classifier using ONNX inference.

    Three-tier precedence for thresholds and temperature:

    1. Hardcoded library defaults (:data:`DEFAULT_TIER2_CLASSIFIER_CONFIG`).
    2. Model-specific defaults from ``<model_dir>/classifier_config.json:calibration``.
    3. Caller-provided ``config`` (always wins).

    Model defaults let new models ship with their fitted ``T`` and
    thresholds baked in without the library needing to know which model the
    caller is loading. Legacy models without a ``classifier_config.json``
    skip step 2 transparently.
    """

    def __init__(self, config: dict | None = None):
        config = dict(config) if config else {}

        model_path = config.get("onnx_model_path") or get_default_model_path()
        model_defaults = _read_calibration_defaults(model_path)

        merged: dict[str, Any] = dict(DEFAULT_TIER2_CLASSIFIER_CONFIG)
        merged["onnx_model_path"] = model_path
        # multihead and temperature_t default to None at this layer; the
        # OnnxClassifier ignores ``None`` and uses T=1.0.
        merged["temperature_t"] = None
        merged["multihead"] = None

        if model_defaults is not None:
            if model_defaults.temperature_t is not None:
                merged["temperature_t"] = model_defaults.temperature_t
            if model_defaults.high_risk_threshold is not None:
                merged["high_risk_threshold"] = model_defaults.high_risk_threshold
            if model_defaults.medium_risk_threshold is not None:
                merged["medium_risk_threshold"] = model_defaults.medium_risk_threshold

        # Caller config wins, but filter out explicit ``None`` keys first.
        # A naive ``{**merged, **config}`` would let ``{"temperature_t": None}``
        # (common when building config conditionally from optional settings)
        # silently clobber a model-loaded calibration value -- and a ``None``
        # ``temperature_t`` then bypasses OnnxClassifier's positive-finite
        # guard, dropping calibration back to T=1.
        defined_config = {k: v for k, v in config.items() if v is not None}
        merged.update(defined_config)

        self._high_risk_threshold: float = float(merged["high_risk_threshold"])
        self._medium_risk_threshold: float = float(merged["medium_risk_threshold"])
        self._min_text_length: int = int(merged["min_text_length"])
        self._max_text_length: int = int(merged["max_text_length"])
        self._model_path: str = merged["onnx_model_path"]
        self._temperature_t: float | None = merged.get("temperature_t")
        self._multihead: MultiheadConfig | None = merged.get("multihead")

        self._onnx = OnnxClassifier(self._model_path, self._temperature_t)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def is_ready(self) -> bool:
        return self._onnx.is_loaded()

    def warmup(self) -> None:
        self._onnx.warmup()

    # ------------------------------------------------------------------
    # Single-text classify
    # ------------------------------------------------------------------

    def classify(self, text: str) -> Tier2Result:
        start = time.perf_counter()
        # Strip defender's own boundary markers before tokenization so nested
        # tool-call chains and spoofed boundary patterns don't corrupt scores.
        text = strip_boundary_patterns(text)
        if len(text) < self._min_text_length:
            return Tier2Result(
                score=0,
                confidence=0,
                skipped=True,
                skip_reason=f"Text too short ({len(text)} < {self._min_text_length})",
                latency_ms=_ms(start),
            )

        analysis_text = text[: self._max_text_length] if len(text) > self._max_text_length else text

        try:
            main, aux = self._onnx.classify_pair(analysis_text)
            confidence = abs(main - 0.5) * 2
            return Tier2Result(
                score=main, confidence=confidence, skipped=False, latency_ms=_ms(start), aux=aux
            )
        except Exception as e:
            return Tier2Result(
                score=0,
                confidence=0,
                skipped=True,
                skip_reason=f"Classification error: {e}",
                latency_ms=_ms(start),
            )

    def classify_batch(self, texts: list[str]) -> list[Tier2Result]:
        return [self.classify(t) for t in texts]

    # ------------------------------------------------------------------
    # Sentence / chunk classification
    # ------------------------------------------------------------------

    def classify_by_sentence(self, text: str) -> dict[str, Any]:
        """Classify text by sentence and return max main score."""
        start = time.perf_counter()
        text = strip_boundary_patterns(text)
        sentences = _split_into_sentences(text)
        if not sentences:
            return _skipped(start, "No sentences found")

        original_sentences: list[str] = []
        classifiable: list[str] = []
        for sentence in sentences:
            if len(sentence) < self._min_text_length:
                continue
            original_sentences.append(sentence)
            classifiable.append(
                sentence[: self._max_text_length] if len(sentence) > self._max_text_length else sentence
            )

        if not classifiable:
            return _skipped(start, "No classifiable sentences")

        try:
            pairs = self._onnx.classify_batch_pair(classifiable)
        except Exception as e:
            return _skipped(start, f"Classification error: {e}")

        sentence_scores: list[dict[str, Any]] = []
        max_score = 0.0
        max_sentence = ""
        for sentence, (main, _aux) in zip(original_sentences, pairs, strict=True):
            safe_score = main if isinstance(main, (int, float)) and main == main else 0.0
            sentence_scores.append({"sentence": sentence, "score": safe_score})
            if safe_score > max_score:
                max_score = safe_score
                max_sentence = sentence

        return {
            "score": max_score,
            "confidence": abs(max_score - 0.5) * 2,
            "skipped": False,
            "latency_ms": _ms(start),
            "max_sentence": max_sentence,
            "sentence_scores": sentence_scores,
        }

    def classify_by_chunks(self, text: str) -> dict[str, Any]:
        start = time.perf_counter()
        text = strip_boundary_patterns(text)
        if len(text) < self._min_text_length:
            return _skipped(start, "Text below minTextLength")

        model_max_len = self._onnx.get_max_length()
        bounded = text[: self._max_text_length] if len(text) > self._max_text_length else text

        try:
            self._onnx.warmup()
        except Exception as e:
            return _skipped(start, f"Warmup error: {e}")

        try:
            total_tokens = self._onnx.count_tokens(bounded)
        except Exception as e:
            return _skipped(start, f"Token count error: {e}")

        if total_tokens <= model_max_len:
            try:
                main, _aux = self._onnx.classify_pair(bounded)
            except Exception as e:
                return _skipped(start, f"Classification error: {e}")
            safe_score = main if isinstance(main, (int, float)) and main == main else 0.0
            return {
                "score": safe_score,
                "confidence": abs(safe_score - 0.5) * 2,
                "skipped": False,
                "max_sentence": bounded,
                "sentence_scores": [{"sentence": bounded, "score": safe_score}],
                "latency_ms": _ms(start),
            }

        max_content_tokens = model_max_len - 2
        sentences = [s for s in _split_into_sentences(bounded) if len(s) >= self._min_text_length]
        if not sentences:
            return _skipped(start, "No classifiable sentences")

        try:
            chunks = self._pack_sentences(sentences, max_content_tokens)
            pairs = self._onnx.classify_batch_pair(chunks)
        except Exception as e:
            return _skipped(start, f"Classification error: {e}")

        max_score = 0.0
        max_chunk = ""
        chunk_scores: list[dict[str, Any]] = []
        for i, (main, _aux) in enumerate(pairs):
            safe_score = main if isinstance(main, (int, float)) and main == main else 0.0
            chunk = chunks[i] if i < len(chunks) else ""
            chunk_scores.append({"sentence": chunk, "score": safe_score})
            if safe_score > max_score:
                max_score = safe_score
                max_chunk = chunk

        return {
            "score": max_score,
            "confidence": abs(max_score - 0.5) * 2,
            "skipped": False,
            "max_sentence": max_chunk,
            "sentence_scores": chunk_scores,
            "latency_ms": _ms(start),
        }

    def prepare_chunks(self, text: str) -> dict[str, Any]:
        text = strip_boundary_patterns(text)
        if len(text) < self._min_text_length:
            return {"chunks": [], "skipped": True, "skip_reason": "Text below minTextLength"}

        model_max_len = self._onnx.get_max_length()
        bounded = text[: self._max_text_length] if len(text) > self._max_text_length else text
        try:
            self._onnx.warmup()
        except Exception as e:
            return {"chunks": [], "skipped": True, "skip_reason": f"Warmup error: {e}"}

        if len(bounded) + 2 <= model_max_len:
            return {"chunks": [bounded], "skipped": False}

        try:
            total_tokens = self._onnx.count_tokens(bounded)
        except Exception as e:
            return {"chunks": [], "skipped": True, "skip_reason": f"Token count error: {e}"}
        if total_tokens <= model_max_len:
            return {"chunks": [bounded], "skipped": False}

        max_content_tokens = model_max_len - 2
        sentences = [s for s in _split_into_sentences(bounded) if len(s) >= self._min_text_length]
        if not sentences:
            return {"chunks": [], "skipped": True, "skip_reason": "No classifiable sentences"}
        return {"chunks": self._pack_sentences(sentences, max_content_tokens), "skipped": False}

    def classify_chunks_batch(self, chunks: list[str]) -> list[float]:
        """Single-head batch classify. Returns main-head scores only."""
        if not chunks:
            return []
        self._onnx.warmup()
        return self._onnx.classify_batch(chunks)

    def classify_chunks_batch_pair(self, chunks: list[str]) -> list[tuple[float, float | None]]:
        """Multi-head variant. Returns ``(main, aux)`` per chunk.

        Aux is ``None`` per-row for single-head models. Callers in the
        multi-head decision path use the aux scores to apply the veto rule.
        """
        if not chunks:
            return []
        self._onnx.warmup()
        return self._onnx.classify_batch_pair(chunks)

    # ------------------------------------------------------------------
    # Helpers / introspection
    # ------------------------------------------------------------------

    def _pack_sentences(self, sentences: list[str], max_content_tokens: int) -> list[str]:
        chunks: list[str] = []
        current: list[str] = []
        current_tokens = 0

        for sentence in sentences:
            sentence_tokens = self._onnx.count_tokens(sentence)
            sentence_content_tokens = max(0, sentence_tokens - 2)

            if sentence_content_tokens > max_content_tokens:
                if current:
                    chunks.append(" ".join(current))
                    current = []
                    current_tokens = 0
                chunks.append(sentence)
                continue

            if current_tokens + sentence_content_tokens > max_content_tokens:
                chunks.append(" ".join(current))
                current = [sentence]
                current_tokens = sentence_content_tokens
            else:
                current.append(sentence)
                current_tokens += sentence_content_tokens

        if current:
            chunks.append(" ".join(current))

        return chunks

    def is_injection(self, text: str, threshold: float | None = None) -> bool:
        result = self.classify(text)
        if result.skipped:
            return False
        return result.score >= (threshold if threshold is not None else self._medium_risk_threshold)

    def get_config(self) -> dict:
        return {
            "high_risk_threshold": self._high_risk_threshold,
            "medium_risk_threshold": self._medium_risk_threshold,
            "min_text_length": self._min_text_length,
            "max_text_length": self._max_text_length,
            "onnx_model_path": self._model_path,
            "temperature_t": self._temperature_t,
            "multihead": self._multihead,
        }

    def get_risk_level(self, score: float) -> RiskLevel:
        if score >= self._high_risk_threshold:
            return "high"
        if score >= self._medium_risk_threshold:
            return "medium"
        return "low"

    def get_temperature(self) -> float:
        """Temperature scaling factor in use (``1.0`` = no calibration)."""
        return self._onnx.get_temperature()

    def is_multihead(self) -> bool:
        """Whether this classifier is configured for multi-head decisions.

        Returns ``False`` when no ``multihead`` config was provided,
        regardless of what the underlying ONNX model emits.
        """
        return self._multihead is not None

    def get_multihead_config(self) -> MultiheadConfig | None:
        return self._multihead


def create_tier2_classifier(config: dict | None = None) -> Tier2Classifier:
    return Tier2Classifier(config)


def _ms(start: float) -> float:
    return (time.perf_counter() - start) * 1000


def _skipped(start: float, reason: str) -> dict[str, Any]:
    return {
        "score": 0,
        "confidence": 0,
        "skipped": True,
        "skip_reason": reason,
        "latency_ms": _ms(start),
    }


def _split_into_sentences(text: str) -> list[str]:
    """Split text into sentences for granular analysis."""
    sentences: list[str] = []
    chunks = re.split(r"(?<=[.!?])\s+|\n\n+|\n(?=[A-Z0-9#\-*])|(?<=:)\s*\n", text)
    for chunk in chunks:
        trimmed = chunk.strip()
        if not trimmed:
            continue
        if len(trimmed) > 200 and "\n" in trimmed:
            for sub in trimmed.split("\n"):
                sub = sub.strip()
                if sub:
                    sentences.append(sub)
        else:
            sentences.append(trimmed)
    return sentences
