"""Semantic Field Extractor (SFE) preprocessing for prompt defense."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Any, Protocol

from ..config import DANGEROUS_KEYS, MAX_TRAVERSAL_DEPTH

_logger = logging.getLogger(__name__)

_predictor_cache: dict[str, SfePredictor] = {}
_predictor_lock = Lock()
_DROPPED = object()


@dataclass
class DropDecision:
    label: str
    prob: float


class SfePredictor(Protocol):
    """FastText-compatible predictor interface."""

    def predict(self, text: str) -> DropDecision:
        ...

    def predict_batch(self, texts: list[str]) -> list[DropDecision]:
        ...


@dataclass
class SfePreprocessResult:
    filtered: Any
    dropped: list[str]
    truncated_at_depth: bool | None = None


class _FastTextPredictor:
    def __init__(self, model: Any):
        self._model = model

    def predict(self, text: str) -> DropDecision:
        labels, probs = self._model.predict(text, k=1)
        raw_label = labels[0] if labels else "__label__pass"
        prob = float(probs[0]) if probs else 0.0
        label = raw_label.replace("__label__", "")
        return DropDecision(label=label, prob=prob)

    def predict_batch(self, texts: list[str]) -> list[DropDecision]:
        return [self.predict(text) for text in texts]


def get_default_sfe_model_path() -> str:
    """Return path to the bundled quantized FastText model."""
    return str(Path(__file__).resolve().parent / "model.ftz")


def get_default_predictor(model_path: str | None = None) -> SfePredictor | None:
    """Load and cache a default predictor. Fail-open on runtime/model issues."""
    resolved = model_path or get_default_sfe_model_path()
    with _predictor_lock:
        if resolved in _predictor_cache:
            return _predictor_cache[resolved]

        try:
            import fasttext  # type: ignore
        except Exception:
            _logger.warning(
                "[defender] use_sfe requires FastText bindings (install the `[sfe]` extra: `fasttext-ng`). "
                "SFE preprocessor disabled; payload passes through."
            )
            return None

        try:
            model = fasttext.load_model(resolved)
            predictor: SfePredictor = _FastTextPredictor(model)
            _predictor_cache[resolved] = predictor
            return predictor
        except Exception as e:
            _logger.warning("[defender] SFE predictor failed to load (%s); payload will pass through.", e)
            return None


def sfe_preprocess(value: Any, options: dict[str, Any] | None = None) -> SfePreprocessResult:
    """Drop metadata-like leaf fields before Tier 1/Tier 2 classification."""
    if value is None or not isinstance(value, (dict, list)):
        return SfePreprocessResult(filtered=value, dropped=[])

    opts = options or {}
    predictor = opts.get("predictor") or get_default_predictor()
    if predictor is None:
        return SfePreprocessResult(filtered=value, dropped=[])

    threshold = float(opts.get("threshold", 0.5))
    depth_flag = {"hit": False}

    fields = _extract_fields(value, depth_flag)
    candidates = [f for f in fields if f.value_type in {"string", "null"}]
    if not candidates:
        return SfePreprocessResult(filtered=value, dropped=[], truncated_at_depth=depth_flag["hit"] or None)

    decisions = predictor.predict_batch([_field_to_text(f) for f in candidates])
    drop_paths: set[str] = set()
    for i, candidate in enumerate(candidates):
        decision = decisions[i]
        if isinstance(decision, dict):
            label = str(decision.get("label", "pass"))
            prob = float(decision.get("prob", 0.0))
        else:
            label = str(getattr(decision, "label", "pass"))
            prob = float(getattr(decision, "prob", 0.0))
        if label == "drop" and prob >= threshold:
            drop_paths.add(candidate.raw_path)

    if not drop_paths:
        return SfePreprocessResult(filtered=value, dropped=[], truncated_at_depth=depth_flag["hit"] or None)

    dropped = sorted(drop_paths)
    filtered = _compact_dropped(_filter_by_paths(value, drop_paths, depth_flag), depth_flag)
    return SfePreprocessResult(filtered=filtered, dropped=dropped, truncated_at_depth=depth_flag["hit"] or None)


@dataclass
class _Field:
    raw_path: str
    value_type: str
    value_truncated: str
    depth: int


def _value_type(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, int):
        return "int"
    if isinstance(value, float):
        return "float"
    if isinstance(value, str):
        return "string"
    if isinstance(value, list):
        return "array"
    if isinstance(value, dict):
        return "object"
    return "string"


def _extract_fields(
    obj: Any,
    depth_flag: dict[str, bool],
    path: str = "",
    depth: int = 0,
    stack_depth: int = 0,
) -> list[_Field]:
    if stack_depth > MAX_TRAVERSAL_DEPTH:
        depth_flag["hit"] = True
        return []

    out: list[_Field] = []
    if isinstance(obj, dict):
        for key, value in obj.items():
            child = f"{path}.{key}" if path else key
            out.extend(_extract_fields(value, depth_flag, child, depth + 1, stack_depth + 1))
    elif isinstance(obj, list):
        # Keep array element path flattened (same behavior as TS).
        for item in obj:
            out.extend(_extract_fields(item, depth_flag, path, depth, stack_depth + 1))
    else:
        val = "" if obj is None else str(obj)[:500]
        out.append(_Field(raw_path=path, value_type=_value_type(obj), value_truncated=val, depth=depth))
    return out


def _field_to_text(field: _Field) -> str:
    path_tokens = field.raw_path.replace(".", " ").replace("_", " ").replace("-", " ")
    val = field.value_truncated[:200]
    text = f"{field.value_type} d{field.depth} {path_tokens} {val}"
    return text.replace("\r", " ").replace("\n", " ")


def _filter_by_paths(
    obj: Any,
    drop_paths: set[str],
    depth_flag: dict[str, bool],
    path: str = "",
    depth: int = 0,
) -> Any:
    if depth > MAX_TRAVERSAL_DEPTH:
        depth_flag["hit"] = True
        return obj

    if isinstance(obj, list):
        # depth+1 on lists so the cap bounds deep nesting (matches TS); here
        # depth only gates the cap, not path construction.
        return [_filter_by_paths(item, drop_paths, depth_flag, path, depth + 1) for item in obj]

    if isinstance(obj, dict):
        out: dict[str, Any] = {}
        for key, value in obj.items():
            if key in DANGEROUS_KEYS:
                continue
            child = f"{path}.{key}" if path else key
            out[key] = _filter_by_paths(value, drop_paths, depth_flag, child, depth + 1)
        return out

    return _DROPPED if path in drop_paths else obj


def _compact_dropped(obj: Any, depth_flag: dict[str, bool], depth: int = 0) -> Any:
    if depth > MAX_TRAVERSAL_DEPTH:
        depth_flag["hit"] = True
        return obj

    if isinstance(obj, list):
        # depth+1 on lists so the cap bounds deep nesting (matches TS).
        return [_compact_dropped(item, depth_flag, depth + 1) for item in obj if item is not _DROPPED]

    if isinstance(obj, dict):
        out: dict[str, Any] = {}
        for key, value in obj.items():
            if key in DANGEROUS_KEYS:
                continue
            if value is _DROPPED:
                continue
            out[key] = _compact_dropped(value, depth_flag, depth + 1)
        return out

    return obj
