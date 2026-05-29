"""ONNX classifier for fine-tuned MiniLM prompt injection detection.

Pipeline: text -> tokenizer -> ONNX Runtime -> logit -> ``sigmoid(logit / T)``
-> score. Supports single-head ``[batch]`` / ``[batch, 1]`` models and
multi-head ``[batch, 2]`` models (main + aux). Temperature ``T`` enables
post-hoc calibration via temperature scaling.
"""

from __future__ import annotations

import logging
import math
import threading
from pathlib import Path
from typing import Literal

_logger = logging.getLogger(__name__)

# Shared across all OnnxClassifier instances (keyed by resolved model dir path).
_session_cache: dict[str, tuple[object, object]] = {}
_registry_lock = threading.Lock()
_load_locks: dict[str, threading.Lock] = {}


def _lock_for_cache_key(cache_key: str) -> threading.Lock:
    with _registry_lock:
        if cache_key not in _load_locks:
            _load_locks[cache_key] = threading.Lock()
        return _load_locks[cache_key]


def get_default_model_path() -> str:
    """Return the absolute path to the bundled ONNX model directory.

    Exported so :class:`Tier2Classifier` can read model-specific calibration
    defaults from ``classifier_config.json`` at construction time without
    needing an :class:`OnnxClassifier` instance.
    """
    return str(Path(__file__).resolve().parent.parent / "models" / "minilm-multihead-v5")


# Back-compat shim retained for internal users; same value as the public name.
def _default_model_path() -> str:
    return get_default_model_path()


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


class OnnxClassifier:
    """ONNX Classifier for fine-tuned MiniLM models.

    Loads the model lazily on first inference. The session and tokenizer
    are cached at module level so multiple instances pointing at the same
    model path share a single backing session (safe: ONNX Runtime
    guarantees thread-safe ``Run()`` from v1.7.0, and the ``tokenizers``
    library's encode methods do not mutate the tokenizer object).
    """

    _MAX_BATCH_CHUNK = 32

    def __init__(self, model_path: str | None = None, temperature_t: float | None = None):
        self._model_path = model_path or get_default_model_path()
        self._session = None
        self._tokenizer = None
        self._max_length = 256
        self._load_failed = False
        # Output mode is detected lazily from the logits shape on the first
        # inference call. ``None`` until then.
        self._output_mode: Literal["single", "multi"] | None = None
        # Temperature ``T`` must be a positive finite number. ``T <= 0`` is
        # undefined (divide-by-zero or sign flip) and almost certainly a
        # programming error rather than a config the caller wants gracefully
        # ignored.
        self._temperature_t = 1.0
        if temperature_t is not None:
            if not math.isfinite(temperature_t) or temperature_t <= 0:
                raise ValueError(
                    f"OnnxClassifier: temperature_t must be a positive finite number, got {temperature_t}"
                )
            self._temperature_t = float(temperature_t)

    # ------------------------------------------------------------------
    # Public introspection
    # ------------------------------------------------------------------

    def get_temperature(self) -> float:
        """Current temperature scaling factor (``1.0`` = no calibration)."""
        return self._temperature_t

    def get_output_mode(self) -> Literal["single", "multi"] | None:
        """Output mode of the loaded model.

        ``None`` until the first inference runs. ``"multi"`` indicates the
        model emits ``[batch, 2]`` logits (main + aux).
        """
        return self._output_mode

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def load_model(self, model_path: str | None = None) -> None:
        if model_path:
            self._model_path = model_path
        if self._session is not None and self._tokenizer is not None:
            return
        if self._load_failed:
            raise ImportError("ONNX dependencies not installed. Install with: pip install stackone-defender[onnx]")
        self._load_model()

    def _load_model(self) -> None:
        cache_key = str(Path(self._model_path).resolve())
        cached = _session_cache.get(cache_key)
        if cached:
            self._session, self._tokenizer = cached
            return

        with _lock_for_cache_key(cache_key):
            cached = _session_cache.get(cache_key)
            if cached:
                self._session, self._tokenizer = cached
                return

            try:
                import numpy as np  # noqa: F401
                import onnxruntime as ort
                from tokenizers import Tokenizer
            except ImportError as e:
                self._load_failed = True
                _logger.warning("[defender] ONNX model failed to load: %s", e)
                raise ImportError(
                    "ONNX dependencies not installed. Install with: pip install stackone-defender[onnx]"
                ) from e

            try:
                tokenizer_path = str(Path(self._model_path) / "tokenizer.json")
                self._tokenizer = Tokenizer.from_file(tokenizer_path)
                self._tokenizer.enable_truncation(max_length=self._max_length)
                self._tokenizer.enable_padding(length=self._max_length)

                onnx_path = str(Path(self._model_path) / "model_quantized.onnx")
                self._session = ort.InferenceSession(onnx_path)
            except Exception as e:
                _logger.warning("[defender] ONNX model failed to load: %s", e)
                raise

            _session_cache[cache_key] = (self._session, self._tokenizer)

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def classify(self, text: str) -> float:
        """Classify a single text, returning the main-head sigmoid score.

        For multi-head models only the main score is returned; callers that
        need the aux score should use :meth:`classify_pair`.
        """
        return self.classify_pair(text)[0]

    def classify_pair(self, text: str) -> tuple[float, float | None]:
        """Classify a single text, returning ``(main, aux)``.

        ``aux`` is ``None`` for single-head models. Both scores are
        sigmoid-activated with the configured temperature ``T``.
        """
        self._ensure_loaded()
        import numpy as np

        encoding = self._tokenizer.encode(text)
        input_ids = np.array([encoding.ids], dtype=np.int64)
        attention_mask = np.array([encoding.attention_mask], dtype=np.int64)

        results = self._session.run(None, {"input_ids": input_ids, "attention_mask": attention_mask})
        logits = results[0]
        self._detect_output_mode(logits.shape)

        t = self._temperature_t
        row = logits[0]
        # row shape: (), (1,) or (2,) depending on model export.
        if self._output_mode == "multi":
            main = _sigmoid(float(row[0]) / t)
            aux = _sigmoid(float(row[1]) / t)
            return main, aux
        main_logit = float(row[0]) if hasattr(row, "__len__") and len(row) > 0 else float(row)
        return _sigmoid(main_logit / t), None

    def classify_batch(self, texts: list[str]) -> list[float]:
        """Classify multiple texts; returns main-head scores only.

        Back-compat wrapper around :meth:`classify_batch_pair`.
        """
        return [main for main, _ in self.classify_batch_pair(texts)]

    def classify_batch_pair(self, texts: list[str]) -> list[tuple[float, float | None]]:
        """Classify multiple texts, returning ``(main, aux)`` per row.

        Aux is ``None`` per-row for single-head models. Chunks the input to
        bound native memory; the attention matrix is ``O(chunk * seq_len^2)``,
        and for MiniLM (``max_length=256``) a chunk of 32 keeps memory
        under ~50MB per call.
        """
        if not texts:
            return []
        self._ensure_loaded()
        all_pairs: list[tuple[float, float | None]] = []
        for offset in range(0, len(texts), self._MAX_BATCH_CHUNK):
            chunk = texts[offset : offset + self._MAX_BATCH_CHUNK]
            all_pairs.extend(self._classify_batch_chunk_pair(chunk))
        return all_pairs

    def _classify_batch_chunk_pair(self, texts: list[str]) -> list[tuple[float, float | None]]:
        import numpy as np

        encodings = self._tokenizer.encode_batch(texts)
        input_ids = np.array([e.ids for e in encodings], dtype=np.int64)
        attention_mask = np.array([e.attention_mask for e in encodings], dtype=np.int64)

        results = self._session.run(None, {"input_ids": input_ids, "attention_mask": attention_mask})
        logits = results[0]
        self._detect_output_mode(logits.shape)

        t = self._temperature_t
        pairs: list[tuple[float, float | None]] = []
        if self._output_mode == "multi":
            for i in range(len(texts)):
                main = _sigmoid(float(logits[i][0]) / t)
                aux = _sigmoid(float(logits[i][1]) / t)
                pairs.append((main, aux))
        else:
            for i in range(len(texts)):
                row = logits[i]
                # ``row`` may be a scalar (shape ``[batch]``) or 1-vector.
                main_logit = float(row[0]) if hasattr(row, "__len__") and len(row) > 0 else float(row)
                pairs.append((_sigmoid(main_logit / t), None))
        return pairs

    def _detect_output_mode(self, dims) -> None:
        """Detect output mode from the logits tensor shape on first inference.

        - ``[batch]`` or ``[batch, 1]`` -> ``"single"``
        - ``[batch, 2]`` -> ``"multi"`` (main + aux dual head)

        Idempotent: subsequent calls are no-ops once mode is set.
        """
        if self._output_mode is not None:
            return
        if dims is None or len(dims) < 2:
            self._output_mode = "single"
            return
        self._output_mode = "multi" if dims[1] == 2 else "single"

    # ------------------------------------------------------------------
    # Misc
    # ------------------------------------------------------------------

    def count_tokens(self, text: str) -> int:
        self._ensure_loaded()
        encoding = self._tokenizer.encode(text)
        # Padding is enabled at a fixed length; count only real (attended) tokens.
        return int(sum(encoding.attention_mask))

    def get_max_length(self) -> int:
        return self._max_length

    def warmup(self) -> None:
        self.load_model()

    def is_loaded(self) -> bool:
        return self._session is not None and self._tokenizer is not None

    def _ensure_loaded(self) -> None:
        if not self.is_loaded():
            self.load_model()
