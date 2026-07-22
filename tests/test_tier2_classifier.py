"""Tests for Tier 2 classifier configuration and behavior."""

import json
from pathlib import Path

from stackone_defender.classifiers import tier2_classifier as t2_mod
from stackone_defender.classifiers.tier2_classifier import Tier2Classifier, create_tier2_classifier
from stackone_defender.types import MultiheadConfig


class TestTier2ClassifierConfig:
    def test_default_config(self):
        c = Tier2Classifier()
        assert c.get_risk_level(0.9) == "high"
        assert c.get_risk_level(0.6) == "medium"
        assert c.get_risk_level(0.3) == "low"

    def test_custom_thresholds(self):
        c = Tier2Classifier(config={"high_risk_threshold": 0.7, "medium_risk_threshold": 0.4})
        assert c.get_risk_level(0.7) == "high"
        assert c.get_risk_level(0.5) == "medium"
        assert c.get_risk_level(0.3) == "low"

    def test_skip_short_text(self):
        c = Tier2Classifier()
        result = c.classify("hi")
        assert result.skipped
        assert "too short" in (result.skip_reason or "")

    def test_not_ready_without_model(self):
        c = Tier2Classifier(config={"onnx_model_path": "/nonexistent/path"})
        assert not c.is_ready()

    def test_create_factory(self):
        c = create_tier2_classifier()
        assert c.get_risk_level(0.9) == "high"

    def test_get_config(self):
        c = Tier2Classifier()
        cfg = c.get_config()
        # The bundled minilm-multihead-v5 model ships calibrated defaults via
        # classifier_config.json (T=2.41, high_risk_threshold=0.64). Library
        # defaults still apply for fields the model doesn't override.
        assert cfg["high_risk_threshold"] == 0.64
        assert cfg["medium_risk_threshold"] == 0.5
        assert cfg["min_text_length"] == 10
        assert cfg["max_text_length"] == 10000
        assert cfg["temperature_t"] == 2.41

    def test_get_config_custom(self):
        c = Tier2Classifier(config={"high_risk_threshold": 0.9})
        cfg = c.get_config()
        assert cfg["high_risk_threshold"] == 0.9

    def test_prepare_chunks_short_text_skips(self):
        c = Tier2Classifier()
        prep = c.prepare_chunks("hi")
        assert prep["skipped"]
        assert prep["chunks"] == []

    def test_classify_chunks_batch_passthrough(self):
        c = Tier2Classifier()

        class _FakeOnnx:
            def warmup(self):
                return None

            def classify_batch(self, chunks):
                return [0.1] * len(chunks)

        c._onnx = _FakeOnnx()  # type: ignore[attr-defined]
        assert c.classify_chunks_batch(["a", "b"]) == [0.1, 0.1]


# ---------------------------------------------------------------------------
# 0.7.0 parity: calibration auto-load + three-tier merge + None-filter regression
# ---------------------------------------------------------------------------


class TestCalibrationAutoLoad:
    def test_temperature_from_model_calibration(self):
        # Bundled model ships T=2.41 in classifier_config.json:calibration.
        c = Tier2Classifier()
        assert c.get_temperature() == 2.41

    def test_thresholds_from_model_calibration(self):
        c = Tier2Classifier()
        cfg = c.get_config()
        assert cfg["high_risk_threshold"] == 0.64

    def test_caller_overrides_model(self, tmp_path: Path):
        # Caller-supplied config wins over model calibration.
        c = Tier2Classifier(config={"high_risk_threshold": 0.91, "temperature_t": 3.0})
        cfg = c.get_config()
        assert cfg["high_risk_threshold"] == 0.91
        assert cfg["temperature_t"] == 3.0

    def test_none_caller_keys_do_not_clobber_model(self):
        # A naive merge would let ``temperature_t=None`` clobber the model's
        # T=2.41 and drop calibration. The implementation filters None values
        # from caller config so the model default stays.
        c = Tier2Classifier(config={"temperature_t": None})
        assert c.get_temperature() == 2.41

    def test_calibration_cache_is_memoized(self, tmp_path: Path):
        # Two classifiers pointing at the same model dir share one cache hit.
        model_dir = tmp_path / "mock-model"
        model_dir.mkdir()
        (model_dir / "classifier_config.json").write_text(
            json.dumps({"calibration": {"temperatureT": 1.5, "highRiskThreshold": 0.7}})
        )
        # Invoke twice; the second call goes through the cache.
        first = t2_mod._read_calibration_defaults(str(model_dir))
        second = t2_mod._read_calibration_defaults(str(model_dir))
        assert first is second
        assert first.temperature_t == 1.5
        assert first.high_risk_threshold == 0.7

    def test_calibration_cache_remembers_none(self, tmp_path: Path):
        # ``None`` is a valid cached value ("no calibration block"). The
        # second probe must hit the cache, not the filesystem.
        model_dir = tmp_path / "nocalib"
        model_dir.mkdir()
        (model_dir / "classifier_config.json").write_text(json.dumps({"other": 1}))
        first = t2_mod._read_calibration_defaults(str(model_dir))
        second = t2_mod._read_calibration_defaults(str(model_dir))
        assert first is None and second is None


class TestMultiheadIntrospection:
    def test_is_multihead_default_false(self):
        c = Tier2Classifier()
        assert not c.is_multihead()
        assert c.get_multihead_config() is None

    def test_is_multihead_when_configured(self):
        mh = MultiheadConfig(main_threshold=0.5, aux_threshold=0.64)
        c = Tier2Classifier(config={"multihead": mh})
        assert c.is_multihead()
        got = c.get_multihead_config()
        assert got is not None
        assert got.main_threshold == 0.5
        assert got.aux_threshold == 0.64


class TestClassifyChunksBatchPair:
    def test_pair_passthrough(self):
        c = Tier2Classifier()

        class _FakeOnnx:
            def warmup(self):
                return None

            def classify_batch_pair(self, chunks):
                return [(0.3, 0.7)] * len(chunks)

        c._onnx = _FakeOnnx()  # type: ignore[attr-defined]
        pairs = c.classify_chunks_batch_pair(["a", "b"])
        assert pairs == [(0.3, 0.7), (0.3, 0.7)]
