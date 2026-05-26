"""Tests for SFE preprocessing and PromptDefense integration."""

from stackone_defender import create_prompt_defense
from stackone_defender.sfe import preprocess as sfe_mod
from stackone_defender.sfe.preprocess import SfePreprocessResult, sfe_preprocess


class _MockPredictor:
    def predict(self, text: str):
        return {"label": "drop", "prob": 0.99} if "uuid" in text else {"label": "pass", "prob": 0.99}

    def predict_batch(self, texts: list[str]):
        return [self.predict(text) for text in texts]


def test_sfe_preprocess_passes_through_primitive():
    result = sfe_preprocess("hello")
    assert isinstance(result, SfePreprocessResult)
    assert result.filtered == "hello"
    assert result.dropped == []


def test_sfe_preprocess_drops_metadata_like_fields():
    payload = {"uuid": "abc-123", "description": "Hello world"}
    result = sfe_preprocess(payload, {"predictor": _MockPredictor(), "threshold": 0.5})
    assert "uuid" in result.dropped
    assert "uuid" not in result.filtered
    assert result.filtered["description"] == "Hello world"


def test_sfe_preprocess_fail_open_when_predictor_unavailable(monkeypatch):
    payload = {"uuid": "abc-123", "description": "Hello world"}
    monkeypatch.setattr(sfe_mod, "get_default_predictor", lambda model_path=None: None)
    result = sfe_preprocess(payload)
    assert result.filtered == payload
    assert result.dropped == []


def test_prompt_defense_use_sfe_reports_fields_dropped_but_keeps_tier1_payload():
    defense = create_prompt_defense(enable_tier1=False, enable_tier2=False, use_sfe={"predictor": _MockPredictor()})
    result = defense.defend_tool_result({"uuid": "abc-123", "description": "Hello"}, "test_tool")
    assert "uuid" in result.fields_dropped
    assert result.sanitized["uuid"] == "abc-123"
    assert result.sanitized["description"] == "Hello"
