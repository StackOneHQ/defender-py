"""Run the shared conformance fixtures from the TypeScript package (StackOneHQ/defender).

The fixtures in tests/conformance/ are a pinned copy (see VERSION); sync them with
scripts/sync_conformance.sh. Each area maps onto the public API, as described in
tests/conformance/README.md. Outputs are compared in the fixtures' camelCase shape.

Cases this package doesn't match yet are listed in conformance_known_failures.txt and
run as strict xfails: when a port makes one pass, the XPASS fails the run until the
case is removed from the list. So coverage can only grow.
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
import re
from pathlib import Path
from typing import Any

import pytest

from stackone_defender import create_prompt_defense

HERE = Path(__file__).parent
FIXTURES = HERE / "conformance"
KNOWN_FAILURES_FILE = HERE / "conformance_known_failures.txt"

# Fixture option names (TypeScript) → this package's names.
_TIER3_OPTION_NAMES = {
    "maxTextLength": "max_text_length",
    "maxChunks": "max_chunks",
    "onOversize": "on_oversize",
    "escalationBand": "escalation_band",
    "blockThreshold": "block_threshold",
    "allowOverride": "allow_override",
    "onError": "on_error",
    "timeoutMs": "timeout_ms",
    "maxConcurrency": "max_concurrency",
    "stopOnFirstBlock": "stop_on_first_block",
    "includeRaw": "include_raw",
}
_TIER3_RESULT_FIELDS = {
    "decision": "decision",
    "score": "score",
    "refused": "refused",
    "reason": "reason",
    "errored": "errored",
    "overrideApplied": "override_applied",
    "overrideSuppressed": "override_suppressed",
    "decidedBy": "decided_by",
    "skipReason": "skip_reason",
}


def _known_failures() -> dict[str, str]:
    out: dict[str, str] = {}
    for line in KNOWN_FAILURES_FILE.read_text().splitlines():
        entry, _, reason = line.partition("#")
        entry = entry.strip()
        if entry:
            out[entry] = reason.strip() or "not ported yet"
    return out


def _camel(name: str) -> str:
    head, *rest = name.split("_")
    return head + "".join(p[:1].upper() + p[1:] for p in rest)


def _plain(obj: Any) -> Any:
    """Dataclass/dict → dict with camelCase keys and None values dropped."""
    if obj is None:
        return None
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        obj = dataclasses.asdict(obj)
    if isinstance(obj, dict):
        return {_camel(k): _plain(v) for k, v in obj.items() if v is not None}
    if isinstance(obj, list):
        return [_plain(v) for v in obj]
    return obj


def _attr(obj: Any, name: str) -> Any:
    if isinstance(obj, dict):
        return obj.get(name)
    return getattr(obj, name, None)


def _tier3_options(opts: dict[str, Any] | None) -> dict[str, Any]:
    return {_TIER3_OPTION_NAMES.get(k, k): v for k, v in (opts or {}).items()}


class _ScriptedJudge:
    """A provider that answers from the fixture's declarative judge script."""

    def __init__(self, judge: dict[str, Any]):
        self._judge = judge

    def classify(self, text: str, *, ctx: dict[str, Any] | None = None) -> dict[str, Any]:
        verdict = next(
            (r["then"] for r in self._judge.get("rules", []) if r["ifContains"] in text),
            self._judge["default"],
        )
        if "throw" in verdict:
            raise RuntimeError(verdict["throw"])
        return dict(verdict)


class _Recorder:
    def __init__(self) -> None:
        self.calls: list[tuple[int, str]] = []

    def classify(self, text: str, *, ctx: dict[str, Any] | None = None) -> dict[str, Any]:
        chunk = (ctx or {}).get("chunk") or {}
        self.calls.append((chunk.get("index", len(self.calls)), text))
        return {"decision": "allow"}


def _pick_tier3(t3: Any) -> dict[str, Any] | None:
    if t3 is None:
        return None
    out = {}
    for fixture_key, attr in _TIER3_RESULT_FIELDS.items():
        v = _attr(t3, attr)
        if v is not None and v is not False:
            out[fixture_key] = v
    raw = _attr(t3, "raw")
    out["hasRaw"] = raw is not None
    return out


def _run(coro):
    return asyncio.run(coro)


def _serialize(inp: dict[str, Any]) -> dict[str, Any]:
    rec = _Recorder()
    defense = create_prompt_defense(
        block_high_risk=True,
        enable_tier3=True,
        defender_mode="tier3_only",
        tier3={"provider": rec, **_tier3_options(inp.get("tier3"))},
    )
    r = _run(defense.defend_tool_result_async(inp["payload"], inp.get("toolName", "test_tool")))
    return {
        "calls": [text for _, text in sorted(rec.calls, key=lambda c: c[0])],
        "chunkSummary": _plain(getattr(r, "tier3_chunk_summary", None)),
        "coverageDegraded": r.coverage_degraded is True,
        "skipReason": _attr(r.tier3, "skip_reason"),
    }


def _prompt(inp: dict[str, Any]) -> dict[str, Any]:
    try:
        from stackone_defender import DEFAULT_TIER3_PROMPT, build_tier3_messages, create_llm_tier3_provider
    except ImportError as e:  # not ported yet
        raise NotImplementedError("build_tier3_messages is not available") from e
    prompt = inp.get("prompt")
    try:
        if prompt is not None:
            create_llm_tier3_provider(complete=lambda req: "", prompt=prompt)
        messages = build_tier3_messages(prompt or DEFAULT_TIER3_PROMPT, inp["text"], inp.get("ctx") or {})
    except Exception:
        return {"error": True}
    marker = re.compile(r"(<<<(?:END_)?[A-Z_]+ id=)[0-9a-f]{16}(>>>)")
    return {
        "messages": [
            {"role": _attr(m, "role"), "content": marker.sub(r"\1<ID>\2", _attr(m, "content"))} for m in messages
        ]
    }


def _parse(inp: dict[str, Any]) -> dict[str, Any]:
    try:
        from stackone_defender import parse_verdict_json
    except ImportError as e:  # not ported yet
        raise NotImplementedError("parse_verdict_json is not available") from e
    try:
        return _plain(parse_verdict_json(inp["reply"]))
    except Exception:
        return {"error": True}


def _redact(inp: dict[str, Any]) -> dict[str, Any]:
    defense = create_prompt_defense(
        enable_tier3=True,
        defender_mode="tier3_only",
        tier3={"provider": _ScriptedJudge({"default": {"throw": inp["message"]}})},
    )
    r = _run(defense.defend_tool_result_async({"body": "x"}, "test_tool"))
    return {"skipReason": _attr(r.tier3, "skip_reason")}


def _policy(inp: dict[str, Any]) -> dict[str, Any]:
    tier3 = _tier3_options(inp.get("tier3"))
    if inp.get("judge"):
        tier3["provider"] = _ScriptedJudge(inp["judge"])
    if inp.get("decide"):
        mode = inp["decide"]

        def decide(_input: Any) -> str:
            if mode == "throw":
                raise RuntimeError("policy bug")
            return mode

        tier3["decide"] = decide
    tier2 = inp.get("tier2") or {"enabled": False}
    tier2_config = {
        k: tier2[c] for c, k in (("highRiskThreshold", "high_risk_threshold"), ("mediumRiskThreshold", "medium_risk_threshold"))
        if c in tier2
    }
    defense = create_prompt_defense(
        block_high_risk=inp.get("blockHighRisk", True),
        enable_tier1=inp.get("enableTier1", False),
        enable_tier2=tier2["enabled"],
        tier2_config=tier2_config or None,
        enable_tier3=True,
        defender_mode=inp["mode"],
        tier3=tier3,
    )
    r = _run(defense.defend_tool_result_async(inp["payload"], inp.get("toolName", "test_tool")))
    out = {
        "allowed": r.allowed,
        "riskLevel": r.risk_level,
        "tier3": _pick_tier3(r.tier3),
        "coverageDegraded": r.coverage_degraded is True,
    }
    summary = _plain(getattr(r, "tier3_chunk_summary", None))
    if summary:
        out["chunkSummary"] = summary
    return out


ADAPTERS = {
    "tier3-serialize": _serialize,
    "tier3-prompt": _prompt,
    "tier3-parse": _parse,
    "tier3-redact": _redact,
    "tier3-policy": _policy,
}


def _cases():
    known = _known_failures()
    seen: set[str] = set()
    for path in sorted(FIXTURES.glob("*.json")):
        fixture = json.loads(path.read_text())
        area = fixture["area"]
        for case in fixture["cases"]:
            key = f"{area}/{case['id']}"
            seen.add(key)
            marks = []
            if case.get("requiresTier2"):
                marks.append(pytest.mark.model)
            if key in known:
                marks.append(pytest.mark.xfail(strict=True, reason=known[key]))
            yield pytest.param(area, case, id=key, marks=marks)
    stale = sorted(set(known) - seen)
    if stale:
        raise RuntimeError(f"conformance_known_failures.txt lists cases that no longer exist: {stale}")


def test_fixtures_are_pinned():
    version = (FIXTURES / "VERSION").read_text().strip()
    assert version, "tests/conformance/VERSION must name the TS ref the fixtures came from"
    areas = {json.loads(p.read_text())["area"] for p in FIXTURES.glob("*.json")}
    assert areas <= set(ADAPTERS), f"fixture areas without an adapter: {areas - set(ADAPTERS)}"


@pytest.mark.parametrize(("area", "case"), list(_cases()))
def test_conformance(area: str, case: dict[str, Any]):
    actual = ADAPTERS[area](case["input"])
    assert actual == case["expect"]
