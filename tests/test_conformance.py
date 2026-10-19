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

import stackone_defender
from stackone_defender import create_prompt_defense

HERE = Path(__file__).parent
FIXTURES = HERE / "conformance"
KNOWN_FAILURES_FILE = HERE / "conformance_known_failures.txt"
FIXTURE_FORMAT_VERSION = 1
# A full commit SHA or a release tag. Branches would float; anything else isn't a TS ref.
PIN_RE = re.compile(r"^([0-9a-f]{40}|defender-v\d+\.\d+\.\d+)$")
MARKER_RE = re.compile(r"(<<<(?:END_)?[A-Z_]+ id=)([0-9a-f]{16})(>>>)")

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
_NESTED_OPTION_NAMES = {"maxTier2Score": "max_tier2_score"}
_CTX_NAMES = {"toolName": "tool_name", "userGoal": "user_goal", "toolDescription": "tool_description"}
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
    "note": "note",
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
    out = {}
    for k, v in (opts or {}).items():
        if isinstance(v, dict):
            v = {_NESTED_OPTION_NAMES.get(nk, nk): nv for nk, nv in v.items()}
        out[_TIER3_OPTION_NAMES.get(k, k)] = v
    return out


def _require(name: str) -> Any:
    """The port's public function, or NotImplementedError when it doesn't exist yet."""
    fn = getattr(stackone_defender, name, None)
    if fn is None:
        raise NotImplementedError(f"stackone_defender.{name} is not available yet")
    return fn


def _tier2_unavailable() -> str | None:
    try:
        import onnxruntime  # noqa: F401
        import tokenizers  # noqa: F401
    except ImportError:
        return "onnxruntime/tokenizers not installed (install the [onnx] extra)"
    model = Path(stackone_defender.get_default_model_path()) / "model_quantized.onnx"
    if not model.exists():
        return f"bundled Tier-2 model missing at {model}"
    return None


class _ScriptedJudge:
    """A provider that answers from the fixture's declarative judge script."""

    def __init__(self, judge: dict[str, Any]):
        self._judge = judge
        self.calls = 0

    def classify(self, text: str, *, ctx: dict[str, Any] | None = None) -> dict[str, Any]:
        self.calls += 1
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

    def classify(self, text: str, *, ctx: Any = None) -> dict[str, Any]:
        chunk = _attr(ctx, "chunk")
        index = _attr(chunk, "index") if chunk is not None else None
        self.calls.append((index if index is not None else len(self.calls), text))
        return {"decision": "allow"}


def _pick_tier3(t3: Any) -> dict[str, Any] | None:
    if t3 is None:
        return None
    out = {}
    for fixture_key, attr in _TIER3_RESULT_FIELDS.items():
        v = _attr(t3, attr)
        # Absent and None mean "not set" (as `undefined` in TS); False is a real value.
        if v is not None:
            out[fixture_key] = v
    raw = _attr(t3, "raw")
    out["hasRaw"] = raw is not None
    return out


def _normalize_ids(contents: list[str]) -> list[str]:
    """Real marker ids → <ID1>, <ID2>… by first appearance, so open/close pairing and
    per-block distinctness stay pinned."""
    ids: dict[str, str] = {}

    def sub(m: re.Match[str]) -> str:
        ids.setdefault(m[2], f"<ID{len(ids) + 1}>")
        return f"{m[1]}{ids[m[2]]}{m[3]}"

    return [MARKER_RE.sub(sub, c) for c in contents]


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
    build_tier3_messages = _require("build_tier3_messages")
    create_llm_tier3_provider = _require("create_llm_tier3_provider")
    default_prompt = _require("DEFAULT_TIER3_PROMPT")
    prompt = inp.get("prompt")
    ctx = {_CTX_NAMES.get(k, k): v for k, v in (inp.get("ctx") or {}).items()}
    # Contract: a rejected template raises ValueError. Anything else is a real failure, so a
    # broken port can't turn the "rejected" cases green by crashing.
    try:
        if prompt is not None:
            create_llm_tier3_provider(complete=lambda req: "", prompt=prompt)
        messages = build_tier3_messages(prompt or default_prompt, inp["text"], ctx)
    except ValueError:
        return {"error": True}
    contents = _normalize_ids([_attr(m, "content") for m in messages])
    return {"messages": [{"role": _attr(m, "role"), "content": c} for m, c in zip(messages, contents, strict=True)]}


def _parse(inp: dict[str, Any]) -> dict[str, Any]:
    parse_verdict_json = _require("parse_verdict_json")
    # Contract: an unusable reply raises ValueError; any other exception is a real failure.
    try:
        return _plain(parse_verdict_json(inp["reply"]))
    except ValueError:
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
    judge = _ScriptedJudge(inp["judge"]) if inp.get("judge") else None
    if judge:
        tier3["provider"] = judge
    if inp.get("decide"):
        mode = inp["decide"]

        def decide(_input: Any) -> str:
            if mode == "throw":
                raise RuntimeError("policy bug")
            return mode

        tier3["decide"] = decide
    tier2 = inp.get("tier2") or {"enabled": False}
    tier2_config = {
        k: tier2[c]
        for c, k in (("highRiskThreshold", "high_risk_threshold"), ("mediumRiskThreshold", "medium_risk_threshold"))
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
    if judge:
        out["judgeCalls"] = judge.calls
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
    tier2_missing = _tier2_unavailable()
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
                if tier2_missing:
                    marks.append(pytest.mark.skip(reason=f"needs Tier 2: {tier2_missing}"))
            if key in known:
                marks.append(pytest.mark.xfail(strict=True, reason=known[key]))
            yield pytest.param(area, case, id=key, marks=marks)
    stale = sorted(set(known) - seen)
    if stale:
        raise RuntimeError(f"conformance_known_failures.txt lists cases that no longer exist: {stale}")


def test_fixtures_are_pinned():
    version = (FIXTURES / "VERSION").read_text().strip()
    assert PIN_RE.match(version), (
        f"tests/conformance/VERSION must be a full TS commit SHA or a defender-v* tag, got {version!r}"
    )
    fixtures = [json.loads(p.read_text()) for p in FIXTURES.glob("*.json")]
    assert fixtures, "no fixtures in tests/conformance/"
    for fx in fixtures:
        assert fx["version"] == FIXTURE_FORMAT_VERSION, (
            f"{fx['area']}: fixture format {fx['version']} isn't supported by this runner"
        )
    areas = {fx["area"] for fx in fixtures}
    assert areas <= set(ADAPTERS), f"fixture areas without an adapter: {areas - set(ADAPTERS)}"


# Not expressible as a fixture: marker ids are random per call. Strict xfail until the
# prompt builder is ported, so it starts running (and must pass) once it exists.
@pytest.mark.xfail(
    not hasattr(stackone_defender, "build_tier3_messages"),
    reason="ENG-3500 phase 2 (ENG-3495 BYO-LLM): prompt + parser",
    raises=NotImplementedError,
    strict=True,
)
def test_marker_ids_random_per_call_and_distinct_per_block():
    build_tier3_messages = _require("build_tier3_messages")
    default_prompt = _require("DEFAULT_TIER3_PROMPT")

    def ids() -> list[str]:
        messages = build_tier3_messages(default_prompt, "c", {"user_goal": "g", "tool_description": "d"})
        return [m[2] for m in MARKER_RE.finditer("\n".join(_attr(m, "content") for m in messages))]

    a, b = ids(), ids()
    assert len(a) == 6  # 3 blocks × open/close
    assert len(set(a)) == 3
    assert not set(a) & set(b)


@pytest.mark.parametrize(("area", "case"), list(_cases()))
def test_conformance(area: str, case: dict[str, Any]):
    actual = ADAPTERS[area](case["input"])
    assert actual == case["expect"]
