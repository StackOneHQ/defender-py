# API reference

For the local quick start, see the [README](../README.md). See [optional LLM review](llm-review.md) for provider setup and policies.

### `create_prompt_defense(**kwargs)`

```python
defense = create_prompt_defense(
    enable_tier1=True,
    enable_tier2=True,
    require_tier2=False,  # True: raise if Tier 2 can't load (fail closed) instead of degrading to Tier 1
    block_high_risk=False,
    default_risk_level="low",
    annotate_boundary=False,  # True: wrap risky strings with [UD-…] tags (npm: annotateBoundary)
    sanitize_content=True,  # False: disable sentence cleaning; boundary annotation is separate
    tier2_fields=["subject", "body", "snippet"],  # optional: scope Tier 2 to these JSON keys (default: all strings)
    tier2_config={"high_risk_threshold": 0.52},  # optional; defaults come from the bundled model's classifier_config.json
    use_sfe=False,  # True, or {"threshold": 0.5, "predictor": ...}: SFE preprocessing before Tier 2
    enable_tier3=False,
    defender_mode="cascade",  # or "tier3_only"; ignored unless enable_tier3
    tier3=None,  # dict of Tier 3 options (see the optional LLM review reference)
)
```

Set Tier 2 thresholds with `tier2_config`. The bundled model's calibrated values override library defaults, so threshold keys under `config["tier2"]` are ignored.

### `defense.defend_tool_result(value, tool_name, *, context=None)`

Runs Tier 1 detection on the original payload, then Tier 2 on its strings (the SFE-filtered view when `use_sfe` is on). `tier2_fields` limits Tier 2 to those keys; omit it to classify all strings. The call is synchronous. With `enable_tier3=True` it runs Tier 3 in its own event loop and raises `RuntimeError` inside a running loop: use `await defense.defend_tool_result_async(...)` there. `context` (`user_goal`, `tool_description`) goes to the Tier 3 provider only.

```python
from dataclasses import dataclass, field

@dataclass
class DefenseResult:
    allowed: bool                             # gating decision (respects block_high_risk)
    risk_level: RiskLevel                     # diagnostic; combines Tier 1 / Tier 2 and optional Tier 3 policy
    sanitized: Any                            # sentence-cleaned copy (cleaning disabled when sanitize_content=False); dropped runs leave a [CONTENT SANITISED] marker; best-effort, still gate on allowed
    detections: list[str]                     # Tier 1 pattern names detected
    fields_sanitized: list[str]               # fields whose content the cleaner changed in sanitized (empty when sanitize_content=False or no Tier 2); for detections read detections/patterns_by_field
    patterns_by_field: dict[str, list[str]]   # patterns detected per field
    detected_field_count: int                 # count of fields with a Tier-1 detection (keys of patterns_by_field); threat-count signal (fields_sanitized len no longer tracks this)
    tier2_score: float | None = None
    tier2_raw_score: float | None = None
    tier2_aux_score: float | None = None      # set only when tier2_config["multihead"] is set
    tier2_multihead_blocked: bool | None = None
    tier2_skip_reason: str | None = None
    max_sentence: str | None = None           # selected Tier 2 chunk; may contain multiple sentences
    # Metadata from the Tier 3 path or an eligible cascade escalation.
    # With no provider and non-blocking on_error, tier3_only falls back to T1 + T2 and leaves this None.
    tier3: Tier3Result | None = None          # verdict or skip metadata when present
    fields_dropped: list[str] = field(default_factory=list)
    sfe_fallback: bool | None = None          # True when SFE kept no scorable text, so Tier 2 scored the original (URLs/IDs excluded)
    truncated_at_depth: bool | None = None
    latency_ms: float = 0.0
    # Cost telemetry — present only when the batched Tier 2 classifier ran
    phase_timings: PhaseTimings | None = None       # prepare / infer / aggregate ms
    tier2_stats: Tier2Stats | None = None           # string/chunk/unique counts, real/padded tokens
    tier1_ms: float | None = None
    cold_load: bool | None = None
    # Operational signals
    tier2_available: bool | None = None       # False when Tier 2 enabled but failed to load
    coverage_degraded: bool | None = None      # True when Tier 1 detection or the Tier 3 review didn't cover everything
    # An unreadable payload adds no risk of its own: risk_level reflects only what was scanned (e.g. a Tier 1
    # detection). Strict mode still blocks it (allowed=False). In tier3_only, tier3.errored is set.
    tier3_chunk_summary: Tier3ChunkSummary | None = None  # tier3_only: chunks, blocked, blocking_chunk, oversize
```

### `defense.defend_tool_results(items)`

Sync batch API. Each item is `{"value": ..., "tool_name": ..., "context": {...}}` (`context` optional). Without Tier 3 it runs items one by one. With `enable_tier3=True` it runs them concurrently in one `asyncio.run()` (sync providers still run one at a time) and raises `RuntimeError` inside a running event loop. From async code, use `defend_tool_results_async`.

Set `block_high_risk=True` to enable blocking. Both batch APIs return verdicts; your application must check each one before forwarding its `sanitized` value.

```python
results = defense.defend_tool_results([
    {"value": email_data, "tool_name": "gmail_get_message"},
    {"value": doc_data, "tool_name": "documents_get"},
    {"value": pr_data, "tool_name": "github_get_pull_request"},
])
for result in results:
    if not result.allowed:
        print("Blocked:", ", ".join(result.detections))
        continue

    # Forward this value to your agent in place of the original tool output.
    print(result.sanitized)
```

### `await defense.defend_tool_results_async(items)`

Async batch API: runs `defend_tool_result_async` per item concurrently via `asyncio.gather`. Required when Tier 3 is enabled inside a running event loop (e.g. FastAPI).

```python
results = await defense.defend_tool_results_async([
    {"value": email_data, "tool_name": "gmail_get_message"},
    {"value": doc_data, "tool_name": "documents_get"},
])

for result in results:
    if not result.allowed:
        print("Blocked:", ", ".join(result.detections))
        continue

    # Forward this value to your agent in place of the original tool output.
    print(result.sanitized)
```

### `defense.analyze(text)`

Tier 1 only: useful for debugging pattern hits without full tool-result traversal.

### Tier 2 warmup

```python
defense = create_prompt_defense()
defense.warmup_tier2()  # no-op if enable_tier2=False; warns (or raises with require_tier2=True) if the [onnx] extra is missing
```

### Parity with the TypeScript package

This package mirrors [`@stackone/defender`](https://github.com/StackOneHQ/defender). `tests/conformance/` is a pinned copy of its shared conformance fixtures: input → expected-output cases for Tier-3 serialization, prompt rendering, verdict parsing, error redaction and policy decisions. `tests/test_conformance.py` runs them against this package.

- `scripts/sync_conformance.sh <ref>` copies the fixtures at a TS tag or commit and records it in `tests/conformance/VERSION`. CI checks the copy matches (`--check`).
- Every case passes. A new TS case this package can't match yet goes in `tests/conformance_known_failures.txt`, where it runs as a strict xfail; remove its line once a port makes it pass.
- Port contract for the runner: a rejected prompt template and an unusable verdict reply raise `ValueError`; optional result fields are `None` when unset (not `False`).

Traps when porting from TypeScript (the fixtures pin TS behavior):

- Lengths and slicing are in UTF-16 code units in TS, but `len()` counts code points.
- TS `Object.entries` puts integer-like keys first. `json.loads` keeps insertion order.
- Number rendering differs: `String(1.0)` is `"1"` but `str(1.0)` is `"1.0"`. Likewise `-0`, `1e-7` and `1e16` render differently in the two languages.
- `core/tier3_input.py` handles the three traps above for Tier-3 input. It works on a UTF-16 view of each string, orders keys like `Object.entries`, and renders numbers like `String(n)`. One deliberate difference: Python ints below 1e21 print exactly, where JS would round past 2**53.
- Python `re` has no `\p{Cf}`, and `unicodedata` versions differ across Python releases.
- `str.splitlines()` also splits on `\x0b`, `\x0c` and `\x1c`, but TS splits only on `\r`, `\n`, `\u2028`, `\u2029` and `\u0085`.
- JS `trim()` removes a BOM, but `str.strip()` doesn't.
- `json.loads` accepts `NaN`, but JS `JSON.parse` rejects it. Use `parse_constant` to reject it.
