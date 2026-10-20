<div align="center">

  <a href="https://www.stackone.com/platform/prompt-injection-guard/?utm_source=pypi&utm_medium=readme&utm_campaign=banner">
    <picture>
      <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/StackOneHQ/defender/main/assets/banner-dark.svg" />
      <img src="https://raw.githubusercontent.com/StackOneHQ/defender/main/assets/banner-light.svg" alt="Defender by StackOne — Indirect prompt injection protection for MCP tool calls" width="800" />
    </picture>
  </a>

  <p>
    <a href="https://pypi.org/project/stackone-defender/"><img src="https://img.shields.io/pypi/v/stackone-defender?style=flat-square&color=047B43&label=pypi" alt="PyPI version" /></a>
    <a href="https://github.com/StackOneHQ/defender-py/releases"><img src="https://img.shields.io/github/v/release/StackOneHQ/defender-py?style=flat-square&color=047B43&label=release" alt="latest GitHub release" /></a>
    <a href="https://github.com/StackOneHQ/defender-py/stargazers"><img src="https://img.shields.io/github/stars/StackOneHQ/defender-py?style=flat-square&color=047B43" alt="GitHub stars" /></a>
    <a href="./LICENSE"><img src="https://img.shields.io/pypi/l/stackone-defender?style=flat-square&color=047B43" alt="License" /></a>
    <img src="https://img.shields.io/badge/Python-3.11+-047B43?style=flat-square" alt="Python 3.11+" />
    <a href="https://defender.stackone.com?utm_source=pypi&utm_medium=readme&utm_campaign=badge"><img src="https://img.shields.io/badge/site-defender.stackone.com-047B43?style=flat-square&logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCIgZmlsbD0iI2ZmZiI+PHBhdGggZD0iTTEyIDFMMyA1djZjMCA1LjI1IDMuOCAxMC4xIDkgMTEgNS4yLS45IDktNS43NSA5LTExVjVsLTktNHoiLz48L3N2Zz4=" alt="Defender site" /></a>
  </p>
  <p>
    <img src="https://img.shields.io/badge/model-22MB-047B43?style=flat-square" alt="Model size: 22MB" />
    <img src="https://img.shields.io/badge/latency-~10ms-047B43?style=flat-square" alt="Latency: ~10ms" />
    <img src="https://img.shields.io/badge/CPU--only-no%20GPU%20needed-047B43?style=flat-square" alt="CPU only" />
    <img src="https://img.shields.io/badge/F1%20Score-90.8%25-047B43?style=flat-square" alt="F1 Score: 90.8%" />
  </p>

</div>

---

Indirect prompt injection defense for AI agents using tool calls (MCP, CLI, or direct APIs). Detects and gates prompt injection attacks hidden in tool results (emails, documents, PRs, etc.) before they reach your LLM.

**Python package:** [`stackone-defender`](https://pypi.org/project/stackone-defender/) — aligned with [`@stackone/defender`](https://www.npmjs.com/package/@stackone/defender) on npm.

## Installation

**pip**

```bash
pip install "stackone-defender[onnx]"
```

**uv**

```bash
uv add "stackone-defender[onnx]"
```

The `[onnx]` extra runs Tier 2 (ML classification). The ONNX model (~22MB) is bundled in the wheel, so nothing is downloaded at runtime.

Tier 2 is on by default. Without the `[onnx]` extra, defender logs one warning, sets `result.tier2_available = False` (alert on it) and runs Tier 1 only. Pass `require_tier2=True` to raise instead, or `enable_tier2=False` to run Tier 1 only on purpose.

**SFE preprocessor (optional)** — add extras:

```bash
pip install "stackone-defender[sfe]"
# or: uv add "stackone-defender[sfe]"
```

**Tier 3 over an OpenAI-compatible endpoint (optional)** — add extras:

```bash
pip install "stackone-defender[llm]"
# or: uv add "stackone-defender[llm]"
```

The `[llm]` extra installs [`httpx`](https://www.python-httpx.org/) for `create_openai_compatible_tier3_provider`. `create_llm_tier3_provider` needs no extra.

The `[sfe]` extra installs [`fasttext-ng`](https://pypi.org/project/fasttext-ng/) (provides the `fasttext` module). It requires **NumPy 2.3+**. PyPI may ship a wheel only for some platforms; otherwise pip/uv builds from source (needs a C++ toolchain).

## Quick start

```python
from stackone_defender import create_prompt_defense

# Tier 1 + Tier 2 are on by default. block_high_risk=True enables allow/block.
defense = create_prompt_defense(block_high_risk=True)

# Optional: preload ONNX to avoid first-call latency (requires [onnx] extra)
defense.warmup_tier2()

result = defense.defend_tool_result(tool_output, "gmail_get_message")

if not result.allowed:
    print(f"Blocked: risk={result.risk_level}, score={result.tier2_score}")
    print(f"Detections: {', '.join(result.detections)}")
else:
    send_to_llm(result.sanitized)
```

## How it works

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/StackOneHQ/defender/main/assets/demo-dark.svg" />
  <img src="https://raw.githubusercontent.com/StackOneHQ/defender/main/assets/demo-light.svg" alt="Defender flow: poisoned tool output is sanitized and evaluated; high-risk content can be blocked before the LLM" width="900" />
</picture>

`defend_tool_result()` runs two tiers:

### Tier 1 — Pattern detection (sync, ~1 ms)

Detects (never rewrites) injection signals and records them as evidence:

- **Unicode / leet normalization** (for matching only) — homoglyph resistance (e.g. Cyrillic `а` → ASCII `a`)
- **Role markers** — `SYSTEM:`, `ASSISTANT:`, `<system>`, `[INST]`, etc.
- **Instruction-override patterns** — phrases like “ignore previous instructions”
- **Encoding detection** — decode-then-detect: a decoded payload is escalated only if it trips a real attack pattern
- **Boundary annotation (opt-in)** — `[UD-{id}]…[/UD-{id}]` wrappers when `annotate_boundary=True` (npm: `annotateBoundary`). Use `generate_boundary_instructions` from the package root in prompts when you enable wrapping.

### Tier 2 — ML classification (ONNX)

Packed-chunk MiniLM classifier (int8 ONNX ~22 MB, bundled):

- Split text into sentences, pack to model-sized chunks, score chunks in batched ONNX calls
- Catches paraphrased or novel injections missed by regex
- Uses chunked batch inference to bound memory on large payloads
- The bundled model is multi-head (variant `jbv2-mh-grid-agpjb2pinstr-s2` since 0.9.3; it still lives in the `minilm-multihead-v5` folder so existing paths keep working). Its auxiliary head flags meta-discussion and documentation that quotes injection text.
- The bundled model carries calibrated defaults in its `classifier_config.json` (`high_risk_threshold` 0.52 at temperature 2.41). They override library defaults when the model loads. Thresholds are model-specific: 0.64 was right for the previous model, not this one.
- Multi-head mode is off by default; Tier 2 then blocks at `high_risk_threshold`. To turn it on, pass a preset as `tier2_config={"multihead": MultiheadConfig(main_threshold=..., aux_threshold=...)}`: a chunk then blocks when its main score >= `main_threshold` and its aux score < `aux_threshold`, and the result reports `tier2_aux_score` and `tier2_multihead_blocked`. Presets tuned for this model: High 0.77/0.325, Medium 0.88/0.325, Low 0.94/0.30. They're also in `classifier_config.json` under `multihead_presets`, but defender doesn't load them for you.

### Optional SFE preprocessor

- `use_sfe=True` runs a field-level FastText pass to build a **classifier-only** view of the payload
- **Tier 1** detects on the raw tool value; SFE drops are classifier-only and never remove fields from the returned `sanitized` payload
- **Tier 2** extracts strings from the SFE-filtered tree; `fields_dropped` lists paths omitted from that extraction (not removed from `sanitized`)
- Fails open if the runtime/model is unavailable: payload continues unfiltered

**Benchmarks** (F1 @ threshold 0.5):

| Benchmark | F1 | Samples |
|-----------|-----|--------|
| Qualifire (in-distribution) | 0.8686 | ~1.5k |
| xxz224 (out-of-distribution) | 0.8834 | ~22.5k |
| jayavibhav (adversarial) | 0.9717 | ~1k |
| **Average** | **0.9079** | ~25k |

### Optional Tier 3 — LLM adjudication (bring your own LLM)

An LLM reviews the cases Tier 2 is unsure about (cascade), or every tool result (`tier3_only`). You bring the model. Defender builds the prompt, reads back a `block`/`allow` verdict and applies your policy. Three ways to plug a model in:

1. **`create_openai_compatible_tier3_provider`**: any OpenAI-compatible `/chat/completions` endpoint (OpenAI, vLLM, Ollama, OpenRouter, LiteLLM). Needs the `[llm]` extra.
2. **`create_llm_tier3_provider`**: any other LLM. You write one `complete` function that calls the model; defender owns the prompt, the untrusted-text markers, strict reply parsing and refusal handling.
3. **A hand-written provider**: an object with `classify(text, *, ctx=None)` that owns everything itself.

Two modes, selected via `defender_mode`:
- **`"cascade"`** (default): Tier 1 → Tier 2 → Tier 3, with Tier 3 invoked only when the Tier 2 effective score falls in the gray band (default `[0.3, 0.85)`). The Tier 3 verdict overrides Tier 2 on the escalated chunk — a `block` forces a block, an `allow` rescues it. Outside the band defender skips the round trip.
- **`"tier3_only"`**: skip Tier 2. Only the Tier 3 verdict decides. Tier 1 still fills `detections` but doesn't block. `sanitized` is the original content.

```python
import os

from stackone_defender import create_openai_compatible_tier3_provider, create_prompt_defense

judge = create_openai_compatible_tier3_provider(
    base_url="https://api.openai.com/v1",
    api_key=os.environ["OPENAI_API_KEY"],
    model="gpt-4.1-mini",
)
defense = create_prompt_defense(
    block_high_risk=True,
    enable_tier3=True,
    defender_mode="tier3_only",
    tier3={"provider": judge, "timeout_ms": 10_000, "on_error": "block"},
)

async def handle(tool_output):
    return await defense.defend_tool_result_async(
        tool_output, "gmail_get_message", context={"user_goal": "Summarize my unread mail"}
    )
```

`create_openai_compatible_tier3_provider` options: `headers`, `response_format` (`"json_schema"` by default, or `"json_object"` / `"none"`), `max_tokens` (300, sent as `max_completion_tokens` unless `legacy_max_tokens=True`), `temperature` (not sent unless set), `retries` (1 extra attempt on 429, 5xx and network errors, within the deadline), `prompt`, `parse`, and `client` (an `httpx.AsyncClient` to reuse; recommended in production for connection pooling). `base_url` must use https, except for localhost and `127.x.x.x`; `allow_insecure_http=True` allows plain http anyway. Put credentials in `api_key` or `headers`, not in the URL. A moderation hit (`content_filter`, `invalid_prompt`) counts as a refusal and blocks.

For any other LLM, write a `complete(req)` function. `req` has `messages` (system + user), `deadline` and `json_schema`. Return the reply text, sync or async:

```python
import time

from stackone_defender import create_llm_tier3_provider

async def complete(req):
    # req.deadline is absolute (time.time() seconds), or None when tier3.timeout_ms isn't set.
    timeout = req.deadline - time.time() if req.deadline else 30
    reply = await my_llm.chat(req.messages, timeout=timeout)
    return reply.text  # or {"text": ..., "refused": True} / {"text": ..., "score": p_block}

judge = create_llm_tier3_provider(complete=complete)
```

The default prompt (`DEFAULT_TIER3_PROMPT`, version `TIER3_PROMPT_VERSION`) wraps the tool output, user goal and tool description in random-id markers and asks for `{"decision": "block" | "allow", "reason": "..."}`. Pass `prompt=Tier3PromptTemplate(system=..., user=...)` (the `user` part must contain `{{content}}` once and may use `{{toolName}}`, `{{userGoal}}`, `{{toolDescription}}`; `system` must not contain placeholders), or a builder function that gets a `PromptInput` and returns the message list. A refusal, a content-filter hit, or a reply that isn't exactly one JSON verdict **blocks** (`result.tier3.refused`), because attacker text can cause those on purpose. `result.tier3.reason` is model output that can quote the attacker: log it, but never feed it back to an agent.

A hand-written provider, registered once at startup:

```python
from typing import Any

from stackone_defender import Tier3Verdict, create_prompt_defense, set_default_tier3_provider

class MyProvider:
    def classify(self, text: str, *, ctx: dict[str, Any] | None = None) -> Tier3Verdict:
        # Call your LLM endpoint here (sync or awaitable). ctx has tool_name, mode, chunk,
        # user_goal / tool_description (untrusted), and deadline when timeout_ms is set.
        verdict = call_my_llm(text, tool_name=(ctx or {}).get("tool_name"))
        return Tier3Verdict(decision="block" if verdict.block else "allow", score=verdict.p_block)

set_default_tier3_provider(MyProvider())

defense = create_prompt_defense(
    block_high_risk=True,
    enable_tier3=True,
    defender_mode="cascade",  # or "tier3_only"
    tier3={
        "provider": MyProvider(),                          # optional: overrides the registry for this instance
        "escalation_band": {"lower": 0.3, "upper": 0.85},  # cascade gray band; [lower, upper), defaults shown
        "max_text_length": 10000,                          # max chars per provider call
        "max_chunks": 5,                                   # tier3_only: chunks reviewed per tool result
        "on_oversize": "skip",                             # tier3_only: "skip" | "block" | "scan_anyway"
        "block_threshold": 0.622,                          # optional; decide on score, not the model's word
    },
)
```

Defender looks up the registered provider on every call, so registering after construction works. `set_default_tier3_provider(None)` clears it (useful in tests), and `get_default_tier3_provider()` reads it.

**What the reviewer sees (`tier3_only`).** The tool result is serialized as `field: value` lines, one block per record, so a bare value keeps its field name (`status: create` rather than `create`, which reads like an instruction). Nested keys are dotted (`user.name: …`), list items are indexed (`items[0]: …`), and each line of a multi-line value keeps its prefix. A large list of only numbers collapses to `key: [N numbers]`. Keys are reviewed even when the value is a number, boolean, `None` or empty, because keys can carry injected text. Python-only types are reviewed too: mappings and dataclasses as objects, sets and other sequences as lists, UTF-8 `bytes` as text, and anything else as `str(value)`.

The serialized text is split into chunks of at most `max_text_length` characters (lengths count UTF-16 code units, as in the TypeScript package). Neighbouring chunks overlap by up to 500 characters, so an injection cut at a chunk boundary still appears whole. All chunks are reviewed in parallel, and any blocking chunk blocks the result. `result.tier3_chunk_summary` reports `chunks`, `blocked` and `blocking_chunk`.

**Oversize (`on_oversize`).** Up to about `max_chunks` chunks are reviewed per tool result. Content beyond that is oversize, and `coverage_degraded` is `True`. Blocking below requires `block_high_risk=True`.
- `"skip"` (default): review the content that fits and allow the rest. A fitting injection still blocks. This favors availability, so set `"block"` or `"scan_anyway"` to fail closed.
- `"block"`: block without calling the provider. `tier3_chunk_summary` is `None` and `result.tier3.skip_reason` says why.
- `"scan_anyway"`: review what fits, then block on the unreviewed rest unless a chunk already blocked.

In `"skip"` and `"scan_anyway"`, `tier3_chunk_summary.oversize` is `True`.

A failed chunk is skipped (fail open), and `result.tier3.errored` and `coverage_degraded` are `True`. With a verdict, `result.tier3.note` says how many chunks failed; it also gives the reason for an `on_oversize` block when the provider allowed. When every chunk fails there's no verdict, and `result.tier3.skip_reason` gives the first failure.

**Choosing the operating point (`block_threshold`).** By default the model's `decision` word decides. That word is an implicit 0.5 cut that moves whenever the model is retrained. Set `tier3.block_threshold` to decide on `verdict.score` (P(block)) instead. Raise it for fewer false positives, lower it for more recall; `0.5` reproduces the word. The provider must report `score` as a real P(block), not "confidence in whichever decision I made". If `score` is missing or outside `[0, 1]`, defender uses `decision` and warns once.

**Policy options** (all in `tier3={...}`; camelCase keys work too):

| Option | Default | What it does |
|---|---|---|
| `allow_override` | `True` | Cascade: whether a Tier-3 `allow` can undo a Tier-2 block. `False`: Tier 3 only adds blocks. `{"max_tier2_score": x}`: only when Tier 2's undamped score is below `x`. A general-purpose LLM can be talked into `allow`, so prefer `False` or a cap. Defender warns once when it's `True` with a provider from `create_llm_tier3_provider` or `create_openai_compatible_tier3_provider`. |
| `on_error` | cascade `"tier2"`, tier3_only `"allow"` | What a failure (error, timeout, invalid verdict) does. `"tier2"` keeps Tier 2's verdict. `"allow"` treats it as an allow. `"block"` blocks: in cascade when Tier 2 rated the payload medium or higher (with the bundled model Tier 2 rarely reports `medium`: its band is [0.5, 0.52)); in tier3_only when any chunk failed (and `on_oversize` then defaults to `"scan_anyway"`). A missing provider fails closed too. `"tier2"` isn't valid in tier3_only: defender warns and uses `"allow"`. |
| `decide` | none | `decide(Tier3DecisionInput) -> "block" \| "allow" \| "default"`, called once per call after Tier 3 ran or failed. Replaces `allow_override` and `on_error` for that call. Gets metadata only (never content). Must be synchronous. A raise or bad return means `"default"`. It can't undo a Tier 1 detection, override `on_oversize`, or block when `block_high_risk=False`. |
| `timeout_ms` | none | Per provider call, from when it gets a concurrency slot. An async provider is cancelled at the deadline (`ctx["deadline"]`); a sync one can't be interrupted. |
| `max_concurrency` | unbounded | Max in-flight provider calls per instance, per event loop. Each sync `defend_tool_result` call runs its own loop, so the cap doesn't span threads that use the sync API. Set `timeout_ms` with it. |
| `stop_on_first_block` | `False` | tier3_only: cancel the remaining chunk calls once one blocks. |
| `on_event` | none | Called once per provider call with a `Tier3Event` (kind, mode, chunk, latency, usage, redacted error code). No content. Exceptions are ignored. |
| `include_raw` | `True` | Keep `raw` on `result.tier3`. `raw` can quote tool content. |

**Result fields.** `result.tier3` is a `Tier3Verdict` (or a `Tier3Skip` when no verdict was obtained) and `None` when Tier 3 didn't run. Besides `decision`, `score`, `reason`, `refused` and `prompt_version`, defender sets: `errored` (a real failure), `note` (why the outcome isn't the verdict's alone), `override_applied` / `override_suppressed` (cascade), and `decided_by` (`"policy"` or `"options"` when `decide` is set). Provider error text in `skip_reason`, `note` and events is redacted (keys, tokens, URL paths) and capped. `Tier3Skip`, `Tier3Event` and `Tier3DecisionInput` are in `stackone_defender.types`.

- `allowed` is the final answer. With `allow_override=False`, `result.tier3.decision` can be `"allow"` while `allowed` is `False`.
- Defender warns once after 20 consecutive provider failures on one instance, and once after 20 consecutive refusals or unusable replies. Use `on_event` for alerting.
- A call that never finishes gives its `max_concurrency` slot back after 10 × `timeout_ms`. Waiting for a slot is capped at 10 × `timeout_ms` too.
- Tier 3 sends tool output to the endpoint you configure. `reason` and `raw` can quote it: don't log them where tool content mustn't go, and set `include_raw=False` if needed.

**Cancellation.** Cancelling the task that awaits `defend_tool_result_async` cancels in-flight provider calls and raises `CancelledError`.

**Fail-open by default.** Without `on_error="block"`, a provider error or timeout records a `skip_reason`; in cascade defender falls back to the Tier 2 decision, in `tier3_only` it allows. `enable_tier3=True` with no provider (and no `on_error="block"`) falls back to Tier 1 + Tier 2 and logs one warning per instance, so Tier 3 misconfiguration never silently disables defense.

### `allowed` vs `risk_level`

- **`DefenseResult.sanitized`** is a **sentence-level cleaned** copy of the tool result (high-scoring sentences dropped within high-risk fields, optionally `[UD-…]` boundary-wrapped). Cleaning is best-effort (capped by detection) — still gate on `allowed`. Set `sanitize_content=False` for pure detect-and-gate: `sanitized` is then the input verbatim.
- Use **`allowed`** for gating when `block_high_risk=True`: `False` means do not pass `sanitized` to the model as-is.
- **`risk_level`** is diagnostic: it starts at `default_risk_level` (default `"low"`) and is **escalated** by Tier 1 / Tier 2 signals. When Tier 3 runs, its verdict can lower it (a cascade `allow` override) or set it (`tier3_only`). Use it for logging, not as the sole block signal unless you implement your own policy.

| Level | Typical trigger |
|-------|------------------|
| `low` | No strong signals |
| `medium` | Lighter pattern / sanitization signals |
| `high` / `critical` | Strong injection patterns, encoding signals, or high Tier 2 score |

## API

### `create_prompt_defense(**kwargs)`

```python
defense = create_prompt_defense(
    enable_tier1=True,
    enable_tier2=True,
    require_tier2=False,  # True: raise if Tier 2 can't load (fail closed) instead of degrading to Tier 1
    block_high_risk=False,
    default_risk_level="low",
    annotate_boundary=False,  # True: wrap risky strings with [UD-…] tags (npm: annotateBoundary)
    sanitize_content=True,  # False: detect-and-gate; sanitized is the input verbatim
    tier2_fields=["subject", "body", "snippet"],  # optional: scope Tier 2 to these JSON keys (default: all strings)
    tier2_config={"high_risk_threshold": 0.52},  # optional; defaults come from the bundled model's classifier_config.json
    use_sfe=False,  # True, or {"threshold": 0.5, "predictor": ...}: SFE preprocessing before Tier 2
    enable_tier3=False,
    defender_mode="cascade",  # or "tier3_only"; ignored unless enable_tier3
    tier3=None,  # dict of Tier 3 options (see Tier 3 above)
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
    risk_level: RiskLevel                     # diagnostic; max of Tier 1 / Tier 2
    sanitized: Any                            # sentence-cleaned copy (input verbatim when sanitize_content=False); dropped runs leave a [CONTENT SANITISED] marker; best-effort, still gate on allowed
    detections: list[str]                     # Tier 1 pattern names detected
    fields_sanitized: list[str]               # fields whose content the cleaner changed in sanitized (empty when sanitize_content=False or no Tier 2); for detections read detections/patterns_by_field
    patterns_by_field: dict[str, list[str]]   # patterns detected per field
    detected_field_count: int                 # count of fields with a Tier-1 detection (keys of patterns_by_field); threat-count signal (fields_sanitized len no longer tracks this)
    tier2_score: float | None = None
    tier2_raw_score: float | None = None
    tier2_aux_score: float | None = None      # set only when tier2_config["multihead"] is set
    tier2_multihead_blocked: bool | None = None
    tier2_skip_reason: str | None = None
    max_sentence: str | None = None
    tier3: Tier3Result | None = None          # present when Tier 3 ran
    fields_dropped: list[str] = field(default_factory=list)
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
    tier3_chunk_summary: Tier3ChunkSummary | None = None  # tier3_only: chunks, blocked, blocking_chunk, oversize
```

### `defense.defend_tool_results(items)`

Sync batch API. Each item is `{"value": ..., "tool_name": ..., "context": {...}}` (`context` optional). Without Tier 3 it runs items one by one. With `enable_tier3=True` it runs them concurrently in one `asyncio.run()` (sync providers still run one at a time) and raises `RuntimeError` inside a running event loop. From async code, use `defend_tool_results_async`.

```python
results = defense.defend_tool_results([
    {"value": email_data, "tool_name": "gmail_get_message"},
    {"value": doc_data, "tool_name": "documents_get"},
    {"value": pr_data, "tool_name": "github_get_pull_request"},
])
for r in results:
    if not r.allowed:
        print("Blocked:", ", ".join(r.detections))
```

### `await defense.defend_tool_results_async(items)`

Async batch API — runs `defend_tool_result_async` per item concurrently via `asyncio.gather`. Required when Tier 3 is enabled inside a running event loop (e.g. FastAPI).

```python
results = await defense.defend_tool_results_async([
    {"value": email_data, "tool_name": "gmail_get_message"},
    {"value": doc_data, "tool_name": "documents_get"},
])
```

### `defense.analyze(text)`

Tier 1 only — useful for debugging pattern hits without full tool-result traversal.

### Tier 2 warmup

```python
defense = create_prompt_defense()
defense.warmup_tier2()  # no-op if enable_tier2=False; warns (or raises with require_tier2=True) if the [onnx] extra is missing
```

## Integration example

```python
from stackone_defender import create_prompt_defense

defense = create_prompt_defense(block_high_risk=True)
defense.warmup_tier2()

def run_tool_and_defend(raw_result: dict, tool_name: str):
    outcome = defense.defend_tool_result(raw_result, tool_name)
    if not outcome.allowed:
        return {"error": "Content blocked by safety filter", "risk_level": outcome.risk_level}
    return outcome.sanitized

# Example agent loop
sanitized = run_tool_and_defend(gmail_api.get_message(msg_id), "gmail_get_message")
```

## Risky field detection

Only **string** values under configured “risky” keys are Tier-1-scanned — including strings nested inside arrays/objects under those keys (e.g. `{"name": ["…"]}`). [`RiskyFieldConfig`](https://github.com/StackOneHQ/defender-py/blob/main/src/stackone_defender/types.py) provides global names/patterns plus **`tool_overrides`** (wildcard tool names → field list), same idea as the npm package. Tier 1 also scans object keys. Tier 2 scans all string values whatever the field name (unless `tier2_fields` is set), so an injection in `id` or `url` is still scored.

| Tool pattern | Scanned fields |
|--------------|----------------|
| `gmail_*`, `email_*` | subject, body, snippet, content |
| `documents_*` | name, description, content, title |
| `github_*` | name, title, body, description, message, content |
| `hris_*` | name, notes, bio, description |
| `ats_*` | name, notes, description, summary |
| `crm_*` | name, description, notes, content |

Otherwise the default list applies: `name`, `description`, `content`, `title`, `notes`, `summary`, `bio`, `body`, `text`, `message`, `comment`, `subject`, plus suffix patterns like `*_body`, `*_description`, etc. Structural keys such as `id`, `url`, `created_at` are not treated as risky by default.

## Development

The bundled models (`*.onnx`, `*.ftz`) are stored with [Git LFS](https://git-lfs.com). Install it before cloning, or run `git lfs install && git lfs pull` in an existing clone. Otherwise the model files are ~130-byte pointers: the build fails with a "Git LFS pointer" error, and Tier 2 / SFE won't load.

```bash
git lfs install && git lfs pull
uv sync --group dev
uv run pytest
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

## Defender by StackOne

Defender is built and maintained by [StackOne](https://www.stackone.com?utm_source=pypi&utm_medium=readme&utm_campaign=about), the enterprise integration platform for AI agents. We run Defender in production to protect agent tool calls against indirect prompt injection.

- **Defender site** — [defender.stackone.com](https://defender.stackone.com?utm_source=pypi&utm_medium=readme&utm_campaign=about)
- **Overview & benchmarks** — [stackone.com/platform/prompt-injection-guard](https://www.stackone.com/platform/prompt-injection-guard/?utm_source=pypi&utm_medium=readme&utm_campaign=about)
- **StackOne platform** — [stackone.com](https://www.stackone.com?utm_source=pypi&utm_medium=readme&utm_campaign=about)
- **TypeScript package** — [`@stackone/defender`](https://www.npmjs.com/package/@stackone/defender) ([StackOneHQ/defender](https://github.com/StackOneHQ/defender))

## License

Apache-2.0 — see [LICENSE](./LICENSE).
