# Optional LLM review

This reference covers the open source provider adapters for a model you supply. StackOne's hosted Tier 3 model and service are separate and are not bundled in this package. For local Tier 1 + Tier 2 usage and evaluations, see the [README](../README.md).

Install the optional HTTP client for an OpenAI-compatible endpoint with `pip install "stackone-defender[llm]"`.

<a id="optional-tier-3--llm-adjudication-bring-your-own-llm"></a>

### Optional Tier 3: LLM review (bring your own LLM)

An LLM reviews the cases Tier 2 is unsure about (cascade), or every tool result (`tier3_only`). You bring the model. Defender builds the prompt, reads back a `block`/`allow` verdict and applies your policy. Three ways to plug a model in:

1. **`create_openai_compatible_tier3_provider`**: any OpenAI-compatible `/chat/completions` endpoint (OpenAI, vLLM, Ollama, OpenRouter, LiteLLM). Needs the `[llm]` extra.
2. **`create_llm_tier3_provider`**: any other LLM. You write one `complete` function that calls the model; defender owns the prompt, the untrusted-text markers, strict reply parsing and refusal handling.
3. **A hand-written provider**: an object with `classify(text, *, ctx=None)` that owns everything itself.

Two modes, selected via `defender_mode`:

- **`"cascade"`** (default): Tier 1 → Tier 2 → Tier 3, with Tier 3 invoked only when the Tier 2 effective score falls in the gray band (default `[0.3, 0.85)`). The Tier 3 verdict overrides Tier 2 on the escalated chunk: a `block` forces a block, an `allow` can undo a Tier 2 block, subject to `allow_override`. Outside the band defender skips the round trip.
- **`"tier3_only"`**: skip Tier 2. Tier 3 verdicts and your error / oversize policies decide whether to block. Tier 1 still fills `detections` but doesn't block. `sanitized` is the original content.

Set `OPENAI_API_KEY` and `OPENAI_MODEL` before running this example. Adjust `response_format` and token options to match your endpoint.

```python
import os

from stackone_defender import create_openai_compatible_tier3_provider, create_prompt_defense

judge = create_openai_compatible_tier3_provider(
    base_url="https://api.openai.com/v1",
    api_key=os.environ["OPENAI_API_KEY"],
    model=os.environ["OPENAI_MODEL"],  # A model supported by your endpoint
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
        # user_goal / tool_description / connector / args (untrusted; args as a JSON string cut
        # to 2,000 characters), and deadline when timeout_ms is set.
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

The serialized text is split into chunks of at most `max_text_length` characters (lengths count UTF-16 code units, as in the TypeScript package). Neighbouring chunks overlap by up to 500 characters, reducing the chance that an injection is split across review windows. All chunks are reviewed in parallel, and any blocking chunk blocks the result. `result.tier3_chunk_summary` reports `chunks`, `blocked` and `blocking_chunk`.

**Oversize (`on_oversize`).** Up to about `max_chunks` chunks are reviewed per tool result. Content beyond that is oversize, and `coverage_degraded` is `True`. Blocking below requires `block_high_risk=True`.

- `"skip"` (default): review the content that fits and allow the rest. A fitting injection still blocks. This favors availability, so set `"block"` or `"scan_anyway"` to fail closed.
- `"block"`: block without calling the provider. `tier3_chunk_summary` is `None` and `result.tier3.skip_reason` says why.
- `"scan_anyway"`: review what fits, then block on the unreviewed rest unless a chunk already blocked.

In `"skip"` and `"scan_anyway"`, `tier3_chunk_summary.oversize` is `True`.

With the default `on_error="allow"`, a failed chunk is skipped (fail open), and `result.tier3.errored` and `coverage_degraded` are `True`. With a verdict, `result.tier3.note` says how many chunks failed; it also gives the reason for an `on_oversize` block when the provider allowed. When every chunk fails there's no verdict, and `result.tier3.skip_reason` gives the first failure.

**Choosing the operating point (`block_threshold`).** By default, Defender uses the provider's `decision` label. The label alone does not define a calibrated probability threshold. Set `tier3.block_threshold` to decide on `verdict.score` (P(block)) instead. Raise the threshold to require a higher score to block, or lower it to block at a lower score. Evaluate the trade-off on representative tool results. The provider must report `score` as a real P(block), not "confidence in whichever decision I made". If `score` is missing or outside `[0, 1]`, defender uses `decision` and warns once.

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

**Error defaults.** A provider error or timeout records a `skip_reason`. The default `on_error="tier2"` in cascade preserves the Tier 2 decision; the default `on_error="allow"` in `tier3_only` skips failed chunks, while successful chunks can still block. `enable_tier3=True` with no provider falls back to Tier 1 + Tier 2 and logs one warning per instance, unless `on_error="block"` is configured. See the policy table above for overrides.
