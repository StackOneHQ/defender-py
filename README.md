<div align="center">

  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/StackOneHQ/defender/main/assets/banner-dark.svg" />
    <img src="https://raw.githubusercontent.com/StackOneHQ/defender/main/assets/banner-light.svg" alt="Defender by StackOne — Indirect prompt injection protection for MCP tool calls" width="800" />
  </picture>

  <p>
    <a href="https://pypi.org/project/stackone-defender/"><img src="https://img.shields.io/pypi/v/stackone-defender?style=flat-square&color=047B43&label=pypi" alt="PyPI version" /></a>
    <a href="https://github.com/StackOneHQ/stackone-defender/releases"><img src="https://img.shields.io/github/v/release/StackOneHQ/stackone-defender?style=flat-square&color=047B43&label=release" alt="latest GitHub release" /></a>
    <a href="https://github.com/StackOneHQ/stackone-defender/stargazers"><img src="https://img.shields.io/github/stars/StackOneHQ/stackone-defender?style=flat-square&color=047B43" alt="GitHub stars" /></a>
    <a href="./LICENSE"><img src="https://img.shields.io/pypi/l/stackone-defender?style=flat-square&color=047B43" alt="License" /></a>
    <img src="https://img.shields.io/badge/Python-3.11+-047B43?style=flat-square" alt="Python 3.11+" />
  </p>
  <p>
    <img src="https://img.shields.io/badge/model-22MB-047B43?style=flat-square" alt="Model size: 22MB" />
    <img src="https://img.shields.io/badge/latency-~10ms-047B43?style=flat-square" alt="Latency: ~10ms" />
    <img src="https://img.shields.io/badge/CPU--only-no%20GPU%20needed-047B43?style=flat-square" alt="CPU only" />
    <img src="https://img.shields.io/badge/F1%20Score-90.8%25-047B43?style=flat-square" alt="F1 Score: 90.8%" />
  </p>

</div>

---

Indirect prompt injection defense for AI agents using tool calls (MCP, CLI, or direct APIs). Detects and neutralizes attacks hidden in tool results (emails, documents, PRs, etc.) before they reach your LLM.

**Python package:** [`stackone-defender`](https://pypi.org/project/stackone-defender/) — aligned with [`@stackone/defender`](https://www.npmjs.com/package/@stackone/defender) on npm.

## Installation

**pip**

```bash
pip install stackone-defender
```

**uv**

```bash
uv add stackone-defender
```

**Tier 2 (ONNX)** — add extras:

```bash
pip install stackone-defender[onnx]
# or: uv add "stackone-defender[onnx]"
```

The ONNX model (~22MB) is bundled in the wheel — no extra downloads at runtime.

**SFE preprocessor (optional)** — add extras:

```bash
pip install stackone-defender[sfe]
# or: uv add "stackone-defender[sfe]"
```

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

- **Unicode normalization** — homoglyph resistance (e.g. Cyrillic `а` → ASCII `a`)
- **Role stripping** — `SYSTEM:`, `ASSISTANT:`, `<system>`, `[INST]`, etc.
- **Pattern removal** — phrases like “ignore previous instructions”
- **Encoding detection** — suspicious Base64/URL-shaped payloads
- **Boundary annotation (opt-in)** — `[UD-{id}]…[/UD-{id}]` wrappers when `annotate_boundary=True` (npm: `annotateBoundary`). Use `generate_boundary_instructions` from the package root in prompts when you enable wrapping.

### Tier 2 — ML classification (ONNX)

Packed-chunk MiniLM classifier (int8 ONNX ~22 MB, bundled):

- Split text into sentences, pack to model-sized chunks, score chunks in batched ONNX calls
- Catches paraphrased or novel injections missed by regex
- Uses chunked batch inference to bound memory on large payloads

### Optional SFE preprocessor

- `use_sfe=True` runs a field-level FastText pass to build a **classifier-only** view of the payload
- **Tier 1** always sanitizes the **original** tool value; **`sanitized`** in `DefenseResult` is unchanged by SFE drops
- **Tier 2** extracts strings from the SFE-filtered tree; `fields_dropped` lists paths omitted from that extraction (not removed from `sanitized`)
- Fails open if the runtime/model is unavailable: payload continues unfiltered

**Benchmarks** (F1 @ threshold 0.5):

| Benchmark | F1 | Samples |
|-----------|-----|--------|
| Qualifire (in-distribution) | 0.8686 | ~1.5k |
| xxz224 (out-of-distribution) | 0.8834 | ~22.5k |
| jayavibhav (adversarial) | 0.9717 | ~1k |
| **Average** | **0.9079** | ~25k |

### `allowed` vs `risk_level`

- Use **`allowed`** for gating when `block_high_risk=True`: `False` means do not pass `sanitized` to the model as-is.
- **`risk_level`** is diagnostic: it starts at `default_risk_level` (default `"medium"`) and is **escalated** by Tier 1 / Tier 2 signals — not reduced. Use it for logging, not as the sole block signal unless you implement your own policy.

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
    block_high_risk=False,
    default_risk_level="medium",
    annotate_boundary=False,  # True: wrap risky strings with [UD-…] tags (npm: annotateBoundary)
    tier2_fields=["subject", "body", "snippet"],  # optional: scope Tier 2 to these JSON keys (default: all strings)
    use_sfe=True,  # optional: enable semantic field extractor preprocessing
    config={
        "tier2": {
            "high_risk_threshold": 0.8,
            "tier2_fields": None,  # or list[str]; constructor tier2_fields wins if set
        },
    },
)
```

### `defense.defend_tool_result(value, tool_name)`

Runs Tier 1 sanitization on risky fields of the **original** payload, then Tier 2 on strings from the SFE-filtered view when SFE is on (otherwise the full value). Optional `tier2_fields` restricts Tier 2 extraction to specific keys; omit it to classify **all** strings (matches `@stackone/defender` 0.6.3). **Synchronous** — no `await`.

```python
from dataclasses import dataclass, field

@dataclass
class DefenseResult:
    allowed: bool
    risk_level: RiskLevel
    sanitized: Any
    detections: list[str]
    fields_sanitized: list[str]
    patterns_by_field: dict[str, list[str]]
    tier2_score: float | None = None
    tier2_skip_reason: str | None = None
    max_sentence: str | None = None
    fields_dropped: list[str] = field(default_factory=list)
    truncated_at_depth: bool | None = None
    latency_ms: float = 0.0
```

### `defense.defend_tool_results(items)`

```python
results = defense.defend_tool_results([
    {"value": email_data, "tool_name": "gmail_get_message"},
    {"value": doc_data, "tool_name": "documents_get"},
    {"value": pr_data, "tool_name": "github_get_pull_request"},
])
for r in results:
    if not r.allowed:
        print("Blocked:", ", ".join(r.fields_sanitized))
```

### `defense.analyze(text)`

Tier 1 only — useful for debugging pattern hits without full tool-result traversal.

### Tier 2 warmup

```python
defense = create_prompt_defense()
defense.warmup_tier2()  # no-op if enable_tier2=False or ONNX extra missing
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

Only **string** values under configured “risky” keys are scanned and sanitized. [`RiskyFieldConfig`](https://github.com/StackOneHQ/stackone-defender/blob/main/src/stackone_defender/types.py) provides global names/patterns plus **`tool_overrides`** (wildcard tool names → field list), same idea as the npm package.

| Tool pattern | Scanned fields |
|--------------|----------------|
| `gmail_*`, `email_*` | subject, body, snippet, content |
| `documents_*` | name, description, content, title |
| `github_*` | name, title, body, description, message |
| `hris_*` | name, notes, bio, description |
| `ats_*` | name, notes, description, summary |
| `crm_*` | name, description, notes, content |

Otherwise the default list applies: `name`, `description`, `content`, `title`, `notes`, `summary`, `bio`, `body`, `text`, `message`, `comment`, `subject`, plus suffix patterns like `*_body`, `*_description`, etc. Structural keys such as `id`, `url`, `created_at` are not treated as risky by default.

## Development

```bash
uv sync --group dev
uv run pytest
```

## License

Apache-2.0 — see [LICENSE](./LICENSE).
