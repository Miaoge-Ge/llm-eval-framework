# LLM Eval Framework

[中文文档](README_CN.md)

A clean, engineering-first evaluation framework for code and reasoning benchmarks. It uses `uv`-managed environments, a single minimal model config, and produces one self-contained Markdown report per run against any OpenAI-compatible endpoint.

## Highlights

- `uv` as the default environment and dependency workflow
- Installable package (`hatchling` build backend) with a `llm-eval` console script
- Single model config file, with `${ENV_VAR}` placeholders for secrets
- Built-in tasks: `humaneval`, `humanevalplus`, `mbpp`, `gsm`, `math500`, `gpqa`
- OpenAI-compatible client with streaming, retries, and usage accounting
- One self-contained Markdown report per run (config + metrics + per-case results)
- Optional thinking mode with `reasoning_effort`
- Immediate startup logs and a live progress bar during evaluation
- Quality tooling wired in: `ruff` (lint + format), `mypy` (type checking), `pytest`

## Project structure

```text
llm_eval/
  cli.py
  clients.py
  reporting.py
  runner.py
  settings.py
  tasks.py
  utils.py
configs/
  model.example.yaml   # committed template (no secrets)
datasets/
tests/
pyproject.toml
```

## Install

Recommended:

```bash
uv sync
```

Install with dev dependencies (`pytest`, `ruff`, `mypy` and type stubs):

```bash
uv sync --extra dev
```

## Model config

Copy the committed template to a local config (the `configs/` directory is git-ignored apart from the example, so your real keys never get committed):

```bash
cp configs/model.example.yaml configs/model.yaml
```

[configs/model.example.yaml](configs/model.example.yaml):

```yaml
base_url: ${OPENAI_BASE_URL:-https://api.openai.com/v1}
api_key: ${OPENAI_API_KEY}
model_name: gpt-4.1-mini
workers: 10
timeout_seconds: 120
execution_timeout_seconds: 20
thinking_enabled: false
reasoning_effort:
```

Prefer `${ENV_VAR}` placeholders for secrets so the API key is never written to disk. Recognized fields:

| Field | Purpose |
| --- | --- |
| `base_url` | OpenAI-compatible endpoint |
| `api_key` | API key (use `${OPENAI_API_KEY}`) |
| `model_name` | Model id sent to the endpoint; also names the output directory |
| `workers` | Concurrent requests (default `10`) |
| `timeout_seconds` | Per-request HTTP timeout (default `120`) |
| `execution_timeout_seconds` | Cap for running generated code during grading (default `20`) |
| `thinking_enabled` | `true`/`false` (also accepts `enabled`/`disabled`) |
| `reasoning_effort` | `low`/`medium`/`high`/`max`, sent only when thinking is enabled |

The output directory is fixed to `results/<model_name>/`.

## CLI

```bash
# run a benchmark
uv run llm-eval run --config configs/model.yaml --task mbpp

# equivalent module form
uv run python -m llm_eval run --config configs/model.yaml --task gsm
```

`--config` defaults to `configs/model.yaml` and `--task` defaults to `humaneval`.

When a run starts, the framework prints the selected task, model, dataset path, worker count, thinking mode, and output directory, then shows a live progress bar (passed / failed / last HTTP status) while cases are processed.

## Thinking mode

For providers that support it, the client sends `reasoning_effort` and `extra_body={"thinking": {"type": "enabled"}}` when `thinking_enabled` is on. Equivalent SDK call:

```python
from openai import OpenAI

client = OpenAI(api_key="...", base_url="https://api.deepseek.com")

response = client.chat.completions.create(
    model="deepseek-v4-pro",
    messages=[
        {"role": "system", "content": "You are a helpful assistant"},
        {"role": "user", "content": "Hello"},
    ],
    reasoning_effort="high",
    extra_body={"thinking": {"type": "enabled"}},
)
```

## Supported tasks

| Task | Dataset | Cases | Grading |
| --- | --- | --- | --- |
| `humaneval` | HumanEval | 164 | Execute generated code against unit tests |
| `humanevalplus` | HumanEval+ | 164 | Same, with extended tests and a numpy shim |
| `mbpp` | MBPP | 974 | Execute generated code against assertion tests |
| `gsm` | GSM8K | 1319 | Exact match on the final numeric answer |
| `math500` | MATH-500 | 500 | Normalize and match the `\boxed{}` answer |
| `gpqa` | GPQA-Diamond | 198 | Multiple-choice letter from `\boxed{}` |

## Output

Each run writes a single self-contained Markdown report:

```text
results/<model_name>/<task>_report.md
```

It contains:

- **Overview** — task, model, dataset, workers, thinking mode
- **Metrics** — pass rate, wall clock, throughput, prompt/completion/total tokens
- **Status counts** — how many cases passed, failed, errored, etc.
- **Results** — a per-case table with status, time, tokens, and a detail column

## Development

All tooling is configured in `pyproject.toml` and runs through `uv`:

```bash
uv run ruff check .    # lint
uv run ruff format .   # auto-format
uv run mypy            # type check
uv run pytest          # tests
```
