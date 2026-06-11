# LLM Eval Framework

[中文文档](README_CN.md)

A clean, engineering-first evaluation framework for LLM benchmarks. It uses `uv`-managed environments, a single minimal model config, and produces one self-contained Markdown report per run against any OpenAI-compatible endpoint.

## Highlights

- `uv` as the default environment and dependency workflow
- Installable package (`hatchling` build backend) with a `llm-eval` console script
- Single model config file, with `${ENV_VAR}` placeholders for secrets
- 10 built-in tasks across code generation, math reasoning, and multiple-choice knowledge
- OpenAI-compatible client with streaming, retries, and usage accounting
- One self-contained Markdown report per run (config + metrics + per-case results)
- Per-subject / per-domain accuracy breakdown for all knowledge tasks
- Optional thinking mode with configurable `reasoning_effort`
- Live progress bar with pass/fail counts during evaluation
- Quality tooling wired in: `ruff` (lint + format), `mypy` (type checking), `pytest`

## Project structure

```text
llm_eval/
  cli.py
  clients.py
  execution.py         # subprocess execution of generated code
  extraction.py        # answer extraction (code blocks, \boxed{}, letters, numbers)
  reporting.py         # run persistence (JSONL + config snapshot) and Markdown rendering
  runner.py
  settings.py
  status.py            # canonical case-status constants
  tasks.py
  utils.py
  ifeval/              # vendored Google IFEval checkers + scoring glue
configs/
  model.example.yaml   # committed template (no secrets)
datasets/
  HumanEval.jsonl
  HumanEvalPlus.jsonl
  MBPP.jsonl
  MBPPPlus.jsonl
  GSM8K.jsonl
  AIME2025.jsonl
  AIME2026.jsonl
  GPQA.jsonl
  MMLUPro.jsonl
  IFEval.jsonl
  LiveCodeBench.jsonl
scripts/
  fetch_datasets.py    # re-download / regenerate datasets from Hugging Face
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

Each run writes to its own timestamped directory under `results/<model_name>/<task>/` (the model name is slugified, so ids like `org/model:beta` are safe on every platform).

## CLI

```bash
# run a benchmark
uv run llm-eval run --config configs/model.yaml --task gpqa

# equivalent module form
uv run python -m llm_eval run --config configs/model.yaml --task gsm

# smoke test: only the first 5 cases, with 2 workers
uv run llm-eval run --config configs/model.yaml --task gsm --limit 5 --workers 2

# list all available tasks
uv run llm-eval run --list-tasks
```

`--config` defaults to `configs/model.yaml` and `--task` defaults to `humaneval`. `--workers` overrides the config value; `--limit N` runs only the first N cases — handy for smoke-testing a new endpoint before a full run.

### Run every benchmark

```bash
# Code generation
uv run llm-eval run --config configs/model.yaml --task humaneval
uv run llm-eval run --config configs/model.yaml --task humanevalplus
uv run llm-eval run --config configs/model.yaml --task mbpp
uv run llm-eval run --config configs/model.yaml --task mbppplus
uv run llm-eval run --config configs/model.yaml --task livecodebench

# Math reasoning
uv run llm-eval run --config configs/model.yaml --task gsm
uv run llm-eval run --config configs/model.yaml --task aime2025
uv run llm-eval run --config configs/model.yaml --task aime2026

# Multiple-choice knowledge
uv run llm-eval run --config configs/model.yaml --task gpqa
uv run llm-eval run --config configs/model.yaml --task mmlu_pro

# Instruction following
uv run llm-eval run --config configs/model.yaml --task ifeval
```

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

### Code generation

| Task | Dataset | Cases | Grading |
| --- | --- | --- | --- |
| `humaneval` | HumanEval | 164 | Execute generated code against unit tests |
| `humanevalplus` | HumanEval+ | 164 | Same, with extended tests and a numpy shim |
| `mbpp` | MBPP | 974 | Execute generated code against assertion tests |
| `mbppplus` | MBPP+ | 378 | EvalPlus extended tests, with a numpy shim |
| `livecodebench` | LiveCodeBench (lite) | 219 | Run as a stdin/stdout program against public test cases |

### Math reasoning

| Task | Dataset | Cases | Grading |
| --- | --- | --- | --- |
| `gsm` | GSM8K | 1319 | Exact match on the final `#### <number>` answer |
| `aime2025` | AIME 2025 (I+II) | 30 | Integer match on the `\boxed{}` answer |
| `aime2026` | AIME 2026 (I+II) | 30 | Integer match on the `\boxed{}` answer |

### Multiple-choice knowledge

The MCQ task asks the model to output `\boxed{A/B/C/D}` and grades by exact letter match. The report includes a per-domain accuracy breakdown.

| Task | Dataset | Cases | Coverage |
| --- | --- | --- | --- |
| `gpqa` | GPQA-Diamond | 198 | PhD-level science (Physics, Chemistry, Biology) |
| `mmlu_pro` | MMLU-Pro | 420 | 14 subjects, 10-way choices (30 sampled per category) |

### Instruction following

| Task | Dataset | Cases | Grading |
| --- | --- | --- | --- |
| `ifeval` | IFEval | 541 | Programmatic verification of each instruction (prompt-level strict accuracy) |

`livecodebench` keeps only stdin/stdout problems (AtCoder/Codeforces) and grades against the plaintext public test cases. `ifeval` vendors Google's instruction checkers under `llm_eval/ifeval/` and depends on `langdetect`. Regenerate any dataset with `uv run --with datasets python scripts/fetch_datasets.py <name>`.

## Output

Each run writes to its own timestamped directory:

```text
results/<model_name>/<task>/<YYYYMMDD-HHMMSS>/
  results.jsonl   # one line per case, written incrementally as cases finish
  config.json     # snapshot of the effective config (API key masked)
  report.md       # the human-readable report
```

`results.jsonl` is flushed after every case, so a crashed or aborted run keeps everything graded so far, and the raw records stay available for re-analysis. Reruns never overwrite earlier results.

`report.md` contains:

- **Overview** — task, model, dataset, workers, thinking mode
- **Metrics** — pass rate, total/completed/graded cases, wall clock, throughput, prompt/completion/total tokens (graded = passed + failed + timeout; API errors are excluded from the pass-rate denominator)
- **Status counts** — how many cases passed, failed, errored, etc.
- **Accuracy by domain** — per-subject breakdown (knowledge tasks only)
- **Results** — a per-case table with status, time, tokens, and a detail column

> **Security note:** code-generation tasks execute model-generated Python directly on your machine with no sandboxing, limited only by `execution_timeout_seconds`. Only evaluate models and datasets you trust, or run inside a container/VM.

## Development

All tooling is configured in `pyproject.toml` and runs through `uv`:

```bash
uv run ruff check .    # lint
uv run ruff format .   # auto-format
uv run mypy            # type check
uv run pytest          # tests
```
