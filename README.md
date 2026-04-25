# LLM Eval Framework

[中文文档](README_CN.md)

A clean evaluation framework for code and reasoning benchmarks with `uv`-managed environments, a minimal model config, structured output artifacts, and optional thinking mode for OpenAI-compatible endpoints.

## Highlights

- `uv` as the default environment and dependency workflow
- Single model config file
- Built-in support for `humaneval`, `humanevalplus`, `mbpp`, and `gsm`
- OpenAI-compatible client interface
- Structured run artifacts in JSON and JSONL
- Optional thinking mode with `reasoning_effort`
- Immediate startup logs and live progress output during evaluation

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
  model.yaml
datasets/
tests/
```

## Install

Recommended:

```bash
uv sync
```

Install with dev dependencies:

```bash
uv sync --extra dev
```

Run commands through the managed environment:

```bash
uv run python -m llm_eval run --config configs/model.yaml --task mbpp
```

## Model config

The framework uses a single config file: [configs/model.yaml](/C:/Users/15080/Desktop/tests/llm-eval-framework/configs/model.yaml)

```yaml
base_url: ${OPENAI_BASE_URL:-https://api.openai.com/v1}
api_key: ${OPENAI_API_KEY}
model_name: gpt-4.1-mini
workers: 10
thinking_enabled: false
reasoning_effort:
```

Runtime settings kept in YAML:

- `base_url`
- `api_key`
- `model_name`
- `workers`
- `thinking_enabled`
- `reasoning_effort`

The output directory is fixed to:

```text
results/<MODEL_NAME>/
```

## Thinking mode

For providers that support it, the client can send:

- `reasoning_effort="low|medium|high"`
- `extra_body={"thinking": {"type": "enabled"}}`

Equivalent SDK style:

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

Framework usage:

```bash
uv run python -m llm_eval run --config configs/model.yaml --task humaneval
```

## CLI

Run a specific benchmark:

```bash
uv run python -m llm_eval run --config configs/model.yaml --task gsm
```

Use a custom model config:

```bash
uv run python -m llm_eval run --config configs/model.yaml --task mbpp
```

The command intentionally stays minimal:

```bash
uv run python -m llm_eval run --config configs/model.yaml --task humaneval
```

When a run starts, the framework prints:

- selected task
- selected model
- dataset path
- worker count
- thinking mode
- output directory
- live progress updates while cases are being processed

## Outputs

Each run writes to:

```text
results/<model_name>/
```

Artifacts:

- `<task>_results.jsonl`
- `<task>_summary.json`
- `<task>_report.md`
- `resolved_config.json`

## Supported tasks

- `humaneval`
- `humanevalplus`
- `mbpp`
- `gsm`

## Tests

```bash
uv run pytest -q
```
