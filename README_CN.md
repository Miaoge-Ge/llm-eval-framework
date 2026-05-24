# LLM Eval Framework

[English Documentation](README.md)

一个面向工程使用的 LLM 评测框架，默认使用 `uv` 管理环境，主打极简模型配置、结构化结果产物，以及支持 OpenAI-compatible 接口的思考模式。

## 特点

- 默认使用 `uv` 管理依赖与运行环境
- 可安装的 Python 包（`hatchling` 构建后端），并提供 `llm-eval` 命令行入口
- 单一模型配置文件
- 内置 `humaneval`、`humanevalplus`、`mbpp`、`gsm`
- OpenAI-compatible 调用方式
- 每次运行都会产出结构化 JSON / JSONL
- 支持 `thinking` 与 `reasoning_effort`
- 启动即打印关键信息，评测过程中持续输出进度
- 内置质量工具链：`ruff`（检查 + 格式化）、`mypy`（类型检查）、`pytest`

## 目录结构

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
pyproject.toml
```

## 安装

推荐方式：

```bash
uv sync
```

安装开发依赖（`pytest`、`ruff`、`mypy` 及类型存根）：

```bash
uv sync --extra dev
```

通过 `uv` 管理的环境执行命令：

```bash
uv run llm-eval run --config configs/model.yaml --task mbpp
# 等价的模块形式
uv run python -m llm_eval run --config configs/model.yaml --task mbpp
```

## 模型配置

框架只保留一个配置文件：[configs/model.yaml](/C:/Users/15080/Desktop/tests/llm-eval-framework/configs/model.yaml)

```yaml
base_url: ${OPENAI_BASE_URL:-https://api.openai.com/v1}
api_key: ${OPENAI_API_KEY}
model_name: gpt-4.1-mini
workers: 10
thinking_enabled: false
reasoning_effort:
```

建议使用 `${ENV_VAR}` 占位符注入密钥，避免把 API Key 明文写入文件。

YAML 中保留的运行配置：

- `base_url`
- `api_key`
- `model_name`
- `workers`
- `thinking_enabled`
- `reasoning_effort`

输出目录固定为：

```text
results/<MODEL_NAME>/
```

## 思考模式

对于支持该能力的服务端，框架会发送：

- `reasoning_effort="low|medium|high"`
- `extra_body={"thinking": {"type": "enabled"}}`

对应的 SDK 调用形式类似：

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

在框架里开启方式：

```bash
uv run llm-eval run --config configs/model.yaml --task humaneval
```

## CLI 用法

运行指定任务：

```bash
uv run llm-eval run --config configs/model.yaml --task gsm
```

指定模型配置文件：

```bash
uv run llm-eval run --config configs/model.yaml --task mbpp
```

最终命令格式保持最小化：

```bash
uv run llm-eval run --config configs/model.yaml --task humaneval
```

命令启动后会立即打印：

- 当前任务
- 当前模型
- 数据集路径
- 并发数
- 思考模式
- 输出目录
- 运行中的实时进度

## 输出产物

每次运行都会写入：

```text
results/<model_name>/
```

其中包含：

- `<task>_results.jsonl`
- `<task>_summary.json`
- `<task>_report.md`
- `resolved_config.json`

## 当前支持任务

- `humaneval`
- `humanevalplus`
- `mbpp`
- `gsm`

## 开发

所有工具均在 `pyproject.toml` 中配置，并通过 `uv` 运行：

```bash
# 代码检查
uv run ruff check .

# 自动格式化
uv run ruff format .

# 类型检查
uv run mypy

# 运行测试
uv run pytest
```
