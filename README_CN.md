# LLM Eval Framework

[English Documentation](README.md)

一个面向工程使用的 LLM 评测框架。默认用 `uv` 管理环境，配置极简（单个模型配置文件），每次运行只产出一个自包含的 Markdown 报告，支持任意 OpenAI-compatible 接口。

## 特点

- 默认使用 `uv` 管理依赖与运行环境
- 可安装的 Python 包（`hatchling` 构建后端），提供 `llm-eval` 命令行入口
- 单一模型配置文件，密钥用 `${ENV_VAR}` 占位符注入
- 内置任务：`humaneval`、`humanevalplus`、`mbpp`、`gsm`、`math500`、`gpqa`
- OpenAI-compatible 客户端，支持流式、重试与 token 用量统计
- 每次运行只产出一个自包含 Markdown 报告（配置 + 指标 + 逐条结果）
- 支持 `thinking` 与 `reasoning_effort`
- 启动即打印关键信息，评测过程中显示实时进度条
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
  model.example.yaml   # 已提交的模板（不含密钥）
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

## 模型配置

把已提交的模板复制为本地配置（`configs/` 目录除示例外都被 git 忽略，因此你的真实密钥不会被提交）：

```bash
cp configs/model.example.yaml configs/model.yaml
```

[configs/model.example.yaml](configs/model.example.yaml)：

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

建议用 `${ENV_VAR}` 占位符注入密钥，避免把 API Key 明文写入文件。可识别的字段：

| 字段 | 含义 |
| --- | --- |
| `base_url` | OpenAI-compatible 接口地址 |
| `api_key` | API 密钥（建议用 `${OPENAI_API_KEY}`） |
| `model_name` | 发送给接口的模型 id，同时用于命名输出目录 |
| `workers` | 并发请求数（默认 `10`） |
| `timeout_seconds` | 单次请求的 HTTP 超时（默认 `120`） |
| `execution_timeout_seconds` | 打分时运行生成代码的时间上限（默认 `20`） |
| `thinking_enabled` | `true`/`false`（也接受 `enabled`/`disabled`） |
| `reasoning_effort` | `low`/`medium`/`high`/`max`，仅在开启思考模式时发送 |

输出目录固定为 `results/<model_name>/`。

## CLI 用法

```bash
# 运行某个任务
uv run llm-eval run --config configs/model.yaml --task mbpp

# 等价的模块形式
uv run python -m llm_eval run --config configs/model.yaml --task gsm

# 列出所有可用任务
uv run llm-eval run --list-tasks
```

`--config` 默认 `configs/model.yaml`，`--task` 默认 `humaneval`。

命令启动后会立即打印：当前任务、模型、数据集路径、并发数、思考模式、输出目录，随后在处理用例时显示实时进度条（通过数 / 失败数 / 最近一次 HTTP 状态码）。

## 思考模式

对于支持该能力的服务端，当 `thinking_enabled` 开启时，客户端会发送 `reasoning_effort` 和 `extra_body={"thinking": {"type": "enabled"}}`。对应的 SDK 调用形式：

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

## 当前支持任务

| 任务 | 数据集 | 题数 | 打分方式 |
| --- | --- | --- | --- |
| `humaneval` | HumanEval | 164 | 执行生成代码并跑单元测试 |
| `humanevalplus` | HumanEval+ | 164 | 同上，含扩展测试与 numpy 兼容垫片 |
| `mbpp` | MBPP | 974 | 执行生成代码并跑断言测试 |
| `gsm` | GSM8K | 1319 | 最终数值答案精确匹配 |
| `math500` | MATH-500 | 500 | 归一化后匹配 `\boxed{}` 内的答案 |
| `gpqa` | GPQA-Diamond | 198 | 从 `\boxed{}` 提取多选字母 |

## 输出产物

每次运行只写入一个自包含的 Markdown 报告：

```text
results/<model_name>/<task>_report.md
```

报告包含：

- **Overview** — 任务、模型、数据集、并发数、思考模式
- **Metrics** — 通过率、总用时、吞吐、prompt/completion/total tokens
- **Status counts** — 各状态（通过 / 失败 / 报错等）计数
- **Results** — 逐条结果表：状态、用时、tokens、详情列

## 开发

所有工具均在 `pyproject.toml` 中配置，并通过 `uv` 运行：

```bash
uv run ruff check .    # 代码检查
uv run ruff format .   # 自动格式化
uv run mypy            # 类型检查
uv run pytest          # 运行测试
```
