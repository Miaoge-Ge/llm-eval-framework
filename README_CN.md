# LLM Eval Framework

[English Documentation](README.md)

一个面向工程使用的 LLM 评测框架。默认用 `uv` 管理环境，配置极简（单个模型配置文件），每次运行只产出一个自包含的 Markdown 报告，支持任意 OpenAI-compatible 接口。

## 特点

- 默认使用 `uv` 管理依赖与运行环境
- 可安装的 Python 包（`hatchling` 构建后端），提供 `llm-eval` 命令行入口
- 单一模型配置文件，密钥用 `${ENV_VAR}` 占位符注入
- 内置 10 个评测任务，涵盖代码生成、数学推理、多选知识题三大类
- OpenAI-compatible 客户端，支持流式输出、自动重试与 token 用量统计
- 每次运行只产出一个自包含 Markdown 报告（配置 + 指标 + 逐条结果）
- 知识题任务自动生成分学科 / 分领域准确率明细
- 支持 `thinking` 与可配置的 `reasoning_effort`
- 评测过程中显示实时进度条（通过数 / 失败数 / 最近 HTTP 状态）
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
  ifeval/              # vendored 的 Google IFEval 校验器 + 打分胶水代码
configs/
  model.example.yaml   # 已提交的模板（不含密钥）
datasets/
  HumanEval.jsonl
  HumanEvalPlus.jsonl
  MBPP.jsonl
  MBPPPlus.jsonl
  GSM8K.jsonl
  AIME2025.jsonl
  GPQA.jsonl
  IFEval.jsonl
  LiveCodeBench.jsonl
scripts/
  fetch_datasets.py    # 从 Hugging Face 重新下载/生成数据集
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

把已提交的模板复制为本地配置（`configs/` 目录除示例外都被 git 忽略，因此真实密钥不会被提交）：

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
# 运行某个评测任务
uv run llm-eval run --config configs/model.yaml --task gpqa

# 等价的模块形式
uv run python -m llm_eval run --config configs/model.yaml --task gsm

# 列出所有可用任务
uv run llm-eval --list-tasks
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

### 代码生成

| 任务 | 数据集 | 题数 | 打分方式 |
| --- | --- | --- | --- |
| `humaneval` | HumanEval | 164 | 执行生成代码并跑单元测试 |
| `humanevalplus` | HumanEval+ | 164 | 同上，含扩展测试与 numpy 兼容垫片 |
| `mbpp` | MBPP | 974 | 执行生成代码并跑断言测试 |
| `mbppplus` | MBPP+ | 378 | EvalPlus 扩展测试，含 numpy 兼容垫片 |
| `livecodebench` | LiveCodeBench (lite) | 219 | 作为 stdin/stdout 程序运行并比对 public 测试用例 |

### 数学推理

| 任务 | 数据集 | 题数 | 打分方式 |
| --- | --- | --- | --- |
| `gsm` | GSM8K | 1319 | 精确匹配 `#### <数字>` 末尾答案 |
| `aime2025` | AIME 2025 (I+II) | 30 | 对 `\boxed{}` 内答案做整数匹配 |

### 多选知识题

多选任务要求模型输出 `\boxed{A/B/C/D}`，按字母精确匹配打分。报告中会自动生成分领域准确率明细。

| 任务 | 数据集 | 题数 | 覆盖范围 |
| --- | --- | --- | --- |
| `gpqa` | GPQA-Diamond | 198 | 博士级科学题（物理、化学、生物） |

### 指令遵循

| 任务 | 数据集 | 题数 | 打分方式 |
| --- | --- | --- | --- |
| `ifeval` | IFEval | 541 | 对每条指令做程序化校验（prompt 级 strict 准确率） |

`livecodebench` 仅保留 stdin/stdout 题（AtCoder/Codeforces），按明文 public 测试用例打分。`ifeval` 在 `llm_eval/ifeval/` 下 vendored 了 Google 的指令校验器，依赖 `langdetect`。可用 `uv run --with datasets python scripts/fetch_datasets.py <name>` 重新生成任意数据集。

## 输出产物

每次运行只写入一个自包含的 Markdown 报告：

```text
results/<model_name>/<task>_report.md
```

报告包含：

- **Overview** — 任务、模型、数据集、并发数、思考模式
- **Metrics** — 通过率、总用时、吞吐、prompt/completion/total tokens
- **Status counts** — 各状态（通过 / 失败 / 报错等）计数
- **Accuracy by domain** — 分学科准确率明细（仅知识题任务）
- **Results** — 逐条结果表：状态、用时、tokens、详情列

## 开发

所有工具均在 `pyproject.toml` 中配置，并通过 `uv` 运行：

```bash
uv run ruff check .    # 代码检查
uv run ruff format .   # 自动格式化
uv run mypy            # 类型检查
uv run pytest          # 运行测试
```
