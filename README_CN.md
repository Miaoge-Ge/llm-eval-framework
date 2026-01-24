# LLM Evaluation Framework (LLM 评测框架)

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/)

轻量级、配置驱动且易于扩展的大语言模型（LLM）评测框架。专为高效和易用性设计，支持多模型并发评测、精确的速率限制以及详尽的性能指标分析。

[English Documentation](README.md)

## ✨ 核心特性

- **多模型与多供应商支持**：无缝切换 OpenAI、DeepSeek、ZhipuAI (GLM) 以及其他兼容 OpenAI 接口的 API。
- **高并发执行**：基于多线程的高性能评测引擎，支持自定义并发数。
- **智能速率限制**：内置令牌桶（Token Bucket）算法，支持精确的 RPM（每分钟请求数）和 TPM（每分钟 Token 数）控制。
- **健壮的错误处理**：自动重试机制、智能错误解析，以及对致命 API 错误（如认证失败、超限）的优雅处理与退出。
- **丰富的指标体系**：
  - **准确率 (Accuracy/Pass@1)**
  - **吞吐量 (Throughput - Tokens/sec)**
  - **延迟 (Latency - 平均任务耗时)**
  - **日志分离**：结果数据 (`results.tsv`) 与执行日志 (`execution.log`) 分离，便于分析。
- **可扩展架构**：通过继承基类，轻松添加新的评测任务（如代码生成、逻辑推理等）。
- **分层配置系统**：`registry.yaml`（静态资源）+ `settings.yaml`（运行时覆盖）的双层配置架构。

## 🚀 支持的任务

- **代码生成 (Code Generation)**:
  - [HumanEval](https://github.com/openai/human-eval)
  - [MBPP](https://github.com/google-research/google-research/tree/master/mbpp)
- **数学推理 (Mathematical Reasoning)**:
  - [GSM8K](https://github.com/openai/grade-school-math)

## 📦 安装指南

1. **克隆仓库**
   ```bash
   git clone https://github.com/your-username/llm-eval-framework.git
   cd llm-eval-framework
   ```

2. **安装依赖**
   ```bash
   pip install -r requirements.txt
   ```

## ⚙️ 配置指南

本框架采用双层配置系统：

1.  **`registry.yaml`**: 定义可用的供应商（Providers）、模型（Models）和数据集（Datasets）。
2.  **`settings.yaml`**: 控制运行时行为，并可覆盖模型参数。

### 1. 注册资源 (`registry.yaml`)

在此文件中定义您的 API Key 和模型端点。

```yaml
providers:
  deepseek:
    api_key: "YOUR_DEEPSEEK_KEY"
    base_url: "https://api.deepseek.com"

models:
  deepseek-chat:
    provider: "deepseek"
    model_name: "deepseek-chat"

datasets:
  humaneval: "./dataset/HumanEval.jsonl"
```

### 2. 运行时配置 (`settings.yaml`)

选择要运行的任务和模型，并调整性能参数。

```yaml
# 选择任务和模型
task: "humaneval"
selected_model: "deepseek-chat"

# 执行设置
workers: 10          # 并发线程数
pass_k: 1            # Pass@k 指标 (默认: 1)

# 运行时参数覆盖 (可选)
temperature: 0.0     # 覆盖模型的 temperature
rpm_limit: 60        # 每分钟最大请求数
tpm_limit: 100000    # 每分钟最大 Token 数
```

## ▶️ 使用方法

运行评测脚本：

```bash
python run_eval.py
```

### 输出结果

评测结果将保存在 `model_test/<model_name>/<task_name>_<timestamp>/` 目录下：

- **`results.tsv`**: 制表符分隔的任务结果文件（包含任务ID、状态、耗时、Token数），易于 Excel/Pandas 处理。
- **`execution.log`**: 完整的执行日志，包含详细的错误信息、警告和最终摘要。

**控制台摘要示例:**
```text
==================================================
Evaluation Summary
--------------------
Tasks Total: 164
Tasks Processed: 164
Passed: 100
Failed: 64
API Errors: 0
Accuracy: 60.98%

Performance Metrics
--------------------
Wall Clock Time: 01:10:05
Throughput: 1500.5 tokens/sec
Total Tokens: 125000

Results saved to: model_test/deepseek-chat/humaneval_20240101_120000
==================================================
```

## 🛠️ 添加新任务

只需继承 `CodeGenerationTask` 或 `ReasoningTask` 即可轻松扩展。

```python
from framework.core import CodeGenerationTask, TaskRegistry

@TaskRegistry.register("my_custom_task")
class MyTask(CodeGenerationTask):
    def process_item(self, item, llm_client):
        # 在此实现您的评测逻辑
        pass
```

## 📄 许可证

本项目基于 MIT 许可证开源。
