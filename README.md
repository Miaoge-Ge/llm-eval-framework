# LLM Evaluation Framework

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/)

A lightweight, configuration-driven, and extensible framework for evaluating Large Language Models (LLMs). Designed for efficiency and ease of use, it supports multi-provider concurrent evaluation, rate limiting, and detailed performance metrics.

[中文文档](README_CN.md)

## ✨ Key Features

- **Multi-Model & Multi-Provider Support**: Seamlessly switch between OpenAI, DeepSeek, ZhipuAI (GLM), and other OpenAI-compatible APIs.
- **Concurrent Execution**: High-performance multi-threaded evaluation with configurable worker count.
- **Smart Rate Limiting**: Built-in Token Bucket algorithm for precise RPM (Requests Per Minute) and TPM (Tokens Per Minute) control.
- **Robust Error Handling**: Automatic retry mechanism, intelligent error parsing, and graceful handling of critical API failures (e.g., Auth Error, Rate Limit).
- **Rich Metrics**:
  - **Accuracy (Pass@1)**
  - **Throughput (Tokens/sec)**
  - **Latency (Avg duration per task)**
  - **Detailed Logs**: Separates clean result metrics (`results.tsv`) from detailed execution logs (`execution.log`).
- **Extensible Architecture**: Easily add new tasks (Code Generation, Reasoning, etc.) by inheriting from base classes.
- **Configuration Hierarchy**: Flexible `registry.yaml` (static resources) + `settings.yaml` (runtime overrides) architecture.

## 🚀 Supported Tasks

- **Code Generation**:
  - [HumanEval](https://github.com/openai/human-eval)
  - [MBPP](https://github.com/google-research/google-research/tree/master/mbpp)
- **Mathematical Reasoning**:
  - [GSM8K](https://github.com/openai/grade-school-math)

## 📦 Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-username/llm-eval-framework.git
   cd llm-eval-framework
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## ⚙️ Configuration

The framework uses a two-layer configuration system:

1.  **`registry.yaml`**: Defines available Providers, Models, and Datasets.
2.  **`settings.yaml`**: Controls runtime behavior and overrides model parameters.

### 1. Setup Registry (`registry.yaml`)

Define your API keys and model endpoints here.

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

### 2. Configure Runtime (`settings.yaml`)

Select which task and model to run, and tune performance parameters.

```yaml
# Select Task and Model
task: "humaneval"
selected_model: "deepseek-chat"

# Execution Settings
workers: 10          # Number of concurrent threads
pass_k: 1            # Pass@k metric (Default: 1)

# Runtime Overrides (Optional)
temperature: 0.0     # Override model temperature
rpm_limit: 60        # Max requests per minute
tpm_limit: 100000    # Max tokens per minute
```

## ▶️ Usage

Run the evaluation script:

```bash
python run_eval.py
```

### Output

Results are saved in `model_test/<model_name>/<task_name>_<timestamp>/`:

- **`results.tsv`**: Tab-separated metrics for each task (Task ID, Status, Duration, Tokens).
- **`execution.log`**: Full execution log including errors, warnings, and summary.

**Console Summary Example:**
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

## 🛠️ Adding New Tasks

Inherit from `CodeGenerationTask` or `ReasoningTask` to add custom evaluations.

```python
from framework.core import CodeGenerationTask, TaskRegistry

@TaskRegistry.register("my_custom_task")
class MyTask(CodeGenerationTask):
    def process_item(self, item, llm_client):
        # Your evaluation logic here
        pass
```

## 📄 License

This project is licensed under the MIT License.
