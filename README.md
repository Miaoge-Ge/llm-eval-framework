# LLM Evaluation Framework

[中文文档](README_CN.md)

A lightweight, extensible, and configuration-driven framework for evaluating Large Language Models (LLMs) on code generation and reasoning tasks.

## Features

- **Configuration-Driven**: Manage models, providers, and tasks via a simple `config.yaml`.
- **Provider Support**: Easily switch between different API providers (e.g., OpenAI, DeepSeek, ZhipuAI).
- **Extensible Architecture**: Add new tasks and datasets with minimal code changes.
- **Concurrent Execution**: Fast evaluation with multi-threading support.
- **Detailed Logging**: Comprehensive logs for debugging and analysis.

## Supported Tasks

- **MBPP** (Mostly Basic Python Programming)
- **HumanEval** & **HumanEval+**
- **GSM8K** (Grade School Math)

## Quick Start

### 1. Installation

Clone the repository and install the dependencies:

```bash
pip install -r requirements.txt
# Note: You need `pyyaml`, `openai`, `tqdm`
```

### 2. Configuration

Copy the example configuration file:

```bash
cp config.example.yaml config.yaml
```

Edit `config.yaml` to add your API keys and model preferences:

```yaml
# Global Settings
task: "mbpp"               # Task to run
selected_model: "my-model" # Model profile to use

# Providers
providers:
  deepseek:
    api_key: "sk-..."
    base_url: "https://api.deepseek.com"

# Models
models:
  my-model:
    provider: "deepseek"
    model_name: "deepseek-chat"
    temperature: 0.0
```

### 3. Run Evaluation

Simply run the script:

```bash
python run_eval.py
```

The results will be saved in the `model_test/` directory.

## Project Structure

```
.
├── config.yaml          # Main configuration file (ignored by git)
├── run_eval.py          # Entry point
├── framework/           # Core framework code
│   ├── core.py          # Evaluation engine
│   ├── config.py        # Configuration manager
│   ├── registry.py      # Task registry
│   └── evaluators/      # Task implementations
└── dataset/             # Evaluation datasets
```

## Adding New Tasks

1. Create a new file in `framework/evaluators/`.
2. Inherit from `BaseTask`.
3. Register the task using `@TaskRegistry.register("task_name")`.

```python
@TaskRegistry.register("my_task")
class MyTask(BaseTask):
    ...
```

## License

MIT
