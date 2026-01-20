# LLM 评测框架

[English Documentation](README.md)

一个轻量级、可扩展且基于配置驱动的大语言模型（LLM）评测框架，专注于代码生成和逻辑推理任务。

## 功能特性

- **配置驱动**: 通过简单的 `config.yaml` 管理模型、API厂商和任务。
- **多厂商支持**: 轻松切换不同的 API 提供商（如 OpenAI, DeepSeek, 智谱AI）。
- **可扩展架构**: 只需少量代码即可添加新的评测任务和数据集。
- **并发执行**: 支持多线程并发评测，大幅提升速度。
- **详细日志**: 提供完整的执行日志，便于调试和分析。

## 支持的任务

- **MBPP** (基础 Python 编程任务)
- **HumanEval** & **HumanEval+** (代码生成)
- **GSM8K** (小学数学逻辑推理)

## 快速开始

### 1. 安装

克隆仓库并安装依赖：

```bash
pip install -r requirements.txt
# 依赖库: `pyyaml`, `openai`, `tqdm`
```

### 2. 配置

复制示例配置文件：

```bash
cp config.example.yaml config.yaml
```

编辑 `config.yaml`，填入您的 API Key 和模型配置：

```yaml
# 全局设置
task: "mbpp"               # 要运行的任务
selected_model: "my-model" # 选择使用的模型配置

# 厂商配置 (Providers)
providers:
  deepseek:
    api_key: "sk-..."
    base_url: "https://api.deepseek.com"

# 模型配置 (Models)
models:
  my-model:
    provider: "deepseek"
    model_name: "deepseek-chat"
    temperature: 0.0
```

### 3. 运行评测

直接运行脚本即可：

```bash
python run_eval.py
```

评测结果将保存在 `model_test/` 目录下。

## 项目结构

```
.
├── config.yaml          # 主配置文件 (Git 已忽略)
├── run_eval.py          # 程序入口
├── framework/           # 核心框架代码
│   ├── core.py          # 评测引擎
│   ├── config.py        # 配置管理器
│   ├── registry.py      # 任务注册中心
│   └── evaluators/      # 具体任务实现
└── dataset/             # 评测数据集
```

## 添加新任务

1. 在 `framework/evaluators/` 下创建一个新文件。
2. 继承 `BaseTask` 类。
3. 使用 `@TaskRegistry.register("task_name")` 装饰器注册任务。

```python
@TaskRegistry.register("my_task")
class MyTask(BaseTask):
    ...
```

## 许可证

MIT
