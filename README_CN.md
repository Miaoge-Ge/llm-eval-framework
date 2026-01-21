# LLM 评测框架

[English Documentation](README.md)

一个轻量级、可扩展且基于配置驱动的大语言模型（LLM）评测框架，专注于代码生成和逻辑推理任务。

## 功能特性

- **模块化配置**: 分离用户设置 (`settings.yaml`) 和资源定义 (`registry.yaml`)，管理更高效。
- **多厂商支持**: 轻松切换不同的 API 提供商（如 OpenAI, DeepSeek, 智谱AI）。
- **可扩展架构**: 支持自动发现机制，只需少量代码即可添加新的评测任务和数据集。
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

本框架使用两个配置文件：

1. **`registry.yaml`**: 定义可用的资源（API厂商、模型、数据集）。
   ```yaml
   providers:
     deepseek:
       api_key: "sk-..."
       base_url: "https://api.deepseek.com"
   
   models:
     deepseek-chat:
       provider: "deepseek"
       model_name: "deepseek-chat"
       temperature: 0.0
   ```

2. **`settings.yaml`**: 控制当前的运行参数。
   ```yaml
   task: "mbpp"
   selected_model: "deepseek-chat"
   workers: 5
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
├── settings.yaml        # 用户运行设置
├── registry.yaml        # 资源定义 (厂商, 模型, 数据集)
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
