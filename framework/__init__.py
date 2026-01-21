from .core import Runner, EvalConfig, BaseTask
from .config import ConfigManager
from .registry import TaskRegistry
from .utils import Logger

# Auto-discover evaluators when framework is imported
import framework.evaluators

__all__ = [
    "Runner",
    "EvalConfig",
    "BaseTask",
    "ConfigManager",
    "TaskRegistry",
    "Logger"
]
