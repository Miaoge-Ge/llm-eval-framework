from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml

from .utils import parse_bool, repo_root, resolve_env_placeholders

DEFAULT_DATASET_PATHS = {
    "humaneval": Path("datasets/HumanEval.jsonl"),
    "humanevalplus": Path("datasets/HumanEvalPlus.jsonl"),
    "mbpp": Path("datasets/mbpp.jsonl"),
    "gsm": Path("datasets/gsm.jsonl"),
}


@dataclass(frozen=True)
class ModelConfig:
    api_key: str
    base_url: str
    model_name: str
    timeout_seconds: int = 120


@dataclass(frozen=True)
class DatasetConfig:
    name: str
    path: Path


@dataclass(frozen=True)
class RunConfig:
    task: str
    workers: int = 10
    output_dir: Path = Path("results")
    execution_timeout_seconds: int = 20
    thinking_enabled: bool = False
    reasoning_effort: str | None = None

    @property
    def reasoning_display(self) -> str:
        return self.reasoning_effort or "default" if self.thinking_enabled else "disabled"


@dataclass(frozen=True)
class FrameworkConfig:
    model: ModelConfig
    dataset: DatasetConfig
    run: RunConfig
    model_config_path: Path

    def to_public_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["dataset"]["path"] = str(self.dataset.path)
        payload["run"]["output_dir"] = str(self.run.output_dir)
        payload["model"]["api_key"] = "***"
        payload["model_config_path"] = str(self.model_config_path)
        return payload


def load_framework_config(
    model_config_path: str | Path | None = None,
    task: str | None = None,
) -> FrameworkConfig:
    root = repo_root()
    model_file = _resolve_file_path(model_config_path, root / "configs" / "model.yaml")
    model_data = _load_yaml(model_file)

    task_name = _require_text(task or "humaneval", "task")
    dataset_path = _resolve_dataset_path(task_name)
    model = ModelConfig(
        api_key=_require_text(model_data.get("api_key"), "api_key"),
        base_url=_require_text(model_data.get("base_url"), "base_url"),
        model_name=_require_text(model_data.get("model_name"), "model_name"),
        timeout_seconds=int(model_data.get("timeout_seconds", 120)),
    )

    run = RunConfig(
        task=task_name,
        workers=max(1, int(model_data.get("workers", 10))),
        output_dir=(root / "results" / _require_text(model.model_name, "model_name")).resolve(),
        execution_timeout_seconds=max(1, int(model_data.get("execution_timeout_seconds", 20))),
        thinking_enabled=parse_bool(model_data.get("thinking_enabled", False), default=False),
        reasoning_effort=_optional_text(model_data.get("reasoning_effort")),
    )

    return FrameworkConfig(
        model=model,
        dataset=DatasetConfig(name=task_name, path=dataset_path),
        run=run,
        model_config_path=model_file,
    )


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        parsed = yaml.safe_load(handle) or {}
    if not isinstance(parsed, dict):
        raise ValueError(f"Expected top-level mapping in {path}")
    return resolve_env_placeholders(parsed)


def _resolve_file_path(path_value: str | Path | None, fallback: Path) -> Path:
    path = Path(path_value) if path_value else fallback
    return path if path.is_absolute() else repo_root() / path


def _resolve_dataset_path(task_name: str) -> Path:
    relative_path = DEFAULT_DATASET_PATHS.get(task_name)
    if not relative_path:
        available = ", ".join(sorted(DEFAULT_DATASET_PATHS))
        raise ValueError(f"Unsupported task '{task_name}'. Available tasks: {available}")
    return (repo_root() / relative_path).resolve()


def _require_text(value: Any, field_name: str) -> str:
    text = str(value).strip() if value is not None else ""
    if not text:
        raise ValueError(f"Missing required configuration field: {field_name}")
    if text.startswith("${") and text.endswith("}"):
        raise ValueError(f"Unresolved environment variable for configuration field: {field_name}")
    return text


def _optional_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None
