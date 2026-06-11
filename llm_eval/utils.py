"""Small general-purpose helpers: config parsing, file loading, formatting."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

BOOL_TRUE = {"1", "true", "yes", "y", "on", "enable", "enabled"}
BOOL_FALSE = {"0", "false", "no", "n", "off", "disable", "disabled"}
ENV_PATTERN = re.compile(r"\$\{([^}:]+)(?::-([^}]*))?\}")


def repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for index, raw_line in enumerate(handle):
            line = raw_line.strip()
            if not line:
                continue
            record = json.loads(line)
            record.setdefault("_row_index", index)
            records.append(record)
    return records


def resolve_env_placeholders(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: resolve_env_placeholders(item) for key, item in value.items()}
    if isinstance(value, list):
        return [resolve_env_placeholders(item) for item in value]
    if not isinstance(value, str):
        return value

    def replacer(match: re.Match[str]) -> str:
        key = match.group(1).strip()
        default = match.group(2)
        return os.getenv(key, default if default is not None else match.group(0))

    return ENV_PATTERN.sub(replacer, value)


def parse_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    lowered = str(value).strip().lower()
    if lowered in BOOL_TRUE:
        return True
    if lowered in BOOL_FALSE:
        return False
    return default


def format_seconds(seconds: float) -> str:
    minutes, second = divmod(int(seconds), 60)
    hour, minute = divmod(minutes, 60)
    return f"{hour:02d}:{minute:02d}:{second:02d}"


def slugify(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "-", value.strip())
    cleaned = cleaned.strip(".-")
    return cleaned or "unknown"


def natural_sort_key(value: str) -> list[Any]:
    return [int(part) if part.isdigit() else part for part in re.split(r"(\d+)", value)]
