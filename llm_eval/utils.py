from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

BOOL_TRUE = {"1", "true", "yes", "y", "on", "enable", "enabled"}
BOOL_FALSE = {"0", "false", "no", "n", "off", "disable", "disabled"}
ENV_PATTERN = re.compile(r"\$\{([^}:]+)(?::-([^}]*))?\}")
PYTHON_BLOCK_PATTERN = re.compile(r"```(?:python|py)?\s*(.*?)```", re.IGNORECASE | re.DOTALL)


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


def extract_python_code(text: str) -> str:
    if not text:
        return ""
    match = PYTHON_BLOCK_PATTERN.search(text)
    extracted = match.group(1) if match else text
    return dedent_code(extracted.strip())


def dedent_code(code: str) -> str:
    lines = code.splitlines()
    min_indent: int | None = None
    for line in lines:
        stripped = line.lstrip()
        if not stripped:
            continue
        indent = len(line) - len(stripped)
        min_indent = indent if min_indent is None else min(min_indent, indent)
    if not min_indent:
        return code.strip()
    return "\n".join(line[min_indent:] if line.strip() else "" for line in lines).strip()


def indent_block(code: str, spaces: int) -> str:
    prefix = " " * spaces
    return "\n".join(prefix + line if line.strip() else "" for line in code.splitlines())


def last_numeric_token(text: str) -> str | None:
    if not text:
        return None
    explicit = re.search(r"####\s*(-?[\d,]+(?:\.\d+)?)", text)
    if explicit:
        return explicit.group(1).replace(",", "")
    matches = re.findall(r"-?[\d,]+(?:\.\d+)?", text)
    if not matches:
        return None
    return matches[-1].replace(",", "")


def extract_last_boxed(text: str) -> str | None:
    if not text:
        return None
    needle = "\\boxed"
    start = text.rfind(needle)
    if start == -1:
        return None
    index = start + len(needle)
    while index < len(text) and text[index] != "{":
        if text[index] == " ":
            index += 1
            continue
        return None
    if index >= len(text):
        return None
    depth = 0
    content_start = index + 1
    for position in range(index, len(text)):
        char = text[position]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[content_start:position]
    return None


def normalize_math_answer(text: str | None) -> str | None:
    if text is None:
        return None
    value = text.strip()
    if not value:
        return None
    for wrapper in ("\\left", "\\right", "\\,", "\\!", "\\ ", "\\;", "\\:", "$", "\\$"):
        value = value.replace(wrapper, "")
    value = re.sub(r"\\text\s*\{([^}]*)\}", r"\1", value)
    value = re.sub(r"\\mbox\s*\{([^}]*)\}", r"\1", value)
    value = value.replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac")
    value = value.replace("\\%", "").replace("%", "")
    value = value.replace("^{\\circ}", "").replace("^\\circ", "").replace("{}^\\circ", "")
    value = value.replace(" ", "")
    if value.endswith("."):
        value = value[:-1]
    value = value.replace("dollars", "").replace("\\cdot", "*")
    return value or None


def extract_choice_letter(text: str) -> str | None:
    if not text:
        return None
    boxed = extract_last_boxed(text)
    if boxed:
        match = re.search(r"[A-D]", boxed.upper())
        if match:
            return match.group(0)
    patterns = [
        r"answer\s*(?:is|:)?\s*\(?([A-D])\)?",
        r"\b([A-D])\b\s*$",
        r"\(([A-D])\)",
    ]
    upper = text.strip()
    for pattern in patterns:
        match = re.search(pattern, upper, re.IGNORECASE | re.MULTILINE)
        if match:
            return match.group(1).upper()
    return None


def execute_python(code: str, timeout_seconds: int) -> tuple[str, str]:
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False, encoding="utf-8") as handle:
            handle.write(code)
            temp_path = Path(handle.name)
        result = subprocess.run(
            [sys.executable, str(temp_path)],
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
        if result.returncode == 0:
            return "PASSED", result.stdout.strip()
        stderr = (result.stderr or result.stdout or "Unknown execution failure").strip()
        return "FAILED", stderr.replace("\n", " | ")
    except subprocess.TimeoutExpired:
        return "TIMEOUT", f"Execution exceeded {timeout_seconds}s"
    finally:
        if temp_path and temp_path.exists():
            temp_path.unlink(missing_ok=True)
