"""Subprocess execution of model-generated Python code.

Note: generated code runs unsandboxed, with the privileges of the current
user. Only evaluate models and datasets you trust.
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

from .status import Status


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
            return Status.PASSED, result.stdout.strip()
        stderr = (result.stderr or result.stdout or "Unknown execution failure").strip()
        return Status.FAILED, stderr.replace("\n", " | ")
    except subprocess.TimeoutExpired:
        return Status.TIMEOUT, f"Execution exceeded {timeout_seconds}s"
    finally:
        if temp_path and temp_path.exists():
            temp_path.unlink(missing_ok=True)


def run_python_with_stdin(code: str, stdin_text: str, timeout_seconds: int) -> tuple[str, str, str]:
    """Run code feeding stdin_text on standard input.

    Returns (status, stdout, detail) where status is OK / RUNTIME_ERROR / TIMEOUT.
    """
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False, encoding="utf-8") as handle:
            handle.write(code)
            temp_path = Path(handle.name)
        result = subprocess.run(
            [sys.executable, str(temp_path)],
            input=stdin_text,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
        if result.returncode != 0:
            return "RUNTIME_ERROR", result.stdout, (result.stderr or "Unknown execution failure").strip()
        return "OK", result.stdout, ""
    except subprocess.TimeoutExpired:
        return "TIMEOUT", "", f"Execution exceeded {timeout_seconds}s"
    finally:
        if temp_path and temp_path.exists():
            temp_path.unlink(missing_ok=True)


def normalize_stdout(text: str) -> str:
    """Canonicalize program output for competitive-programming comparison:
    strip trailing whitespace per line and drop trailing blank lines."""
    lines = [line.rstrip() for line in text.strip().split("\n")]
    while lines and not lines[-1]:
        lines.pop()
    return "\n".join(lines)
