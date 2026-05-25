from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .settings import FrameworkConfig
from .utils import slugify


@dataclass(frozen=True)
class RunPaths:
    root: Path
    report_md: Path


class ResultWriter:
    def __init__(self, config: FrameworkConfig) -> None:
        root = config.run.output_dir
        root.mkdir(parents=True, exist_ok=True)
        self.paths = RunPaths(
            root=root,
            report_md=root / f"{slugify(config.run.task)}_report.md",
        )

    def write_report(self, content: str) -> None:
        self.paths.report_md.write_text(content, encoding="utf-8")

    def __enter__(self) -> ResultWriter:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        return None
