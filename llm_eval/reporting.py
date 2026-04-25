from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .settings import FrameworkConfig
from .utils import slugify


@dataclass(frozen=True)
class RunPaths:
    root: Path
    results_jsonl: Path
    summary_json: Path
    config_json: Path
    report_md: Path


class ResultWriter:
    def __init__(self, config: FrameworkConfig) -> None:
        root = config.run.output_dir
        root.mkdir(parents=True, exist_ok=True)
        self.paths = RunPaths(
            root=root,
            results_jsonl=root / f"{slugify(config.run.task)}_results.jsonl",
            summary_json=root / f"{slugify(config.run.task)}_summary.json",
            config_json=root / "resolved_config.json",
            report_md=root / f"{slugify(config.run.task)}_report.md",
        )
        self.result_handle = self.paths.results_jsonl.open("w", encoding="utf-8")
        self.paths.config_json.write_text(
            json.dumps(config.to_public_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def close(self) -> None:
        self.result_handle.close()

    def write_result(self, payload: dict[str, Any]) -> None:
        self.result_handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
        self.result_handle.flush()

    def write_summary(self, payload: dict[str, Any]) -> None:
        self.paths.summary_json.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def write_report(self, content: str) -> None:
        self.paths.report_md.write_text(content, encoding="utf-8")

    def __enter__(self) -> "ResultWriter":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()


def dataclass_to_dict(value: Any) -> dict[str, Any]:
    return asdict(value)
