"""Run persistence and report rendering.

Each run gets its own directory under ``results/<model>/<task>/<timestamp>/``
containing:

- ``results.jsonl`` — one line per case, written incrementally so a crashed
  run keeps everything graded so far
- ``config.json`` — snapshot of the effective configuration (API key masked)
- ``report.md`` — the final human-readable report
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TextIO

from .settings import FrameworkConfig
from .status import GRADED_STATUSES, Status
from .utils import natural_sort_key, slugify


@dataclass(frozen=True)
class RunSummary:
    task: str
    model: str
    total_cases: int
    completed_cases: int
    pass_rate: float
    wall_clock_seconds: float
    wall_clock_human: str
    average_case_seconds: float
    throughput_tokens_per_second: float
    status_counts: dict[str, int]
    token_usage: dict[str, int]


@dataclass(frozen=True)
class RunPaths:
    root: Path
    results_jsonl: Path
    report_md: Path
    config_json: Path


class ResultWriter:
    def __init__(self, config: FrameworkConfig) -> None:
        self.config = config
        root = _allocate_run_dir(config)
        self.paths = RunPaths(
            root=root,
            results_jsonl=root / "results.jsonl",
            report_md=root / "report.md",
            config_json=root / "config.json",
        )
        self._results_handle: TextIO | None = None

    def __enter__(self) -> ResultWriter:
        self.paths.config_json.write_text(
            json.dumps(self.config.to_public_dict(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        self._results_handle = self.paths.results_jsonl.open("w", encoding="utf-8")
        return self

    def write_result(self, payload: dict[str, Any]) -> None:
        if self._results_handle is None:
            raise RuntimeError("ResultWriter must be entered before writing results")
        self._results_handle.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")
        self._results_handle.flush()

    def write_report(self, content: str) -> None:
        self.paths.report_md.write_text(content, encoding="utf-8")

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if self._results_handle is not None:
            self._results_handle.close()
            self._results_handle = None


def _allocate_run_dir(config: FrameworkConfig) -> Path:
    base = config.run.output_dir / slugify(config.run.task)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    candidate = base / stamp
    counter = 2
    while candidate.exists():
        candidate = base / f"{stamp}-{counter}"
        counter += 1
    candidate.mkdir(parents=True)
    return candidate


def render_markdown_report(
    summary: RunSummary,
    config: FrameworkConfig,
    results: list[dict[str, Any]],
) -> str:
    usage = summary.token_usage
    generated_at = time.strftime("%Y-%m-%d %H:%M:%S")
    graded = sum(summary.status_counts.get(status, 0) for status in GRADED_STATUSES)
    lines = [
        f"# Evaluation Report - {summary.task}",
        "",
        f"_Generated at {generated_at}_",
        "",
        "## Overview",
        "",
        "| Field | Value |",
        "| --- | --- |",
        f"| Task | `{summary.task}` |",
        f"| Model | `{summary.model}` |",
        f"| Dataset | `{config.dataset.path}` |",
        f"| Workers | `{config.run.workers}` |",
        f"| Thinking enabled | `{config.run.thinking_enabled}` |",
        f"| Reasoning effort | `{config.run.reasoning_display}` |",
        "",
        "## Metrics",
        "",
        "| Metric | Value |",
        "| --- | --- |",
        f"| Pass rate | **{summary.pass_rate:.2%}** |",
        f"| Total cases | {summary.total_cases} |",
        f"| Completed cases | {summary.completed_cases} |",
        f"| Graded cases | {graded} |",
        f"| Wall clock | {summary.wall_clock_human} |",
        f"| Average case time | {summary.average_case_seconds:.2f}s |",
        f"| Throughput | {summary.throughput_tokens_per_second:.1f} tokens/s |",
        f"| Prompt tokens | {usage.get('prompt_tokens', 0):,} |",
        f"| Completion tokens | {usage.get('completion_tokens', 0):,} |",
        f"| Total tokens | {usage.get('total_tokens', 0):,} |",
        "",
        "### Status counts",
        "",
        "| Status | Count |",
        "| --- | --- |",
    ]
    for status, count in sorted(summary.status_counts.items()):
        lines.append(f"| {status} | {count} |")

    domain_rows = _domain_breakdown(results)
    if domain_rows:
        lines.extend(
            [
                "",
                "### Accuracy by domain",
                "",
                "| Domain | Passed | Total | Pass rate |",
                "| --- | --- | --- | --- |",
            ]
        )
        for domain, passed, total in domain_rows:
            rate = passed / total if total else 0.0
            lines.append(f"| {domain} | {passed} | {total} | {rate:.2%} |")

    lines.extend(
        [
            "",
            "## Results",
            "",
            "| # | Case | Status | Time | Tokens | Detail |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
    )
    ordered = sorted(results, key=lambda r: natural_sort_key(str(r.get("case_id", ""))))
    for index, item in enumerate(ordered, start=1):
        lines.append(
            "| {idx} | {case} | {status} | {time} | {tokens} | {detail} |".format(
                idx=index,
                case=_cell(item.get("case_id", "unknown")),
                status=item.get("status", "unknown"),
                time=item.get("duration_human", "n/a"),
                tokens=item.get("total_tokens", 0),
                detail=_cell(_detail(item)),
            )
        )

    return "\n".join(lines).strip() + "\n"


def _domain_breakdown(results: list[dict[str, Any]]) -> list[tuple[str, int, int]]:
    stats: dict[str, list[int]] = {}
    for item in results:
        domain = (item.get("metadata") or {}).get("domain")
        if not domain:
            continue
        entry = stats.setdefault(domain, [0, 0])
        entry[1] += 1
        if item.get("status") == Status.PASSED:
            entry[0] += 1
    return [(domain, passed, total) for domain, (passed, total) in sorted(stats.items())]


def _shorten(value: str, limit: int = 160) -> str:
    compact = " ".join(str(value).split())
    if len(compact) <= limit:
        return compact
    return compact[: limit - 3] + "..."


def _cell(value: Any) -> str:
    text = _shorten(str(value)).replace("|", "\\|")
    return text or "n/a"


def _detail(item: dict[str, Any]) -> str:
    if item.get("status") == Status.PASSED:
        return "passed"
    error = item.get("error")
    if error:
        return str(error)
    expected = item.get("expected")
    actual = item.get("actual")
    if expected is not None or actual is not None:
        return f"expected {expected} -> got {actual}"
    return "n/a"
