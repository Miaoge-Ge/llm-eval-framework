from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .runner import EvaluationRunner, RunSummary
from .settings import load_framework_config
from .tasks import DEFAULT_TASK_NAME, TASK_REGISTRY


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="llm-eval", description="Engineering-first LLM evaluation framework")
    subparsers = parser.add_subparsers(dest="command")

    run_parser = subparsers.add_parser("run", help="Run an evaluation")
    run_parser.add_argument("--config", default="configs/model.yaml", help="Path to model config YAML")
    run_parser.add_argument("--task", default=DEFAULT_TASK_NAME, help="Task name")
    run_parser.add_argument("--list-tasks", action="store_true", help="List available tasks and exit")

    return parser


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0].startswith("-"):
        argv = ["run", *argv]
    parser = build_parser()
    args = parser.parse_args(argv)

    if getattr(args, "list_tasks", False):
        print("Available tasks:")
        for name in sorted(TASK_REGISTRY):
            print(f"  - {name}")
        return 0

    try:
        config = load_framework_config(
            model_config_path=args.config,
            task=args.task,
        )
    except (ValueError, FileNotFoundError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    summary, writer = EvaluationRunner(config).run()
    print(_render_summary(summary, writer.paths.report_md))
    return 0


def _render_summary(summary: RunSummary, report_path: Path) -> str:
    lines = [
        "=" * 60,
        "LLM Evaluation Summary",
        "=" * 60,
        f"Task: {summary.task}",
        f"Model: {summary.model}",
        f"Cases: {summary.completed_cases}/{summary.total_cases}",
        f"Pass rate: {summary.pass_rate:.2%}",
        f"Wall clock: {summary.wall_clock_human}",
        f"Avg case: {summary.average_case_seconds:.2f}s",
        f"Throughput: {summary.throughput_tokens_per_second:.1f} tokens/s",
        f"Statuses: {summary.status_counts}",
        f"Tokens: {summary.token_usage}",
        f"Report: {report_path}",
        "=" * 60,
    ]
    return "\n".join(lines)
