from __future__ import annotations

import concurrent.futures
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tqdm import tqdm

from .clients import OpenAICompatibleClient
from .reporting import ResultWriter
from .settings import FrameworkConfig
from .tasks import TASK_REGISTRY, BaseEvaluationTask, TaskCase, TaskResult
from .utils import format_seconds


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


class EvaluationRunner:
    def __init__(self, config: FrameworkConfig) -> None:
        self.config = config
        self.client = OpenAICompatibleClient(config)

    def build_task(self) -> BaseEvaluationTask:
        task_factory = TASK_REGISTRY.get(self.config.run.task)
        if not task_factory:
            available = ", ".join(sorted(TASK_REGISTRY))
            raise ValueError(f"Unsupported task '{self.config.run.task}'. Available tasks: {available}")
        return task_factory(self.config)

    def run(self) -> tuple[RunSummary, ResultWriter]:
        task = self.build_task()
        cases = task.load_cases()
        if not cases:
            raise ValueError(f"No cases found in dataset: {self.config.dataset.path}")

        totals: Counter[str] = Counter()
        total_duration = 0.0
        total_tokens = 0
        prompt_tokens = 0
        completion_tokens = 0
        last_http_status_code: int | None = None
        started_at = time.time()
        results: list[dict[str, Any]] = []

        print(
            "\n".join(
                [
                    "Starting evaluation...",
                    f"  Task: {self.config.run.task}",
                    f"  Model: {self.config.model.model_name}",
                    f"  Dataset: {self.config.dataset.path}",
                    f"  Workers: {self.config.run.workers}",
                    f"  Thinking: {self.config.run.thinking_enabled}",
                    f"  Reasoning effort: {self.config.run.reasoning_display}",
                    f"  Output: {self.config.run.output_dir}",
                    f"  Total cases: {len(cases)}",
                ]
            ),
            flush=True,
        )

        with ResultWriter(self.config) as writer:
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.config.run.workers) as pool:
                future_to_case = {pool.submit(self._evaluate_case, task, case): case for case in cases}
                with tqdm(total=len(cases), desc=self.config.run.task, unit="case", dynamic_ncols=True) as progress:
                    for future in concurrent.futures.as_completed(future_to_case):
                        result = future.result()
                        payload = result.to_dict()
                        results.append(payload)
                        totals[result.status] += 1
                        total_duration += result.duration_seconds
                        total_tokens += result.total_tokens
                        prompt_tokens += result.prompt_tokens
                        completion_tokens += result.completion_tokens
                        last_http_status_code = result.http_status_code or last_http_status_code

                        progress.update(1)
                        progress.set_postfix_str(
                            "passed={passed} failed={failed} http={http}".format(
                                passed=totals.get("PASSED", 0),
                                failed=totals.get("FAILED", 0),
                                http=last_http_status_code if last_http_status_code is not None else "-",
                            ),
                            refresh=False,
                        )

            wall_clock_seconds = time.time() - started_at
            graded = totals["PASSED"] + totals["FAILED"] + totals["TIMEOUT"]
            pass_rate = (totals["PASSED"] / graded) if graded else 0.0
            summary = RunSummary(
                task=self.config.run.task,
                model=self.config.model.model_name,
                total_cases=len(cases),
                completed_cases=sum(totals.values()),
                pass_rate=pass_rate,
                wall_clock_seconds=wall_clock_seconds,
                wall_clock_human=format_seconds(wall_clock_seconds),
                average_case_seconds=(total_duration / sum(totals.values())) if totals else 0.0,
                throughput_tokens_per_second=(total_tokens / wall_clock_seconds) if wall_clock_seconds else 0.0,
                status_counts=dict(totals),
                token_usage={
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": total_tokens,
                },
            )
            writer.write_report(
                self._build_markdown_report(
                    summary=summary,
                    output_dir=writer.paths.root,
                    results=results,
                )
            )
            return summary, writer

    def _evaluate_case(self, task: BaseEvaluationTask, case: TaskCase) -> TaskResult:
        try:
            return task.evaluate_case(case, self.client)
        except Exception as exc:
            return TaskResult(
                case_id=case.case_id,
                status="INTERNAL_ERROR",
                duration_seconds=0.0,
                error=str(exc),
            )

    def _build_markdown_report(
        self,
        summary: RunSummary,
        output_dir: Path,
        results: list[dict[str, Any]],
    ) -> str:
        usage = summary.token_usage
        generated_at = time.strftime("%Y-%m-%d %H:%M:%S")
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
            f"| Dataset | `{self.config.dataset.path}` |",
            f"| Workers | `{self.config.run.workers}` |",
            f"| Thinking enabled | `{self.config.run.thinking_enabled}` |",
            f"| Reasoning effort | `{self.config.run.reasoning_display}` |",
            "",
            "## Metrics",
            "",
            "| Metric | Value |",
            "| --- | --- |",
            f"| Pass rate | **{summary.pass_rate:.2%}** |",
            f"| Total cases | {summary.total_cases} |",
            f"| Completed cases | {summary.completed_cases} |",
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

        lines.extend(
            [
                "",
                "## Results",
                "",
                "| # | Case | Status | Time | Tokens | Detail |",
                "| --- | --- | --- | --- | --- | --- |",
            ]
        )
        for index, item in enumerate(sorted(results, key=lambda r: r.get("case_id", "")), start=1):
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


def _shorten(value: str, limit: int = 160) -> str:
    compact = " ".join(str(value).split())
    if len(compact) <= limit:
        return compact
    return compact[: limit - 3] + "..."


def _cell(value: Any) -> str:
    text = _shorten(str(value)).replace("|", "\\|")
    return text or "n/a"


def _detail(item: dict[str, Any]) -> str:
    if item.get("status") == "PASSED":
        return "passed"
    error = item.get("error")
    if error:
        return str(error)
    expected = item.get("expected")
    actual = item.get("actual")
    if expected is not None or actual is not None:
        return f"expected {expected} | got {actual}"
    return "n/a"
