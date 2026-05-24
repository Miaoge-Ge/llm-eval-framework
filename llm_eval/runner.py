from __future__ import annotations

import concurrent.futures
import time
from collections import Counter
from dataclasses import asdict, dataclass
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
        sample_passes: list[dict[str, Any]] = []
        sample_issues: list[dict[str, Any]] = []

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
                        writer.write_result(payload)
                        totals[result.status] += 1
                        total_duration += result.duration_seconds
                        total_tokens += result.total_tokens
                        prompt_tokens += result.prompt_tokens
                        completion_tokens += result.completion_tokens
                        last_http_status_code = result.http_status_code or last_http_status_code

                        if result.status == "PASSED" and len(sample_passes) < 3:
                            sample_passes.append(payload)
                        elif result.status != "PASSED" and len(sample_issues) < 5:
                            sample_issues.append(payload)

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
            payload = asdict(summary)
            payload["artifacts"] = {
                "results_jsonl": str(writer.paths.results_jsonl),
                "summary_json": str(writer.paths.summary_json),
                "resolved_config_json": str(writer.paths.config_json),
                "report_md": str(writer.paths.report_md),
            }
            writer.write_summary(payload)
            writer.write_report(
                self._build_markdown_report(
                    summary=summary,
                    output_dir=writer.paths.root,
                    sample_passes=sample_passes,
                    sample_issues=sample_issues,
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
        sample_passes: list[dict[str, Any]],
        sample_issues: list[dict[str, Any]],
    ) -> str:
        lines = [
            "# Evaluation Report",
            "",
            "## Overview",
            "",
            f"- Task: `{summary.task}`",
            f"- Model: `{summary.model}`",
            f"- Dataset: `{self.config.dataset.path}`",
            f"- Workers: `{self.config.run.workers}`",
            f"- Thinking enabled: `{self.config.run.thinking_enabled}`",
            f"- Reasoning effort: `{self.config.run.reasoning_display}`",
            f"- Output directory: `{output_dir}`",
            "",
            "## Metrics",
            "",
            f"- Total cases: `{summary.total_cases}`",
            f"- Completed cases: `{summary.completed_cases}`",
            f"- Pass rate: `{summary.pass_rate:.2%}`",
            f"- Wall clock: `{summary.wall_clock_human}`",
            f"- Average case time: `{summary.average_case_seconds:.2f}s`",
            f"- Throughput: `{summary.throughput_tokens_per_second:.1f} tokens/s`",
            f"- Status counts: `{summary.status_counts}`",
            f"- Token usage: `{summary.token_usage}`",
            "",
            "## Generated Files",
            "",
            f"- `{summary.task}_results.jsonl`",
            f"- `{summary.task}_summary.json`",
            f"- `{summary.task}_report.md`",
            "- `resolved_config.json`",
            "",
        ]

        if sample_issues:
            lines.extend(["## Sample Issues", ""])
            for item in sample_issues:
                lines.extend(
                    [
                        f"### {item.get('case_id', 'unknown')}",
                        f"- Status: `{item.get('status', 'unknown')}`",
                        f"- Duration: `{item.get('duration_human', 'n/a')}`",
                        f"- Detail: `{_shorten(item.get('error') or item.get('actual') or 'n/a')}`",
                        "",
                    ]
                )

        if sample_passes:
            lines.extend(["## Sample Passes", ""])
            for item in sample_passes:
                lines.extend(
                    [
                        f"### {item.get('case_id', 'unknown')}",
                        f"- Status: `{item.get('status', 'unknown')}`",
                        f"- Duration: `{item.get('duration_human', 'n/a')}`",
                        "",
                    ]
                )

        return "\n".join(lines).strip() + "\n"


def _shorten(value: str, limit: int = 160) -> str:
    compact = " ".join(str(value).split())
    if len(compact) <= limit:
        return compact
    return compact[: limit - 3] + "..."
