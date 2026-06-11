from __future__ import annotations

import concurrent.futures
import time
from collections import Counter
from typing import Any

from tqdm import tqdm

from .clients import OpenAICompatibleClient
from .reporting import ResultWriter, RunSummary, render_markdown_report
from .settings import FrameworkConfig
from .status import GRADED_STATUSES, Status
from .tasks import TASK_REGISTRY, BaseEvaluationTask, TaskCase, TaskResult
from .utils import format_seconds


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
        if self.config.run.limit is not None:
            cases = cases[: self.config.run.limit]
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
        fatal_error: str | None = None

        with ResultWriter(self.config) as writer:
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
                        f"  Output: {writer.paths.root}",
                        f"  Total cases: {len(cases)}",
                    ]
                ),
                flush=True,
            )

            with concurrent.futures.ThreadPoolExecutor(max_workers=self.config.run.workers) as pool:
                future_to_case = {pool.submit(self._evaluate_case, task, case): case for case in cases}
                with tqdm(total=len(cases), desc=self.config.run.task, unit="case", dynamic_ncols=True) as progress:
                    for future in concurrent.futures.as_completed(future_to_case):
                        if future.cancelled():
                            continue
                        result = future.result()
                        payload = result.to_dict()
                        writer.write_result(payload)
                        results.append(payload)
                        totals[result.status] += 1
                        total_duration += result.duration_seconds
                        total_tokens += result.total_tokens
                        prompt_tokens += result.prompt_tokens
                        completion_tokens += result.completion_tokens
                        last_http_status_code = result.http_status_code or last_http_status_code

                        if result.status == Status.API_ERROR_FATAL and fatal_error is None:
                            fatal_error = result.error or "fatal API error"
                            for pending in future_to_case:
                                pending.cancel()
                            progress.write(f"Fatal API error, cancelling remaining cases: {fatal_error}")

                        progress.update(1)
                        progress.set_postfix_str(
                            "passed={passed} failed={failed} http={http}".format(
                                passed=totals.get(Status.PASSED, 0),
                                failed=totals.get(Status.FAILED, 0),
                                http=last_http_status_code if last_http_status_code is not None else "-",
                            ),
                            refresh=False,
                        )

            wall_clock_seconds = time.time() - started_at
            graded = sum(totals[status] for status in GRADED_STATUSES)
            pass_rate = (totals[Status.PASSED] / graded) if graded else 0.0
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
            writer.write_report(render_markdown_report(summary=summary, config=self.config, results=results))
            return summary, writer

    def _evaluate_case(self, task: BaseEvaluationTask, case: TaskCase) -> TaskResult:
        try:
            return task.evaluate_case(case, self.client)
        except Exception as exc:
            return TaskResult(
                case_id=case.case_id,
                status=Status.INTERNAL_ERROR,
                duration_seconds=0.0,
                error=str(exc),
            )
