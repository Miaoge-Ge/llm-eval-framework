from __future__ import annotations

import math
import re
import time
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from typing import Any

from .clients import GenerationResult, OpenAICompatibleClient
from .settings import FrameworkConfig
from .utils import (
    dedent_code,
    execute_python,
    extract_choice_letter,
    extract_python_code,
    indent_block,
    last_numeric_token,
    load_jsonl,
)

DEFAULT_TASK_NAME = "humaneval"


@dataclass(frozen=True)
class TaskCase:
    case_id: str
    payload: dict[str, Any]


@dataclass
class TaskResult:
    case_id: str
    status: str
    duration_seconds: float
    http_status_code: int | None = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    expected: Any = None
    actual: Any = None
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["duration_human"] = f"{self.duration_seconds:.2f}s"
        return payload


class BaseEvaluationTask(ABC):
    task_name: str

    def __init__(self, config: FrameworkConfig) -> None:
        self.config = config
        self.dataset_path = config.dataset.path

    def load_cases(self) -> list[TaskCase]:
        rows = load_jsonl(self.dataset_path)
        return [TaskCase(case_id=self.case_id_for(row), payload=row) for row in rows]

    @abstractmethod
    def case_id_for(self, row: dict[str, Any]) -> str:
        raise NotImplementedError

    @abstractmethod
    def evaluate_case(self, case: TaskCase, client: OpenAICompatibleClient) -> TaskResult:
        raise NotImplementedError

    def _api_error_result(self, case: TaskCase, started_at: float, generation: GenerationResult) -> TaskResult:
        return TaskResult(
            case_id=case.case_id,
            status="API_ERROR_FATAL" if generation.fatal else "API_ERROR",
            duration_seconds=time.time() - started_at,
            http_status_code=generation.http_status_code,
            error=generation.error,
        )

    def _usage_result(
        self,
        case: TaskCase,
        started_at: float,
        generation: GenerationResult,
        status: str,
        **kwargs: Any,
    ) -> TaskResult:
        return TaskResult(
            case_id=case.case_id,
            status=status,
            duration_seconds=time.time() - started_at,
            http_status_code=generation.http_status_code,
            prompt_tokens=generation.usage.get("prompt_tokens", 0),
            completion_tokens=generation.usage.get("completion_tokens", 0),
            total_tokens=generation.usage.get("total_tokens", 0),
            **kwargs,
        )


class CodeGenerationTask(BaseEvaluationTask):
    system_prompt = (
        "You are a rigorous Python engineer. "
        "Return only executable Python code. "
        "Do not add Markdown, commentary, or explanations."
    )

    shared_header = "\n".join(
        [
            "from typing import Any, Dict, List, Optional, Set, Tuple, Union",
            "import collections",
            "import functools",
            "import heapq",
            "import itertools",
            "import math",
            "import re",
            "import sys",
        ]
    )

    def build_messages(self, case: TaskCase) -> list[dict[str, str]]:
        return [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": self.user_prompt(case)},
        ]

    @abstractmethod
    def user_prompt(self, case: TaskCase) -> str:
        raise NotImplementedError

    @abstractmethod
    def build_test_program(self, case: TaskCase, generated_code: str) -> str:
        raise NotImplementedError

    def evaluate_case(self, case: TaskCase, client: OpenAICompatibleClient) -> TaskResult:
        started_at = time.time()
        generation = client.generate(self.build_messages(case))
        if generation.error:
            return self._api_error_result(case, started_at, generation)
        code = self.normalize_candidate(case, generation.content)
        test_program = self.build_test_program(case, code)
        status, details = execute_python(test_program, self.config.run.execution_timeout_seconds)
        return self._usage_result(
            case,
            started_at,
            generation,
            status=status,
            actual=details if status != "PASSED" else "passed",
            error=details if status != "PASSED" else None,
        )

    def normalize_candidate(self, case: TaskCase, raw_text: str) -> str:
        return extract_python_code(raw_text)


class HumanEvalTask(CodeGenerationTask):
    task_name = "humaneval"

    def case_id_for(self, row: dict[str, Any]) -> str:
        return str(row["task_id"])

    def user_prompt(self, case: TaskCase) -> str:
        prompt = case.payload["prompt"].rstrip()
        return f"Complete the Python function below. Keep the original signature and docstring.\n\n{prompt}"

    def normalize_candidate(self, case: TaskCase, raw_text: str) -> str:
        code = super().normalize_candidate(case, raw_text)
        entry_point = case.payload["entry_point"]
        if f"def {entry_point}" in code:
            return code
        prompt = case.payload["prompt"].rstrip()
        return f"{prompt}\n{indent_block(dedent_code(code), 4)}"

    def build_test_program(self, case: TaskCase, generated_code: str) -> str:
        return "\n\n".join(
            [
                self.shared_header,
                generated_code,
                case.payload["test"].strip(),
                f"check({case.payload['entry_point']})",
            ]
        )


class HumanEvalPlusTask(HumanEvalTask):
    task_name = "humanevalplus"

    numpy_shim = "\n".join(
        [
            "import types",
            "_numpy = types.ModuleType('numpy')",
            "def _is_seq(x): return isinstance(x, (list, tuple))",
            "def _allclose(a, b, rtol=1e-7, atol=0.0):",
            "    if _is_seq(a) and _is_seq(b):",
            "        return len(a) == len(b) and all(_allclose(x, y, rtol=rtol, atol=atol) for x, y in zip(a, b))",
            "    try:",
            "        return abs(a - b) <= (atol + rtol * abs(b))",
            "    except Exception:",
            "        return a == b",
            "_numpy.allclose = _allclose",
            "_numpy.isclose = lambda a, b, rtol=1e-7, atol=0.0: _allclose(a, b, rtol=rtol, atol=atol)",
            "_numpy.ndarray = type('ndarray', (), {})",
            "_numpy.float64 = float",
            "_numpy.float32 = float",
            "_numpy.nan = float('nan')",
            "_numpy.inf = float('inf')",
            "sys.modules['numpy'] = _numpy",
        ]
    )

    def build_test_program(self, case: TaskCase, generated_code: str) -> str:
        return "\n\n".join(
            [
                self.shared_header,
                self.numpy_shim,
                generated_code,
                case.payload["test"].strip(),
                f"check({case.payload['entry_point']})",
            ]
        )


class MBPPTask(CodeGenerationTask):
    task_name = "mbpp"

    def case_id_for(self, row: dict[str, Any]) -> str:
        return str(row["task_id"])

    def user_prompt(self, case: TaskCase) -> str:
        tests = case.payload.get("test_list") or []
        function_name_hint = ""
        if tests:
            match = re.search(r"assert\s+(\w+)\(", tests[0])
            if match:
                function_name_hint = f"\nThe public function name must be `{match.group(1)}`."
        return (
            f"Write a Python solution for this MBPP problem:\n{case.payload['text']}"
            f"{function_name_hint}\n"
            "Return only code."
        )

    def build_test_program(self, case: TaskCase, generated_code: str) -> str:
        sections = [self.shared_header, generated_code]
        setup = (case.payload.get("test_setup_code") or "").strip()
        if setup:
            sections.append(setup)
        test_list = list(case.payload.get("test_list") or [])
        challenge_list = list(case.payload.get("challenge_test_list") or [])
        if challenge_list:
            test_list.extend(challenge_list)
        sections.append("\n".join(test_list))
        return "\n\n".join(section for section in sections if section)


class GSM8KTask(BaseEvaluationTask):
    task_name = "gsm"

    def case_id_for(self, row: dict[str, Any]) -> str:
        return str(row.get("_row_index", row.get("id", "unknown")))

    def evaluate_case(self, case: TaskCase, client: OpenAICompatibleClient) -> TaskResult:
        started_at = time.time()
        messages = [
            {
                "role": "system",
                "content": ("Solve grade-school math carefully. End with a single line in the format `#### <answer>`."),
            },
            {"role": "user", "content": case.payload["question"]},
        ]
        generation = client.generate(messages)
        if generation.error:
            return self._api_error_result(case, started_at, generation)

        expected = last_numeric_token(case.payload["answer"])
        actual = last_numeric_token(generation.content)
        is_correct = self._numbers_match(actual, expected)
        return self._usage_result(
            case,
            started_at,
            generation,
            status="PASSED" if is_correct else "FAILED",
            expected=expected,
            actual=actual,
        )

    def _numbers_match(self, actual: str | None, expected: str | None) -> bool:
        if actual is None or expected is None:
            return False
        try:
            return math.isclose(float(actual), float(expected), rel_tol=0.0, abs_tol=1e-9)
        except ValueError:
            return actual.strip() == expected.strip()


class GPQATask(BaseEvaluationTask):
    task_name = "gpqa"

    def case_id_for(self, row: dict[str, Any]) -> str:
        return str(row.get("id", row.get("_row_index", "unknown")))

    def evaluate_case(self, case: TaskCase, client: OpenAICompatibleClient) -> TaskResult:
        started_at = time.time()
        messages = [
            {
                "role": "system",
                "content": (
                    "Answer the multiple-choice question. Reason briefly, "
                    "then give the final answer as \\boxed{A}, \\boxed{B}, \\boxed{C}, or \\boxed{D}."
                ),
            },
            {"role": "user", "content": case.payload["problem"]},
        ]
        generation = client.generate(messages)
        if generation.error:
            return self._api_error_result(case, started_at, generation)

        expected = str(case.payload["answer"]).strip().upper()
        actual = extract_choice_letter(generation.content)
        is_correct = actual is not None and actual == expected
        return self._usage_result(
            case,
            started_at,
            generation,
            status="PASSED" if is_correct else "FAILED",
            expected=expected,
            actual=actual,
            metadata={"domain": case.payload.get("domain", "")},
        )


TASK_REGISTRY: dict[str, Callable[[FrameworkConfig], BaseEvaluationTask]] = {
    HumanEvalTask.task_name: HumanEvalTask,
    HumanEvalPlusTask.task_name: HumanEvalPlusTask,
    MBPPTask.task_name: MBPPTask,
    GSM8KTask.task_name: GSM8KTask,
    GPQATask.task_name: GPQATask,
}
