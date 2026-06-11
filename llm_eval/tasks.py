from __future__ import annotations

import math
import re
import time
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from typing import Any

from .clients import GenerationResult, OpenAICompatibleClient
from .execution import execute_python, normalize_stdout, run_python_with_stdin
from .extraction import (
    dedent_code,
    extract_choice_letter,
    extract_last_boxed,
    extract_python_code,
    indent_block,
    last_numeric_token,
)
from .ifeval import evaluate as evaluate_ifeval
from .settings import FrameworkConfig
from .status import Status
from .utils import load_jsonl

DEFAULT_TASK_NAME = "humaneval"

# Populated automatically: any subclass that defines its own `task_name`
# registers itself via BaseEvaluationTask.__init_subclass__.
TASK_REGISTRY: dict[str, type[BaseEvaluationTask]] = {}


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
    task_name: str = ""
    # Looked up in order to build a case id; falls back to the row index.
    id_fields: tuple[str, ...] = ("id",)

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        registered_name = cls.__dict__.get("task_name")
        if registered_name:
            TASK_REGISTRY[registered_name] = cls

    def __init__(self, config: FrameworkConfig) -> None:
        self.config = config
        self.dataset_path = config.dataset.path

    def load_cases(self) -> list[TaskCase]:
        rows = load_jsonl(self.dataset_path)
        return [TaskCase(case_id=self.case_id_for(row), payload=row) for row in rows]

    def case_id_for(self, row: dict[str, Any]) -> str:
        for field_name in self.id_fields:
            if field_name in row:
                return str(row[field_name])
        return str(row.get("_row_index", "unknown"))

    @abstractmethod
    def evaluate_case(self, case: TaskCase, client: OpenAICompatibleClient) -> TaskResult:
        raise NotImplementedError

    def _api_error_result(self, case: TaskCase, started_at: float, generation: GenerationResult) -> TaskResult:
        return TaskResult(
            case_id=case.case_id,
            status=Status.API_ERROR_FATAL if generation.fatal else Status.API_ERROR,
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
            actual=details if status != Status.PASSED else "passed",
            error=details if status != Status.PASSED else None,
        )

    def normalize_candidate(self, case: TaskCase, raw_text: str) -> str:
        return extract_python_code(raw_text)


class HumanEvalTask(CodeGenerationTask):
    task_name = "humaneval"
    id_fields = ("task_id",)

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
    id_fields = ("task_id",)

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


class MBPPPlusTask(CodeGenerationTask):
    task_name = "mbppplus"
    id_fields = ("task_id",)

    numpy_shim = HumanEvalPlusTask.numpy_shim

    def user_prompt(self, case: TaskCase) -> str:
        return (
            f"Write a Python solution for this problem:\n{case.payload['prompt']}\n"
            f"The public function must be named `{case.payload['entry_point']}`.\n"
            "Return only code."
        )

    def build_test_program(self, case: TaskCase, generated_code: str) -> str:
        sections = [self.shared_header, self.numpy_shim, generated_code]
        sections.extend(case.payload.get("test_imports") or [])
        sections.append(case.payload["test"].strip())
        return "\n\n".join(section for section in sections if section)


class GSM8KTask(BaseEvaluationTask):
    task_name = "gsm"
    id_fields = ("_row_index",)

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
            status=Status.PASSED if is_correct else Status.FAILED,
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


class AIMETask(BaseEvaluationTask):
    task_name = "aime2025"

    def evaluate_case(self, case: TaskCase, client: OpenAICompatibleClient) -> TaskResult:
        started_at = time.time()
        messages = [
            {
                "role": "system",
                "content": (
                    "Solve the AIME competition math problem step by step. "
                    "The final answer is an integer between 0 and 999. "
                    "Put it inside \\boxed{} on the last line."
                ),
            },
            {"role": "user", "content": case.payload["problem"]},
        ]
        generation = client.generate(messages)
        if generation.error:
            return self._api_error_result(case, started_at, generation)

        boxed = extract_last_boxed(generation.content)
        actual = last_numeric_token(boxed) if boxed else last_numeric_token(generation.content)
        expected = last_numeric_token(str(case.payload["answer"]))
        is_correct = self._integers_match(actual, expected)
        return self._usage_result(
            case,
            started_at,
            generation,
            status=Status.PASSED if is_correct else Status.FAILED,
            expected=case.payload["answer"],
            actual=actual,
        )

    def _integers_match(self, actual: str | None, expected: str | None) -> bool:
        if actual is None or expected is None:
            return False
        try:
            return int(round(float(actual))) == int(round(float(expected)))
        except ValueError:
            return actual.strip() == expected.strip()


class AIME2026Task(AIMETask):
    task_name = "aime2026"


class MultipleChoiceTask(BaseEvaluationTask):
    """Shared grading skeleton: ask the question, extract a letter, compare."""

    domain_field = "domain"

    @abstractmethod
    def system_prompt(self, row: dict[str, Any]) -> str:
        raise NotImplementedError

    @abstractmethod
    def user_prompt(self, row: dict[str, Any]) -> str:
        raise NotImplementedError

    def last_letter(self, row: dict[str, Any]) -> str:
        return "D"

    def evaluate_case(self, case: TaskCase, client: OpenAICompatibleClient) -> TaskResult:
        started_at = time.time()
        row = case.payload
        messages = [
            {"role": "system", "content": self.system_prompt(row)},
            {"role": "user", "content": self.user_prompt(row)},
        ]
        generation = client.generate(messages)
        if generation.error:
            return self._api_error_result(case, started_at, generation)

        expected = str(row["answer"]).strip().upper()
        actual = extract_choice_letter(generation.content, last_letter=self.last_letter(row))
        is_correct = actual is not None and actual == expected
        return self._usage_result(
            case,
            started_at,
            generation,
            status=Status.PASSED if is_correct else Status.FAILED,
            expected=expected,
            actual=actual,
            metadata={"domain": row.get(self.domain_field, "")},
        )


class GPQATask(MultipleChoiceTask):
    task_name = "gpqa"

    def system_prompt(self, row: dict[str, Any]) -> str:
        return (
            "Answer the multiple-choice question. Reason briefly, "
            "then give the final answer as \\boxed{A}, \\boxed{B}, \\boxed{C}, or \\boxed{D}."
        )

    def user_prompt(self, row: dict[str, Any]) -> str:
        return row["problem"]


class MMLUProTask(MultipleChoiceTask):
    task_name = "mmlu_pro"
    choice_letters = "ABCDEFGHIJ"
    domain_field = "category"

    def system_prompt(self, row: dict[str, Any]) -> str:
        return (
            "Answer the multiple-choice question. Reason briefly, then give the final "
            "answer as \\boxed{X}, where X is the letter of the correct option."
        )

    def user_prompt(self, row: dict[str, Any]) -> str:
        letters = self._letters(row)
        rendered = "\n".join(f"{letter}. {text}" for letter, text in zip(letters, row["options"], strict=False))
        return f"{row['question']}\n\n{rendered}"

    def last_letter(self, row: dict[str, Any]) -> str:
        return self._letters(row)[-1]

    def _letters(self, row: dict[str, Any]) -> str:
        return self.choice_letters[: len(row["options"])]


class IFEvalTask(BaseEvaluationTask):
    task_name = "ifeval"
    id_fields = ("key",)

    def evaluate_case(self, case: TaskCase, client: OpenAICompatibleClient) -> TaskResult:
        started_at = time.time()
        messages = [{"role": "user", "content": case.payload["prompt"]}]
        generation = client.generate(messages)
        if generation.error:
            return self._api_error_result(case, started_at, generation)

        report = evaluate_ifeval(
            case.payload["prompt"],
            generation.content or "",
            list(case.payload["instruction_id_list"]),
            list(case.payload.get("kwargs") or []),
        )
        total = report["instructions_total"]
        strict = report["instructions_followed_strict"]
        loose = report["instructions_followed_loose"]
        return self._usage_result(
            case,
            started_at,
            generation,
            status=Status.PASSED if report["prompt_strict"] else Status.FAILED,
            expected=f"follow all {total} instructions",
            actual=f"strict {strict}/{total}, loose {loose}/{total}",
            metadata={
                "prompt_loose": report["prompt_loose"],
                "instructions_total": total,
                "instructions_followed_strict": strict,
                "instructions_followed_loose": loose,
            },
        )


class LiveCodeBenchTask(BaseEvaluationTask):
    task_name = "livecodebench"
    id_fields = ("question_id",)

    system_prompt = (
        "You are a competitive programming expert. Write a complete Python 3 program "
        "that reads from standard input and prints the answer to standard output. "
        "Return only the program code, without explanations or Markdown."
    )

    def evaluate_case(self, case: TaskCase, client: OpenAICompatibleClient) -> TaskResult:
        started_at = time.time()
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": case.payload["problem"]},
        ]
        generation = client.generate(messages)
        if generation.error:
            return self._api_error_result(case, started_at, generation)

        code = extract_python_code(generation.content)
        timeout = self.config.run.execution_timeout_seconds
        tests = case.payload.get("tests") or []
        status, detail = self._run_tests(code, tests, timeout)
        return self._usage_result(
            case,
            started_at,
            generation,
            status=status,
            actual="passed" if status == Status.PASSED else detail,
            error=None if status == Status.PASSED else detail,
            metadata={
                "domain": case.payload.get("difficulty", ""),
                "platform": case.payload.get("platform", ""),
            },
        )

    def _run_tests(self, code: str, tests: list[dict[str, str]], timeout: int) -> tuple[str, str]:
        for index, test in enumerate(tests):
            run_status, stdout, error = run_python_with_stdin(code, test["input"], timeout)
            if run_status == "TIMEOUT":
                return Status.TIMEOUT, f"test {index}: {error}"
            if run_status != "OK":
                return Status.FAILED, f"test {index} runtime error: {error[:160]}"
            if normalize_stdout(stdout) != normalize_stdout(test["output"]):
                return Status.FAILED, f"test {index}: wrong answer"
        return Status.PASSED, ""
