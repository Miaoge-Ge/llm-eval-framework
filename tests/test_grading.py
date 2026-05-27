from llm_eval.clients import GenerationResult
from llm_eval.runner import _domain_breakdown
from llm_eval.tasks import GPQATask, TaskCase
from llm_eval.utils import (
    extract_choice_letter,
    extract_last_boxed,
    natural_sort_key,
)


class FakeClient:
    def __init__(self, content: str) -> None:
        self.content = content

    def generate(self, messages):
        return GenerationResult(
            content=self.content,
            usage={"prompt_tokens": 5, "completion_tokens": 7, "total_tokens": 12},
            http_status_code=200,
        )


def test_extract_last_boxed_handles_nested_braces():
    assert extract_last_boxed(r"so \boxed{\frac{\pi}{2}} done") == r"\frac{\pi}{2}"


def test_extract_last_boxed_returns_last_match():
    assert extract_last_boxed(r"\boxed{A} then \boxed{B}") == "B"


def test_extract_last_boxed_missing_returns_none():
    assert extract_last_boxed("no box here") is None


def test_extract_choice_letter_variants():
    assert extract_choice_letter(r"reasoning ... \boxed{D}") == "D"
    assert extract_choice_letter("I think the answer is (B).") == "B"
    assert extract_choice_letter("Answer: C") == "C"
    assert extract_choice_letter("no clear choice") is None


def test_natural_sort_orders_numeric_suffixes():
    ids = ["HumanEval/10", "HumanEval/2", "HumanEval/1", "HumanEval/20"]
    assert sorted(ids, key=natural_sort_key) == [
        "HumanEval/1",
        "HumanEval/2",
        "HumanEval/10",
        "HumanEval/20",
    ]


def test_domain_breakdown_aggregates_by_metadata():
    results = [
        {"status": "PASSED", "metadata": {"domain": "Physics"}},
        {"status": "FAILED", "metadata": {"domain": "Physics"}},
        {"status": "PASSED", "metadata": {"domain": "Biology"}},
        {"status": "PASSED", "metadata": {}},
    ]
    assert _domain_breakdown(results) == [("Biology", 1, 1), ("Physics", 1, 2)]


def test_gpqa_grades_choice_letter(fake_config):
    task = GPQATask(fake_config)
    case = TaskCase(case_id="gpqa/1", payload={"problem": "Q?", "answer": "B", "domain": "Physics"})

    passed = task.evaluate_case(case, FakeClient(r"Reasoning... \boxed{B}"))
    assert passed.status == "PASSED"
    assert passed.metadata["domain"] == "Physics"

    failed = task.evaluate_case(case, FakeClient(r"\boxed{A}"))
    assert failed.status == "FAILED"
