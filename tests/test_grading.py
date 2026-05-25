from llm_eval.clients import GenerationResult
from llm_eval.tasks import GPQATask, Math500Task, TaskCase
from llm_eval.utils import extract_choice_letter, extract_last_boxed, normalize_math_answer


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


def test_normalize_math_answer_equivalences():
    assert normalize_math_answer(r"\left( 3, \frac{\pi}{2} \right)") == normalize_math_answer(r"(3, \frac{\pi}{2})")
    assert normalize_math_answer(r"\dfrac{1}{2}") == normalize_math_answer(r"\frac{1}{2}")
    assert normalize_math_answer("42.") == "42"
    assert normalize_math_answer("   ") is None


def test_extract_choice_letter_variants():
    assert extract_choice_letter(r"reasoning ... \boxed{D}") == "D"
    assert extract_choice_letter("I think the answer is (B).") == "B"
    assert extract_choice_letter("Answer: C") == "C"
    assert extract_choice_letter("no clear choice") is None


def test_math500_grades_boxed_answer(fake_config):
    task = Math500Task(fake_config)
    case = TaskCase(case_id="math/1", payload={"problem": "1+1?", "answer": r"\frac{1}{2}"})

    passed = task.evaluate_case(case, FakeClient(r"The answer is \boxed{\dfrac{1}{2}}."))
    assert passed.status == "PASSED"

    failed = task.evaluate_case(case, FakeClient(r"\boxed{3}"))
    assert failed.status == "FAILED"


def test_gpqa_grades_choice_letter(fake_config):
    task = GPQATask(fake_config)
    case = TaskCase(case_id="gpqa/1", payload={"problem": "Q?", "answer": "B", "domain": "Physics"})

    passed = task.evaluate_case(case, FakeClient(r"Reasoning... \boxed{B}"))
    assert passed.status == "PASSED"
    assert passed.metadata["domain"] == "Physics"

    failed = task.evaluate_case(case, FakeClient(r"\boxed{A}"))
    assert failed.status == "FAILED"
