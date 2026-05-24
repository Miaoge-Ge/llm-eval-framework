from llm_eval.tasks import GSM8KTask, HumanEvalTask, TaskCase
from llm_eval.utils import extract_python_code


def test_extract_python_code_prefers_fenced_block():
    text = "hello\n```python\nprint('ok')\n```\nbye"
    assert extract_python_code(text) == "print('ok')"


def test_humaneval_reinjects_prompt_signature(fake_config):
    task = HumanEvalTask(fake_config)
    case = TaskCase(
        case_id="HumanEval/0",
        payload={
            "prompt": 'def add(a, b):\n    """add"""\n',
            "entry_point": "add",
            "test": "def check(candidate):\n    assert candidate(1, 2) == 3",
        },
    )
    candidate = "return a + b"
    normalized = task.normalize_candidate(case, candidate)
    assert normalized.startswith("def add")
    assert "return a + b" in normalized


def test_gsm_exact_number_match(fake_config):
    task = GSM8KTask(fake_config)
    assert task._numbers_match("18", "18")
    assert not task._numbers_match("19", "18")
