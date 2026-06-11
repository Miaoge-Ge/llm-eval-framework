import json
import time

from llm_eval.clients import GenerationResult
from llm_eval.runner import EvaluationRunner
from llm_eval.settings import DatasetConfig, FrameworkConfig, ModelConfig, RunConfig
from llm_eval.tasks import TASK_REGISTRY


class FatalClient:
    """Simulates an invalid API key: every call fails fast with a fatal error."""

    def __init__(self) -> None:
        self.calls = 0

    def generate(self, messages):
        self.calls += 1
        time.sleep(0.1)
        return GenerationResult(
            content="",
            usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            error="Incorrect API key provided",
            fatal=True,
            http_status_code=401,
        )


class EchoClient:
    """Always answers `#### 4`, so questions expecting 4 pass."""

    def generate(self, messages):
        return GenerationResult(
            content="#### 4",
            usage={"prompt_tokens": 5, "completion_tokens": 7, "total_tokens": 12},
            http_status_code=200,
        )


def _gsm_config(tmp_path, rows, **run_kwargs):
    dataset = tmp_path / "gsm.jsonl"
    dataset.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    return FrameworkConfig(
        model=ModelConfig(api_key="x", base_url="https://example.invalid/v1", model_name="demo"),
        dataset=DatasetConfig(name="gsm", path=dataset),
        run=RunConfig(task="gsm", workers=1, output_dir=tmp_path / "results", **run_kwargs),
        model_config_path=tmp_path / "model.yaml",
    )


def test_runner_aborts_on_fatal_api_error(tmp_path):
    rows = [{"question": f"What is {i}+{i}?", "answer": f"#### {2 * i}"} for i in range(20)]
    runner = EvaluationRunner(_gsm_config(tmp_path, rows))
    runner.client = FatalClient()

    summary, writer = runner.run()

    assert summary.status_counts.get("API_ERROR_FATAL", 0) >= 1
    assert summary.completed_cases < summary.total_cases
    # every completed case is persisted incrementally
    persisted = writer.paths.results_jsonl.read_text(encoding="utf-8").strip().splitlines()
    assert len(persisted) == summary.completed_cases


def test_runner_writes_run_artifacts(tmp_path):
    rows = [{"question": "What is 2+2?", "answer": "#### 4"} for _ in range(3)]
    runner = EvaluationRunner(_gsm_config(tmp_path, rows))
    runner.client = EchoClient()

    summary, writer = runner.run()

    assert summary.pass_rate == 1.0
    assert writer.paths.report_md.exists()
    assert writer.paths.config_json.exists()
    config_snapshot = json.loads(writer.paths.config_json.read_text(encoding="utf-8"))
    assert config_snapshot["model"]["api_key"] == "***"
    persisted = [json.loads(line) for line in writer.paths.results_jsonl.read_text(encoding="utf-8").splitlines()]
    assert len(persisted) == 3
    assert all(item["status"] == "PASSED" for item in persisted)


def test_runner_respects_case_limit(tmp_path):
    rows = [{"question": "What is 2+2?", "answer": "#### 4"} for _ in range(10)]
    runner = EvaluationRunner(_gsm_config(tmp_path, rows, limit=3))
    runner.client = EchoClient()

    summary, _ = runner.run()

    assert summary.total_cases == 3
    assert summary.completed_cases == 3


def test_task_registry_auto_registers_all_tasks():
    expected = {
        "humaneval",
        "humanevalplus",
        "mbpp",
        "mbppplus",
        "gsm",
        "aime2025",
        "aime2026",
        "gpqa",
        "mmlu_pro",
        "ifeval",
        "livecodebench",
    }
    assert expected == set(TASK_REGISTRY)
