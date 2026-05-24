from llm_eval.settings import load_framework_config
from llm_eval.utils import repo_root


def test_load_framework_config_resolves_env(tmp_path, monkeypatch):
    model_config = tmp_path / "model.yaml"

    monkeypatch.setenv("TEST_API_KEY", "secret-token")
    model_config.write_text(
        "\n".join(
            [
                "api_key: ${TEST_API_KEY}",
                "base_url: https://example.invalid/v1",
                "model_name: demo-model",
                "workers: 10",
                "thinking_enabled: true",
                "reasoning_effort: high",
            ]
        ),
        encoding="utf-8",
    )
    config = load_framework_config(model_config, task="humaneval")

    assert config.model.api_key == "secret-token"
    assert config.model.model_name == "demo-model"
    assert config.run.workers == 10
    assert config.run.thinking_enabled is True
    assert config.run.reasoning_effort == "high"
    assert config.dataset.path == (repo_root() / "datasets" / "HumanEval.jsonl").resolve()
    assert config.run.output_dir == (repo_root() / "results" / "demo-model").resolve()


def test_load_framework_config_parses_disabled_thinking(tmp_path, monkeypatch):
    model_config = tmp_path / "model.yaml"

    monkeypatch.setenv("TEST_API_KEY", "secret-token")
    model_config.write_text(
        "\n".join(
            [
                "api_key: ${TEST_API_KEY}",
                "base_url: https://example.invalid/v1",
                "model_name: demo-model",
                "thinking_enabled: disabled",
                "reasoning_effort: high",
            ]
        ),
        encoding="utf-8",
    )
    config = load_framework_config(model_config, task="humaneval")

    assert config.run.thinking_enabled is False
    assert config.run.reasoning_effort == "high"
