from llm_eval.cli import main


def test_list_tasks_prints_registry(capsys):
    exit_code = main(["run", "--list-tasks"])
    out = capsys.readouterr().out
    assert exit_code == 0
    for name in ("humaneval", "gsm", "math500", "gpqa"):
        assert name in out


def test_unknown_task_returns_error(tmp_path, capsys):
    config = tmp_path / "model.yaml"
    config.write_text(
        "\n".join(
            [
                "api_key: dummy",
                "base_url: https://example.invalid/v1",
                "model_name: demo-model",
            ]
        ),
        encoding="utf-8",
    )
    exit_code = main(["run", "--config", str(config), "--task", "does-not-exist"])
    err = capsys.readouterr().err
    assert exit_code == 1
    assert "does-not-exist" in err
