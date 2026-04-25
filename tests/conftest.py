from pathlib import Path

import pytest

from llm_eval.settings import DatasetConfig, FrameworkConfig, ModelConfig, RunConfig


@pytest.fixture
def fake_config(tmp_path):
    dataset_path = tmp_path / "dataset.jsonl"
    dataset_path.write_text("", encoding="utf-8")
    return FrameworkConfig(
        model=ModelConfig(api_key="x", base_url="https://example.invalid/v1", model_name="demo"),
        dataset=DatasetConfig(name="humaneval", path=dataset_path),
        run=RunConfig(task="humaneval", output_dir=tmp_path / "results" / "demo"),
        model_config_path=Path("model.yaml"),
    )
