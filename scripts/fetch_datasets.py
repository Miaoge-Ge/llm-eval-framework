"""Download benchmark datasets from Hugging Face and normalize them to the
JSONL layout expected by llm_eval tasks.

Run with a throwaway dependency set so the runtime project stays lean:

    uv run --with datasets python scripts/fetch_datasets.py aime
    uv run --with datasets python scripts/fetch_datasets.py all

Each builder writes a single file under datasets/ with the field names the
matching task class reads.
"""

from __future__ import annotations

import argparse
import json
import re
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

from datasets import load_dataset

DATASETS_DIR = Path(__file__).resolve().parent.parent / "datasets"


def _write_jsonl(filename: str, rows: Iterable[dict[str, Any]]) -> None:
    path = DATASETS_DIR / filename
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    print(f"wrote {count} rows -> {path.relative_to(DATASETS_DIR.parent)}")


def build_aime2025() -> None:
    rows: list[dict[str, Any]] = []
    for config in ("AIME2025-I", "AIME2025-II"):
        dataset = load_dataset("opencompass/AIME2025", config, split="test")
        for index, record in enumerate(dataset):
            rows.append(
                {
                    "id": f"{config}/{index}",
                    "problem": record["question"],
                    "answer": str(record["answer"]).strip(),
                }
            )
    _write_jsonl("AIME2025.jsonl", rows)


def _infer_entry_point(code: str, test: str) -> str:
    defs = re.findall(r"def (\w+)\(", code)
    used = [name for name in defs if name in test]
    candidates = used or defs
    return candidates[-1] if candidates else "solution"


def build_mbppplus() -> None:
    dataset = load_dataset("evalplus/mbppplus", split="test")
    rows: list[dict[str, Any]] = []
    for record in dataset:
        rows.append(
            {
                "task_id": f"Mbpp/{record['task_id']}",
                "prompt": record["prompt"],
                "entry_point": _infer_entry_point(record["code"], record["test"]),
                "test_imports": list(record["test_imports"]),
                "test": record["test"],
            }
        )
    _write_jsonl("MBPPPlus.jsonl", rows)


def build_ifeval() -> None:
    dataset = load_dataset("google/IFEval", split="train")
    rows: list[dict[str, Any]] = []
    for record in dataset:
        rows.append(
            {
                "key": record["key"],
                "prompt": record["prompt"],
                "instruction_id_list": list(record["instruction_id_list"]),
                "kwargs": [dict(item) for item in record["kwargs"]],
            }
        )
    _write_jsonl("IFEval.jsonl", rows)


def build_livecodebench() -> None:
    # code_generation_lite ships a (now-unsupported) loading script, so we read
    # the raw test.jsonl directly. Only stdin/stdout problems are kept, using the
    # plaintext public test cases (private cases are large base64+zlib blobs).
    from huggingface_hub import hf_hub_download

    path = hf_hub_download("livecodebench/code_generation_lite", "test.jsonl", repo_type="dataset")
    rows: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            public = json.loads(record["public_test_cases"])
            if not public or any(case.get("testtype") != "stdin" for case in public):
                continue
            rows.append(
                {
                    "question_id": record["question_id"],
                    "title": record["question_title"],
                    "platform": record["platform"],
                    "difficulty": record["difficulty"],
                    "problem": record["question_content"],
                    "tests": [{"input": case["input"], "output": case["output"]} for case in public],
                }
            )
    _write_jsonl("LiveCodeBench.jsonl", rows)


BUILDERS: dict[str, Callable[[], None]] = {
    "aime": build_aime2025,
    "mbppplus": build_mbppplus,
    "ifeval": build_ifeval,
    "livecodebench": build_livecodebench,
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("target", choices=[*BUILDERS, "all"], help="dataset to fetch")
    args = parser.parse_args()

    targets = BUILDERS.values() if args.target == "all" else [BUILDERS[args.target]]
    for builder in targets:
        builder()


if __name__ == "__main__":
    main()
