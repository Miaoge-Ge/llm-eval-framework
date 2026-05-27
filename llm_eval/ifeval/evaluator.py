"""Strict and loose instruction-following scoring for IFEval.

Wraps the vendored Google IFEval checkers (instructions*.py) with the
evaluation logic from the original `evaluation_lib`, exposing a single
`evaluate` entry point used by the IFEval task.
"""

from __future__ import annotations

from typing import Any

from . import instructions_registry


def _build_checker(instruction_id: str, kwargs: dict[str, Any], prompt: str) -> Any:
    checker_cls = instructions_registry.INSTRUCTION_DICT[instruction_id]
    checker = checker_cls(instruction_id)
    cleaned = {key: value for key, value in (kwargs or {}).items() if value is not None}
    checker.build_description(**cleaned)
    args = checker.get_instruction_args()
    if args and "prompt" in args:
        checker.build_description(prompt=prompt)
    return checker


def _response_variants(response: str) -> list[str]:
    """Response rewrites used by IFEval's loose metric (strip leading/trailing
    lines and markdown emphasis markers, in every combination)."""
    lines = response.split("\n")
    base = [
        response,
        "\n".join(lines[1:]).strip(),
        "\n".join(lines[:-1]).strip(),
        "\n".join(lines[1:-1]).strip(),
    ]
    return base + [variant.replace("*", "") for variant in base]


def evaluate(
    prompt: str,
    response: str,
    instruction_id_list: list[str],
    kwargs_list: list[dict[str, Any]],
) -> dict[str, Any]:
    strict_flags: list[bool] = []
    loose_flags: list[bool] = []
    variants = _response_variants(response)

    for index, instruction_id in enumerate(instruction_id_list):
        kwargs = kwargs_list[index] if index < len(kwargs_list) else {}
        checker = _build_checker(instruction_id, kwargs, prompt)

        strict_flags.append(bool(response.strip()) and checker.check_following(response))
        loose_flags.append(any(bool(v.strip()) and checker.check_following(v) for v in variants))

    total = len(instruction_id_list)
    return {
        "instructions_total": total,
        "instructions_followed_strict": sum(strict_flags),
        "instructions_followed_loose": sum(loose_flags),
        "prompt_strict": total > 0 and all(strict_flags),
        "prompt_loose": total > 0 and all(loose_flags),
    }
