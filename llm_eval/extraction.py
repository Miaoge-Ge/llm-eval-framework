"""Parsing helpers that pull gradable answers out of raw model output."""

from __future__ import annotations

import re

PYTHON_BLOCK_PATTERN = re.compile(r"```(?:python|py)?\s*(.*?)```", re.IGNORECASE | re.DOTALL)


def extract_python_code(text: str) -> str:
    if not text:
        return ""
    match = PYTHON_BLOCK_PATTERN.search(text)
    extracted = match.group(1) if match else text
    return dedent_code(extracted.strip())


def dedent_code(code: str) -> str:
    lines = code.splitlines()
    min_indent: int | None = None
    for line in lines:
        stripped = line.lstrip()
        if not stripped:
            continue
        indent = len(line) - len(stripped)
        min_indent = indent if min_indent is None else min(min_indent, indent)
    if not min_indent:
        return code.strip()
    return "\n".join(line[min_indent:] if line.strip() else "" for line in lines).strip()


def indent_block(code: str, spaces: int) -> str:
    prefix = " " * spaces
    return "\n".join(prefix + line if line.strip() else "" for line in code.splitlines())


def last_numeric_token(text: str) -> str | None:
    if not text:
        return None
    explicit = re.search(r"####\s*(-?\d[\d,]*(?:\.\d+)?)", text)
    if explicit:
        return explicit.group(1).replace(",", "")
    matches = re.findall(r"-?\d[\d,]*(?:\.\d+)?", text)
    if not matches:
        return None
    return matches[-1].replace(",", "")


def extract_last_boxed(text: str) -> str | None:
    if not text:
        return None
    needle = "\\boxed"
    start = text.rfind(needle)
    if start == -1:
        return None
    index = start + len(needle)
    while index < len(text) and text[index] != "{":
        if text[index] == " ":
            index += 1
            continue
        return None
    if index >= len(text):
        return None
    depth = 0
    content_start = index + 1
    for position in range(index, len(text)):
        char = text[position]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[content_start:position]
    return None


def extract_choice_letter(text: str, last_letter: str = "D") -> str | None:
    if not text:
        return None
    char_class = f"[A-{last_letter}]"
    boxed = extract_last_boxed(text)
    if boxed:
        match = re.search(char_class, boxed.upper())
        if match:
            return match.group(0)
    stripped = text.strip()
    match = re.search(rf"answer\s*(?:is|:)?\s*\(?({char_class})\)?", stripped, re.IGNORECASE | re.MULTILINE)
    if match:
        return match.group(1).upper()
    # Bare letters must stay case-sensitive: with IGNORECASE the article "a"
    # at the end of a line would be read as answer A.
    for pattern in (rf"\b({char_class})\b\s*$", rf"\(({char_class})\)"):
        match = re.search(pattern, stripped, re.MULTILINE)
        if match:
            return match.group(1)
    return None
