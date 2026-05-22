"""Small utilities shared across the codebase."""

from __future__ import annotations

import re


def fix_floats(code_str: str) -> str:
    """Repair common decimal-formatting glitches in generated code.

    The BPE tokenizer occasionally produces float tokens like ``3.5.6`` or
    ``3..`` which break Python parsing. We strip the obvious cases here.
    Keep this conservative — over-aggressive regex edits can damage valid code.
    """
    # '3.5.6' -> '3.56'
    code_str = re.sub(r"(\d+\.\d+)\.(\d+)", r"\1\2", code_str)
    # '..' -> '.'
    code_str = re.sub(r"\.(?=\.)", "", code_str)
    # trailing '.' on integers like '5.' (followed by space or EOL) -> '5'
    code_str = re.sub(r"(\d+)\.(\s|$)", r"\1\2", code_str)
    return code_str


def print_code_with_line_numbers(code_str: str, title: str = "Code") -> None:
    """Pretty-print code with line numbers for debugging."""
    print(f"──────────── {title} ────────────────")
    for i, line in enumerate(code_str.strip().split("\n"), 1):
        print(f"{i:02d}: {line}")
    print("────────────────────────────────────")
