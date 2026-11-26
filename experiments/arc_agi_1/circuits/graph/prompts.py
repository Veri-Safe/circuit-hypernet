from __future__ import annotations

from typing import Any


def build_math_prompt_content(query: str) -> str:
    return (
        query.strip()
        + "\n\nSolve the problem step by step. Put the final numeric answer after ####."
    )


def build_chat_prompt(tokenizer: Any, query: str) -> str:
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": build_math_prompt_content(query)}],
        tokenize=False,
        add_generation_prompt=True,
    )


def build_trace_text(prompt_text: str, generated_answer: str) -> str:
    return prompt_text + generated_answer.rstrip()
