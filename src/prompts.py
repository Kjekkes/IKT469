"""Prompt templates.

Kept in one place so the ablation script can iterate over them.
"""
from __future__ import annotations

from typing import List

from .retriever import Retrieved


SYSTEM_GROUNDED = (
    "You are a helpful assistant that answers questions strictly using the "
    "provided context. If the context does not contain the answer, say "
    '"I don\'t know based on the provided context." Be concise.'
)

SYSTEM_VANILLA = "You are a helpful assistant. Answer the question concisely."


def format_context(results: List[Retrieved]) -> str:
    blocks = []
    for r in results:
        title = r.chunk.metadata.get("title") or r.chunk.source_id
        blocks.append(f"[{r.rank + 1}] {title}\n{r.chunk.text}")
    return "\n\n".join(blocks)


def grounded_prompt(question: str, results: List[Retrieved]) -> str:
    return (
        f"Context:\n{format_context(results)}\n\n"
        f"Question: {question}\n\n"
        "Answer using only the context above. Cite the bracketed source "
        "numbers you used (e.g. [1], [3])."
    )


def vanilla_prompt(question: str, _results: List[Retrieved]) -> str:
    return f"Question: {question}\n\nAnswer:"


PROMPT_VARIANTS = {
    "grounded": (SYSTEM_GROUNDED, grounded_prompt),
    "vanilla": (SYSTEM_VANILLA, vanilla_prompt),
}
