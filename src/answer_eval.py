"""LLM-as-judge faithfulness scoring.

For BEIR/NQ we don't have short reference answers, so we evaluate the
generated answer against the *retrieved context*: did the model use only
information present in the context, or did it hallucinate?

We score each (context, answer) pair on a 0-2 scale:
    2 = fully grounded
    1 = partially grounded
    0 = unsupported / hallucinated

Implementation is deliberately simple: ask Ollama for a single integer and
parse it. Robust to occasional parse failures (returns None).
"""
from __future__ import annotations

import re
from typing import List, Optional

from .generator import OllamaGenerator
from .retriever import Retrieved


JUDGE_SYSTEM = (
    "You are a strict evaluator. You judge whether an answer is grounded in "
    "a given context. You output ONLY a single integer: 0, 1, or 2."
)


JUDGE_PROMPT_TEMPLATE = """Context:
{context}

Question: {question}

Candidate answer: {answer}

Rate the answer:
  2 = every claim in the answer is directly supported by the context
  1 = some claims are supported, but parts are not in the context
  0 = the answer is not supported by the context or is irrelevant

Output ONLY the integer (0, 1, or 2)."""


def _format_ctx(retrieved: List[Retrieved]) -> str:
    return "\n\n".join(f"[{r.rank + 1}] {r.chunk.text}" for r in retrieved)


def score_faithfulness(
    judge: OllamaGenerator,
    question: str,
    answer: str,
    retrieved: List[Retrieved],
) -> Optional[int]:
    prompt = JUDGE_PROMPT_TEMPLATE.format(
        context=_format_ctx(retrieved),
        question=question,
        answer=answer,
    )
    try:
        raw = judge.generate(prompt, system=JUDGE_SYSTEM)
    except Exception:
        return None
    m = re.search(r"[012]", raw)
    return int(m.group(0)) if m else None
