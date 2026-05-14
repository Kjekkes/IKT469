"""UiA eval harness.

Three things we score per question:

1. **Source recall** — did the retriever surface a chunk whose URL or title
   contains any of `expected_source_contains`? Binary {0, 1}.

2. **Keyword match** — does the answer contain at least one substring from
   `must_include_any` AND every substring in `must_include_all`?
   Case-insensitive. Binary {0, 1}.

3. **LLM-judge faithfulness** — same 0/1/2 scoring used in the BEIR
   generation ablation (`src/answer_eval.py`).

For abstention questions (`should_abstain: true`), we flip the keyword check
so we only credit the system when the answer indicates "I don't know".

Aggregate metrics: per-question CSV plus a small summary printed to stdout
and saved as `uia_eval_summary.csv`.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import yaml

from .answer_eval import score_faithfulness
from .generator import OllamaGenerator
from .pipeline import RagPipeline, RagAnswer
from .retriever import Retrieved


@dataclass
class EvalItem:
    id: str
    question: str
    reference_answer: str
    must_include_any: List[str]
    must_include_all: List[str]
    expected_source_contains: List[str]
    should_abstain: bool


def load_eval_set(path: Path) -> List[EvalItem]:
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    items: List[EvalItem] = []
    for entry in raw:
        items.append(EvalItem(
            id=entry["id"],
            question=entry["question"],
            reference_answer=str(entry.get("reference_answer", "")).strip(),
            must_include_any=[s.lower() for s in (entry.get("must_include_any") or [])],
            must_include_all=[s.lower() for s in (entry.get("must_include_all") or [])],
            expected_source_contains=[s.lower() for s in (entry.get("expected_source_contains") or [])],
            should_abstain=bool(entry.get("should_abstain", False)),
        ))
    return items


# --- Scoring primitives ------------------------------------------------


def source_recall(retrieved: List[Retrieved], expected: List[str]) -> Optional[int]:
    """1 if any retrieved chunk's URL or title contains an expected substring.

    Returns None when the eval item has no source expectations (e.g. for
    abstention questions); the metric is undefined there.
    """
    if not expected:
        return None
    for r in retrieved:
        haystack = " ".join([
            str(r.chunk.metadata.get("url", "")),
            str(r.chunk.metadata.get("title", "")),
            str(r.chunk.source_id),
        ]).lower()
        if any(sub in haystack for sub in expected):
            return 1
    return 0


def keyword_match(answer: str, item: EvalItem) -> int:
    """1 if the answer satisfies the keyword constraints for the item.

    For non-abstention items: must contain >=1 from `must_include_any` AND
    every substring in `must_include_all`. For abstention items: must
    contain >=1 from `must_include_any` (which lists "don't know"-style
    phrases).
    """
    a = answer.lower()
    any_ok = (not item.must_include_any) or any(s in a for s in item.must_include_any)
    all_ok = all(s in a for s in item.must_include_all)
    return int(any_ok and all_ok)


# --- Driver ------------------------------------------------------------


@dataclass
class EvalRow:
    id: str
    question: str
    answer: str
    source_recall: Optional[int]
    keyword_match: int
    judge_score: Optional[int]
    abstain_expected: bool


def run_eval(
    pipeline: RagPipeline,
    items: List[EvalItem],
    judge: Optional[OllamaGenerator],
) -> List[EvalRow]:
    rows: List[EvalRow] = []
    for item in items:
        ans: RagAnswer = pipeline.answer(item.question)
        sr = source_recall(ans.retrieved, item.expected_source_contains)
        km = keyword_match(ans.answer, item)
        js = score_faithfulness(judge, item.question, ans.answer, ans.retrieved) if judge else None
        rows.append(EvalRow(
            id=item.id, question=item.question, answer=ans.answer,
            source_recall=sr, keyword_match=km, judge_score=js,
            abstain_expected=item.should_abstain,
        ))
    return rows


def summarize(rows: List[EvalRow]) -> Dict[str, float]:
    """Compute aggregate metrics. Abstention vs answerable are reported separately."""
    answerable = [r for r in rows if not r.abstain_expected]
    abstain = [r for r in rows if r.abstain_expected]

    def _avg(values, default=float("nan")):
        clean = [v for v in values if v is not None]
        return sum(clean) / len(clean) if clean else default

    summary: Dict[str, float] = {}
    summary["n_total"] = float(len(rows))
    summary["n_answerable"] = float(len(answerable))
    summary["n_abstain"] = float(len(abstain))

    if answerable:
        summary["answerable_source_recall"] = _avg([r.source_recall for r in answerable])
        summary["answerable_keyword_match"] = _avg([r.keyword_match for r in answerable])
        summary["answerable_judge_mean"]   = _avg([r.judge_score   for r in answerable])
    if abstain:
        # On abstention rows, "keyword_match" already encodes "did the model
        # correctly say it doesn't know".
        summary["abstain_correct_rate"] = _avg([r.keyword_match for r in abstain])

    return summary


def write_results(rows: List[EvalRow], summary: Dict[str, float], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    per_q = out_dir / "uia_eval_per_question.csv"
    with open(per_q, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "abstain_expected", "source_recall",
                    "keyword_match", "judge_score", "question", "answer"])
        for r in rows:
            w.writerow([r.id, int(r.abstain_expected),
                        "" if r.source_recall is None else r.source_recall,
                        r.keyword_match,
                        "" if r.judge_score is None else r.judge_score,
                        r.question.replace("\n", " "),
                        r.answer.replace("\n", " ")])
    summary_csv = out_dir / "uia_eval_summary.csv"
    with open(summary_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["metric", "value"])
        for k, v in summary.items():
            w.writerow([k, f"{v:.4f}" if isinstance(v, float) else v])
