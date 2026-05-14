"""Evaluate the UiA chatbot against the hand-written eval set.

Reads:
    eval/uia_eval.yaml
    indexes/uia/ (built by 04_build_uia_index.py)

Writes:
    report/figures/uia_eval_per_question.csv
    report/figures/uia_eval_summary.csv

Run:
    python scripts/09_eval_uia.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.generator import OllamaGenerator
from src.pipeline import RagPipeline, load_config
from src.uia_eval import load_eval_set, run_eval, summarize, write_results


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    config = load_config(repo_root / "config.yaml")

    index_dir = repo_root / config["paths"]["index_dir"] / "uia"
    eval_path = repo_root / "eval" / "uia_eval.yaml"
    out_dir = repo_root / "report" / "figures"

    if not index_dir.exists():
        print(f"No UiA index at {index_dir}. Run 04_build_uia_index.py first.")
        sys.exit(1)
    if not eval_path.exists():
        print(f"No eval set at {eval_path}.")
        sys.exit(1)

    print(f"Loading eval set from {eval_path}...")
    items = load_eval_set(eval_path)
    print(f"  {len(items)} items "
          f"({sum(1 for i in items if i.should_abstain)} abstention)")

    print("Loading pipeline + index...")
    pipeline = RagPipeline(config)
    pipeline.load_index(index_dir)

    print("Loading LLM-judge (same Ollama model as generator)...")
    judge = OllamaGenerator(
        model=config["generator"]["model"],
        temperature=0.0,
        num_ctx=config["generator"]["num_ctx"],
    )

    print(f"Running eval ({len(items)} questions, ~10-30s each on CPU)...\n")
    rows = run_eval(pipeline, items, judge=judge)

    summary = summarize(rows)
    write_results(rows, summary, out_dir)

    print("\n=== UiA eval summary ===")
    for k, v in summary.items():
        if isinstance(v, float):
            print(f"  {k:30s} {v:.4f}")
        else:
            print(f"  {k:30s} {v}")
    print(f"\nWrote per-question CSV and summary CSV to {out_dir}")


if __name__ == "__main__":
    main()
