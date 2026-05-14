"""Evaluate the BEIR index built by 01_build_beir_index.py.

Outputs:
    report/figures/beir_<dataset>_metrics.csv    <- one row per metric
    report/figures/beir_<dataset>_results.json   <- per-query top-k passages

Run:
    python scripts/02_eval_beir.py
"""
from __future__ import annotations

import csv
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.beir_eval import Subsample, evaluate_retrieval, run_retrieval
from src.pipeline import RagPipeline, load_config


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    config = load_config(repo_root / "config.yaml")

    dataset = config["beir"]["dataset"]
    k_values = config["beir"]["metrics_k"]
    subset_dir = repo_root / config["paths"]["beir_dir"] / f"subsample_{dataset}"
    index_dir = repo_root / config["paths"]["index_dir"] / f"beir_{dataset}"
    out_dir = repo_root / "report" / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    if not subset_dir.exists() or not index_dir.exists():
        print(f"Missing artifacts. Run scripts/01_build_beir_index.py first.\n"
              f"  subset_dir = {subset_dir}\n  index_dir  = {index_dir}")
        sys.exit(1)

    print(f"[1/4] Loading subset from {subset_dir}...")
    sub = Subsample.load(subset_dir)
    print(f"      {sub.stats()}")

    print(f"[2/4] Loading FAISS index from {index_dir}...")
    pipeline = RagPipeline(config)
    pipeline.load_index(index_dir)

    print(f"[3/4] Running retrieval over {len(sub.queries)} queries (top_k={max(k_values)})...")
    t0 = time.time()
    results = run_retrieval(pipeline, sub.queries, top_k=max(k_values))
    print(f"      Done ({time.time()-t0:.1f}s).")

    print("[4/4] Computing BEIR metrics...")
    metrics = evaluate_retrieval(sub.qrels, results, k_values=k_values)

    metrics_path = out_dir / f"beir_{dataset}_metrics.csv"
    with open(metrics_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["metric", "value"])
        for k, v in sorted(metrics.items()):
            w.writerow([k, f"{v:.4f}"])

    results_path = out_dir / f"beir_{dataset}_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    print("\n=== Retrieval metrics ===")
    for k, v in sorted(metrics.items()):
        print(f"  {k:20s} {v:.4f}")
    print(f"\nWrote {metrics_path}")
    print(f"Wrote {results_path}")


if __name__ == "__main__":
    main()
