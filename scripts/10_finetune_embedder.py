"""Fine-tune the MiniLM bi-encoder on a held-out BEIR/NQ training subset.

Uses MultipleNegativesRankingLoss (the standard contrastive objective for
dense retrieval). The 200 queries used by our eval subset are excluded
from training pairs, so the comparison in the retrieval ablation is fair.

Outputs:
    models/<finetune.output_subdir>/

Run:
    python scripts/10_finetune_embedder.py

CPU runtime: roughly 30-60 minutes for the default 5000 pairs / 2 epochs.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.beir_eval import load_beir, subsample
from src.finetune import build_training_pairs, finetune_embedder
from src.pipeline import load_config


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    config = load_config(repo_root / "config.yaml")

    dataset = config["beir"]["dataset"]
    beir_dir = repo_root / config["paths"]["beir_dir"]
    out_dir = repo_root / "models" / config["finetune"]["output_subdir"]
    base_model = config["embedding"]["model_name"]

    print(f"[1/4] Loading BEIR/{dataset} (test split)...")
    corpus, queries, qrels = load_beir(dataset, beir_dir, split="test")
    print(f"      corpus={len(corpus)}  queries={len(queries)}  qrel_rows="
          f"{sum(len(v) for v in qrels.values())}")

    print("[2/4] Identifying held-out eval queries (same seed as 01_build_beir_index.py)...")
    eval_sub = subsample(
        corpus, queries, qrels,
        n_queries=config["beir"]["query_subset_size"],
        n_corpus=config["beir"]["corpus_subset_size"],
        seed=42,
    )
    eval_qids = set(eval_sub.queries.keys())
    print(f"      excluding {len(eval_qids)} queries from training pool")

    print(f"[3/4] Building training pairs (target {config['finetune']['max_pairs']})...")
    pairs = build_training_pairs(
        corpus=corpus, queries=queries, qrels=qrels,
        exclude_qids=eval_qids,
        max_pairs=config["finetune"]["max_pairs"],
        seed=43,
    )
    print(f"      built {len(pairs)} (query, positive_passage) pairs")
    if not pairs:
        print("      No pairs to train on. Aborting.")
        sys.exit(1)

    print(f"[4/4] Fine-tuning {base_model}")
    print(f"      epochs={config['finetune']['epochs']}  "
          f"batch_size={config['finetune']['batch_size']}  "
          f"warmup={config['finetune']['warmup_steps']}")
    print(f"      output: {out_dir}")
    print(f"      (CPU runtime: ~30-60 min for the default settings)\n")
    t0 = time.time()
    finetune_embedder(
        base_model=base_model,
        pairs=pairs,
        output_dir=out_dir,
        epochs=config["finetune"]["epochs"],
        batch_size=config["finetune"]["batch_size"],
        warmup_steps=config["finetune"]["warmup_steps"],
    )
    print(f"\nDone in {(time.time()-t0)/60:.1f} min. Saved to {out_dir}")
    print("\nThe retrieval ablation (script 06) already includes a row that uses "
          "this fine-tuned model. Re-run it to see the lift.")


if __name__ == "__main__":
    main()
