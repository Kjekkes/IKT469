"""Generation ablation: top_k x prompt variant.

Slow because every cell requires a generator call per query. We default to
20 queries per cell to keep total runtime around 30-60 minutes on CPU. Bump
`n_eval_queries` in the matrix below if you can afford more.

Each (top_k, prompt_variant) combination is evaluated by:
    1. Generating an answer with our RagPipeline.
    2. Scoring faithfulness with an LLM-as-judge (same Ollama model).
    3. Recording avg latency and avg faithfulness.

Output:
    report/figures/ablate_generation_<dataset>.csv

Run:
    python scripts/07_ablate_generation.py
"""
from __future__ import annotations

import csv
import sys
import time
from pathlib import Path
from typing import List

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.ablations import GenerationCfg, build_index_for_chunk_size
from src.answer_eval import score_faithfulness
from src.beir_eval import Subsample
from src.generator import OllamaGenerator
from src.pipeline import RagPipeline, load_config


ABLATIONS: List[GenerationCfg] = [
    GenerationCfg(name="grounded_k1",   top_k=1, prompt_variant="grounded"),
    GenerationCfg(name="grounded_k3",   top_k=3, prompt_variant="grounded"),
    GenerationCfg(name="grounded_k5",   top_k=5, prompt_variant="grounded"),
    GenerationCfg(name="grounded_k10",  top_k=10, prompt_variant="grounded"),
    GenerationCfg(name="vanilla_k5",    top_k=5, prompt_variant="vanilla"),
]


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    config = load_config(repo_root / "config.yaml")

    dataset = config["beir"]["dataset"]
    subset_dir = repo_root / config["paths"]["beir_dir"] / f"subsample_{dataset}"
    out_dir = repo_root / "report" / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    if not subset_dir.exists():
        print(f"Missing subset at {subset_dir}. Run scripts/01_build_beir_index.py first.")
        sys.exit(1)

    print(f"Loading BEIR subset from {subset_dir}...")
    sub = Subsample.load(subset_dir)
    print(f"  {sub.stats()}")

    # All ablations share the baseline (no-split) index — chunk-size effects
    # belong to the retrieval ablation, not this one.
    pipeline = RagPipeline(config)
    print("Building baseline index (one chunk per passage)...")
    n = build_index_for_chunk_size(pipeline, sub.corpus, chunk_size=None,
                                   overlap=config["chunking"]["overlap"])
    print(f"  indexed {n} passages")

    # The judge uses the same Ollama model as the generator. For a stronger
    # judgment swap this for a larger model; for speed keep them equal.
    judge = OllamaGenerator(
        model=config["generator"]["model"],
        temperature=0.0,
        num_ctx=config["generator"]["num_ctx"],
    )

    qids = list(sub.queries.keys())
    rows = []
    t_total = time.time()

    for cfg in ABLATIONS:
        print(f"\n--- Ablation: {cfg.name} (top_k={cfg.top_k}, prompt={cfg.prompt_variant}) ---")
        eval_qids = qids[: cfg.n_eval_queries]
        scores: List[int] = []
        latencies: List[float] = []

        for i, qid in enumerate(eval_qids, 1):
            q = sub.queries[qid]
            t0 = time.time()
            ans = pipeline.answer_with(
                question=q,
                top_k=cfg.top_k,
                prompt_variant=cfg.prompt_variant,
            )
            dt = time.time() - t0
            latencies.append(dt)

            score = score_faithfulness(judge, q, ans.answer, ans.retrieved)
            if score is not None:
                scores.append(score)

            short_a = ans.answer.replace("\n", " ")[:80]
            print(f"  [{i:2d}/{len(eval_qids)}] dt={dt:5.1f}s score={score}  Q: {q[:60]!r}")
            print(f"          A: {short_a!r}")

        avg_score = sum(scores) / len(scores) if scores else float("nan")
        avg_lat = sum(latencies) / len(latencies) if latencies else float("nan")
        print(f"  -> avg faithfulness={avg_score:.2f}  avg latency={avg_lat:.1f}s  n={len(scores)}")

        rows.append({
            "ablation": cfg.name,
            "top_k": cfg.top_k,
            "prompt_variant": cfg.prompt_variant,
            "n_eval_queries": cfg.n_eval_queries,
            "n_scored": len(scores),
            "avg_faithfulness": f"{avg_score:.4f}",
            "avg_latency_s":    f"{avg_lat:.2f}",
        })

    out_csv = out_dir / f"ablate_generation_{dataset}.csv"
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print(f"\nTotal time: {(time.time()-t_total)/60:.1f} min")
    print(f"Wrote {out_csv}")


if __name__ == "__main__":
    main()
