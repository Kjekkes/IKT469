"""Retrieval ablation: chunk size x reranker.

For each (chunk_size, rerank) combination:
    1. Build (or reuse a cached) FAISS index on the BEIR subset.
    2. Run retrieval over the held-out queries.
    3. Compute Recall, nDCG, MRR via BEIR's official evaluator.

Results land in long format so they're trivial to pivot/plot:

    ablation, chunk_size, rerank, metric, value
    chunk_size, none, false, NDCG@10, 0.4123
    chunk_size, 256,  false, NDCG@10, 0.4015
    rerank,     none, true,  NDCG@10, 0.4480
    ...

Run:
    python scripts/06_ablate_retrieval.py
"""
from __future__ import annotations

import csv
import sys
import time
from pathlib import Path
from typing import List

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.ablations import (
    RetrievalCfg,
    build_index_for_chunk_size,
    run_retrieval_ablation,
    with_overrides,
)
from src.beir_eval import Subsample, evaluate_retrieval
from src.pipeline import RagPipeline, load_config
from src.reranker import CrossEncoderReranker


# Edit this matrix to change which ablations run. Order matters for cache:
# entries with the same (embedding_model, chunk_size) reuse the in-memory
# pipeline + index. Group rows that share an embedding_model together so we
# only load each encoder once.
ABLATIONS: List[RetrievalCfg] = [
    RetrievalCfg(name="baseline (no split)",  chunk_size=None, rerank=False),
    RetrievalCfg(name="chunk_128",            chunk_size=128,  rerank=False),
    RetrievalCfg(name="chunk_256",            chunk_size=256,  rerank=False),
    RetrievalCfg(name="chunk_512",            chunk_size=512,  rerank=False),
    RetrievalCfg(name="baseline + rerank",    chunk_size=None, rerank=True),
    RetrievalCfg(name="chunk_256 + rerank",   chunk_size=256,  rerank=True),
    # The fine-tuned encoder. Only included if scripts/10_finetune_embedder.py
    # has been run; otherwise the row is skipped at runtime with a warning.
    RetrievalCfg(name="MiniLM-FT (no split)", chunk_size=None, rerank=False,
                 embedding_model="models/minilm-nq-ft"),
]


def _resolve_embedding_model(repo_root: Path, name: str) -> str:
    """Treat the value as a local path if it exists; otherwise pass through as a HF id."""
    candidate = (repo_root / name).resolve()
    return str(candidate) if candidate.exists() else name


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    config = load_config(repo_root / "config.yaml")

    dataset = config["beir"]["dataset"]
    k_values = config["beir"]["metrics_k"]
    top_k_docs = max(k_values)
    subset_dir = repo_root / config["paths"]["beir_dir"] / f"subsample_{dataset}"
    out_dir = repo_root / "report" / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    if not subset_dir.exists():
        print(f"Missing subset at {subset_dir}. Run scripts/01_build_beir_index.py first.")
        sys.exit(1)

    print(f"Loading BEIR subset from {subset_dir}...")
    sub = Subsample.load(subset_dir)
    print(f"  {sub.stats()}")

    pipeline: RagPipeline | None = None
    reranker = None  # lazy: only loaded when first ablation needs it

    rows = []
    last_chunk_size = "__sentinel__"      # avoids None == None false-positive
    last_embedding_model = "__sentinel__"
    t_total = time.time()

    for cfg in ABLATIONS:
        print(f"\n--- Ablation: {cfg.name} ---")

        # Resolve which embedder this row needs (default to config).
        embed_id = cfg.embedding_model or config["embedding"]["model_name"]
        embed_resolved = _resolve_embedding_model(repo_root, embed_id)

        # Skip the fine-tuned row gracefully if the model isn't on disk yet.
        if cfg.embedding_model and not Path(embed_resolved).exists() \
                and "/" in cfg.embedding_model:
            print(f"  (skip) {cfg.embedding_model} not found. "
                  f"Run scripts/10_finetune_embedder.py to produce it.")
            continue

        # Swap the pipeline whenever the encoder changes; this also forces
        # an index rebuild because vectors from the old encoder don't apply.
        if embed_resolved != last_embedding_model:
            print(f"  loading embedder: {embed_id}")
            cfg_for_pipeline = with_overrides(
                config, **{"embedding.model_name": embed_resolved}
            )
            pipeline = RagPipeline(cfg_for_pipeline)
            last_embedding_model = embed_resolved
            last_chunk_size = "__sentinel__"  # invalidate cached index

        # Index reuse: only rebuild when chunk_size changes (within the same encoder).
        if cfg.chunk_size != last_chunk_size:
            t0 = time.time()
            n_chunks = build_index_for_chunk_size(
                pipeline, sub.corpus,
                chunk_size=cfg.chunk_size,
                overlap=config["chunking"]["overlap"],
            )
            print(f"  built index: {n_chunks} chunks ({time.time()-t0:.1f}s)")
            last_chunk_size = cfg.chunk_size
        else:
            print("  reusing in-memory index from previous ablation")

        if cfg.rerank and reranker is None:
            print(f"  loading cross-encoder: {config['retrieval']['reranker_model']}")
            reranker = CrossEncoderReranker(config["retrieval"]["reranker_model"])

        t0 = time.time()
        results = run_retrieval_ablation(
            pipeline, sub.queries, cfg,
            top_k_docs=top_k_docs, reranker=reranker,
        )
        metrics = evaluate_retrieval(sub.qrels, results, k_values=k_values)
        print(f"  evaluated in {time.time()-t0:.1f}s")
        for m, v in sorted(metrics.items()):
            print(f"    {m:20s} {v:.4f}")

        for metric, value in metrics.items():
            rows.append({
                "ablation":        cfg.name,
                "chunk_size":      "none" if cfg.chunk_size is None else cfg.chunk_size,
                "rerank":          str(cfg.rerank).lower(),
                "embedding_model": cfg.embedding_model or "default",
                "metric":          metric,
                "value":           f"{value:.4f}",
            })

    out_csv = out_dir / f"ablate_retrieval_{dataset}.csv"
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "ablation", "chunk_size", "rerank", "embedding_model", "metric", "value",
        ])
        w.writeheader()
        w.writerows(rows)

    print(f"\nTotal time: {time.time()-t_total:.1f}s")
    print(f"Wrote {out_csv}")


if __name__ == "__main__":
    main()
