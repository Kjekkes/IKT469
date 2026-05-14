"""Download BEIR/<dataset>, subsample for CPU, build a FAISS index.

Outputs:
    data/beir/<dataset>/                <- raw BEIR download (full dataset)
    data/beir/subsample_<dataset>/      <- our deterministic subset (corpus, queries, qrels)
    indexes/beir_<dataset>/             <- FAISS index built from the subset

Run:
    python scripts/01_build_beir_index.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.beir_eval import (
    corpus_to_chunks,
    load_beir,
    subsample,
)
from src.pipeline import RagPipeline, load_config


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    config = load_config(repo_root / "config.yaml")

    dataset = config["beir"]["dataset"]
    n_corpus = config["beir"]["corpus_subset_size"]
    n_queries = config["beir"]["query_subset_size"]
    beir_dir = repo_root / config["paths"]["beir_dir"]
    subset_dir = beir_dir / f"subsample_{dataset}"
    index_dir = repo_root / config["paths"]["index_dir"] / f"beir_{dataset}"

    print(f"[1/4] Loading BEIR/{dataset} (downloads ~1-2 GB the first time)...")
    t0 = time.time()
    corpus, queries, qrels = load_beir(dataset, beir_dir, split="test")
    print(f"      Full corpus={len(corpus)}  queries={len(queries)}  ({time.time()-t0:.1f}s)")

    print(f"[2/4] Subsampling to corpus<={n_corpus}, queries<={n_queries}...")
    sub = subsample(corpus, queries, qrels,
                    n_queries=n_queries, n_corpus=n_corpus, seed=42)
    print(f"      {sub.stats()}")
    sub.save(subset_dir)
    print(f"      Saved subset to {subset_dir}")

    print("[3/4] Embedding subsampled corpus and building FAISS index...")
    t0 = time.time()
    pipeline = RagPipeline(config)
    chunks = corpus_to_chunks(sub.corpus)
    pipeline.retriever.build(chunks)
    print(f"      Indexed {len(chunks)} passages ({time.time()-t0:.1f}s)")

    print(f"[4/4] Saving index to {index_dir}...")
    pipeline.save_index(index_dir)
    print("      Done.")


if __name__ == "__main__":
    main()
