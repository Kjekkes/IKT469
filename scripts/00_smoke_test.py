"""Smoke test: 5-document toy corpus through the full pipeline.

Run this first to verify everything wires up. It exercises chunking ->
embedding -> FAISS -> retrieval. Generation is optional (only runs if
Ollama is reachable), so the test still passes on a fresh machine.

    python scripts/00_smoke_test.py
"""
from __future__ import annotations

import sys
from pathlib import Path

# Allow `python scripts/...` from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.pipeline import RagPipeline, load_config


TOY_DOCS = [
    {"id": "d1", "title": "UiA",
     "text": "The University of Agder (UiA) is a public university in southern Norway, "
             "with campuses in Kristiansand and Grimstad."},
    {"id": "d2", "title": "IKT469",
     "text": "IKT469 is a deep learning course at the University of Agder. "
             "It covers neural networks, transformers, and modern architectures."},
    {"id": "d3", "title": "RAG",
     "text": "Retrieval-augmented generation (RAG) combines a retriever over a document "
             "store with a generator language model to answer questions with citations."},
    {"id": "d4", "title": "FAISS",
     "text": "FAISS is a library for efficient similarity search and clustering of dense "
             "vectors. It supports flat indexes for exact search and IVF/HNSW for ANN."},
    {"id": "d5", "title": "Ollama",
     "text": "Ollama runs open-weight LLMs locally. Models are pulled with `ollama pull` "
             "and served via an HTTP API on localhost:11434."},
]


def main() -> None:
    config = load_config(Path(__file__).resolve().parents[1] / "config.yaml")
    pipeline = RagPipeline(config)

    print("[1/3] Building toy index...")
    chunks = pipeline.build_index_from_documents(TOY_DOCS)
    print(f"      Indexed {len(chunks)} chunks from {len(TOY_DOCS)} docs.")

    print("[2/3] Retrieval check...")
    question = "What does FAISS do?"
    hits = pipeline.retrieve(question, top_k=3)
    for h in hits:
        print(f"      rank={h.rank}  score={h.score:.3f}  src={h.chunk.source_id}  "
              f"text={h.chunk.text[:60]!r}")
    assert hits[0].chunk.source_id == "d4", "Expected FAISS doc as top hit."
    print("      Retrieval OK.")

    print("[3/3] Generation check (optional, needs Ollama)...")
    try:
        ans = pipeline.answer(question)
        print(f"      Q: {ans.question}")
        print(f"      A: {ans.answer}")
        print("      Generation OK.")
    except Exception as e:
        print(f"      Skipped generation: {e}")
        print("      (That's fine — pipeline still works without Ollama.)")


if __name__ == "__main__":
    main()
