"""FAISS-backed retriever.

We keep the chunk metadata in a parallel list rather than inside FAISS so
the index file stays small and language-agnostic. Persistence is just the
.faiss file plus a pickle of the chunk list.
"""
from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import faiss
import numpy as np

from .chunking import Chunk
from .embed import Embedder


@dataclass
class Retrieved:
    chunk: Chunk
    score: float
    rank: int


class FaissRetriever:
    def __init__(self, embedder: Embedder):
        self.embedder = embedder
        self.index: faiss.Index | None = None
        self.chunks: List[Chunk] = []

    def build(self, chunks: List[Chunk]) -> None:
        if not chunks:
            raise ValueError("Cannot build a retriever from an empty chunk list.")
        texts = [c.text for c in chunks]
        vecs = self.embedder.embed_texts(texts)
        self.index = faiss.IndexFlatIP(self.embedder.dim)
        self.index.add(vecs)
        self.chunks = chunks

    def search(self, query: str, top_k: int = 5) -> List[Retrieved]:
        if self.index is None:
            raise RuntimeError("Retriever has no index. Call build() or load() first.")
        q = self.embedder.embed_query(query).reshape(1, -1)
        scores, idxs = self.index.search(q, top_k)
        results: List[Retrieved] = []
        for rank, (i, s) in enumerate(zip(idxs[0], scores[0])):
            if i == -1:
                continue
            results.append(Retrieved(chunk=self.chunks[i], score=float(s), rank=rank))
        return results

    def save(self, dir_path: str | Path) -> None:
        dir_path = Path(dir_path)
        dir_path.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(dir_path / "index.faiss"))
        with open(dir_path / "chunks.pkl", "wb") as f:
            pickle.dump(self.chunks, f)

    def load(self, dir_path: str | Path) -> None:
        dir_path = Path(dir_path)
        self.index = faiss.read_index(str(dir_path / "index.faiss"))
        with open(dir_path / "chunks.pkl", "rb") as f:
            self.chunks = pickle.load(f)
