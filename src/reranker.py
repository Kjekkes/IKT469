"""Cross-encoder reranker.

Bi-encoders (MiniLM sentence-transformers) are fast but lose precision in
the top-k. Cross-encoders score each (query, passage) pair jointly and
typically lift NDCG@10 by 5-15 points on BEIR tasks. Cost: O(top_n) model
calls per query, but ms-marco-MiniLM-L-6-v2 is small enough to do this on
CPU at ~10 ms/pair.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

from sentence_transformers import CrossEncoder

from .retriever import Retrieved


class CrossEncoderReranker:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.model = CrossEncoder(model_name)

    def rerank(self, query: str, candidates: List[Retrieved], top_k: int) -> List[Retrieved]:
        if not candidates:
            return []
        pairs = [(query, c.chunk.text) for c in candidates]
        scores = self.model.predict(pairs, show_progress_bar=False)
        # Replace bi-encoder scores with cross-encoder scores, re-sort, re-rank.
        scored = list(zip(candidates, scores))
        scored.sort(key=lambda x: -float(x[1]))
        out: List[Retrieved] = []
        for new_rank, (cand, s) in enumerate(scored[:top_k]):
            out.append(Retrieved(chunk=cand.chunk, score=float(s), rank=new_rank))
        return out
