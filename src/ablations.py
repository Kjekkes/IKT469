"""Shared ablation helpers.

Two things the per-ablation scripts both need:

1. **Indexing variants** — build the FAISS index either with one chunk per
   passage (BEIR baseline) or with the recursive splitter at a given
   `chunk_size`. The split variant produces multiple chunks per source doc.

2. **Doc-level aggregation** — when there are multiple chunks per doc, BEIR
   metrics still need {qid: {doc_id: score}}. We max-pool chunk scores per
   source_id, which is the standard approach (e.g. Khattab & Zaharia 2020).

Plus a tiny `RetrievalCfg` dataclass so each ablation row carries its own
parameters as data, not implicit globals.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from tqdm import tqdm

from .beir_eval import Corpus, Queries, corpus_to_chunks
from .chunking import Chunk, chunk_documents
from .pipeline import RagPipeline
from .reranker import CrossEncoderReranker
from .retriever import Retrieved


# --- Indexing -----------------------------------------------------------


def corpus_to_split_chunks(corpus: Corpus, chunk_size: int, overlap: int) -> List[Chunk]:
    """Split BEIR passages with the recursive char splitter.

    Used for ablations where we want to study the effect of chunk size.
    Each output chunk keeps `source_id` = the original passage id, so the
    BEIR qrels still apply after max-pool aggregation.
    """
    docs = []
    for did, doc in corpus.items():
        title = doc.get("title", "") or ""
        body = doc.get("text", "") or ""
        text = f"{title}. {body}".strip(". ").strip()
        docs.append({"id": did, "text": text, "title": title})
    return chunk_documents(docs, chunk_size_tokens=chunk_size, overlap_tokens=overlap)


def build_index_for_chunk_size(
    pipeline: RagPipeline,
    corpus: Corpus,
    chunk_size: Optional[int],
    overlap: int,
) -> int:
    """Build the in-memory FAISS index for an ablation run.

    chunk_size=None  -> one chunk per passage (baseline).
    chunk_size=int   -> split with the recursive splitter.

    Returns the number of chunks indexed.
    """
    if chunk_size is None:
        chunks = corpus_to_chunks(corpus)
    else:
        chunks = corpus_to_split_chunks(corpus, chunk_size=chunk_size, overlap=overlap)
    pipeline.retriever.build(chunks)
    return len(chunks)


# --- Doc-level aggregation ---------------------------------------------


def chunks_to_doc_scores(hits: List[Retrieved]) -> Dict[str, float]:
    """Max-pool chunk scores back to doc-level scores.

    {doc_id: best_score_among_its_chunks}. Order is not preserved here;
    BEIR's evaluator only needs the dict.
    """
    doc_scores: Dict[str, float] = {}
    for h in hits:
        sid = h.chunk.source_id
        score = float(h.score)
        if sid not in doc_scores or score > doc_scores[sid]:
            doc_scores[sid] = score
    return doc_scores


# --- Ablation config ---------------------------------------------------


@dataclass
class RetrievalCfg:
    """One row of the retrieval-ablation matrix."""
    name: str
    chunk_size: Optional[int]   # None means "no further splitting"
    overlap: int = 32
    rerank: bool = False
    rerank_pool: int = 50       # how many bi-encoder hits to feed to the cross-encoder
    embedding_model: Optional[str] = None  # None -> use config default. May be
                                           # a HuggingFace id or a local path.


@dataclass
class GenerationCfg:
    """One row of the generation-ablation matrix."""
    name: str
    top_k: int
    prompt_variant: str         # "grounded" | "vanilla"
    n_eval_queries: int = 20    # cap for CPU; LLM calls dominate runtime


# --- Retrieval-ablation runner -----------------------------------------


def run_retrieval_ablation(
    pipeline: RagPipeline,
    queries: Queries,
    cfg: RetrievalCfg,
    top_k_docs: int,
    reranker: Optional[CrossEncoderReranker] = None,
) -> Dict[str, Dict[str, float]]:
    """Returns BEIR-shaped results for one ablation row.

    The pipeline's index must already be built for this `cfg.chunk_size`.
    Reranker is optional and reused across calls (loading it is the slow
    part, ~3 s; predicting is fast).
    """
    # When passages are split into many chunks, retrieve more so we have
    # enough unique docs after max-pool. 5x is safe; 3x is usually fine.
    chunk_oversample = 1 if cfg.chunk_size is None else 5
    raw_top_k = top_k_docs * chunk_oversample
    if cfg.rerank:
        raw_top_k = max(raw_top_k, cfg.rerank_pool)

    results: Dict[str, Dict[str, float]] = {}
    for qid, q in tqdm(queries.items(), desc=f"retrieving [{cfg.name}]"):
        hits = pipeline.retrieve(q, top_k=raw_top_k)
        if cfg.rerank:
            assert reranker is not None, "rerank=True requires a reranker"
            hits = reranker.rerank(q, hits, top_k=top_k_docs * 3)
        results[qid] = chunks_to_doc_scores(hits)
    return results


# --- Config helpers ----------------------------------------------------


def with_overrides(config: dict, **overrides) -> dict:
    """Return a deep-copied config with `overrides` merged into the top level.

    Use case: ablation passes need to nudge `chunking.size` or `retrieval.top_k`
    without mutating the global config. We accept dotted keys for nesting.
    """
    new = copy.deepcopy(config)
    for dotted_key, value in overrides.items():
        d = new
        parts = dotted_key.split(".")
        for p in parts[:-1]:
            d = d.setdefault(p, {})
        d[parts[-1]] = value
    return new
