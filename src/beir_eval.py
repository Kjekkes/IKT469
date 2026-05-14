"""BEIR/NQ benchmark utilities.

Three concerns live here:

1. **Loading** — download a BEIR dataset on demand and load corpus/queries/qrels
   via the official `beir` package.
2. **Subsampling** — keep CPU-friendly while preserving evaluation validity.
   We sample N queries, keep every doc referenced by their qrels, then top
   the corpus up to a target size with random distractors.
3. **Evaluation** — run our `RagPipeline` retriever over the held-out queries,
   aggregate chunk hits to document-level scores, and compute Recall / nDCG /
   MRR via `beir.retrieval.evaluation.EvaluateRetrieval`.

Why not use `beir.retrieval.search.dense.DenseRetrievalExactSearch`? Because
we want the *same* retriever the chatbot uses — anything else makes the
benchmark numbers irrelevant to the deployed system.
"""
from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

from tqdm import tqdm

from .chunking import Chunk
from .pipeline import RagPipeline


# Type aliases for clarity. BEIR's GenericDataLoader returns these shapes.
Corpus = Dict[str, Dict[str, str]]   # {doc_id: {"title": str, "text": str}}
Queries = Dict[str, str]             # {qid: query_text}
Qrels = Dict[str, Dict[str, int]]    # {qid: {doc_id: relevance}}


# --- Loading ------------------------------------------------------------


def download_beir_dataset(name: str, base_dir: Path) -> Path:
    """Download a BEIR dataset zip into `base_dir/<name>/` if not already there.

    The official BEIR mirror is sometimes flaky; we try the canonical URL
    and surface the error rather than papering over it.
    """
    from beir import util as beir_util  # local import: heavy + optional

    target = base_dir / name
    if target.exists() and (target / "corpus.jsonl").exists():
        return target

    base_dir.mkdir(parents=True, exist_ok=True)
    url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{name}.zip"
    extracted = beir_util.download_and_unzip(url, str(base_dir))
    return Path(extracted)


def load_beir(name: str, base_dir: Path, split: str = "test") -> Tuple[Corpus, Queries, Qrels]:
    from beir.datasets.data_loader import GenericDataLoader

    data_path = download_beir_dataset(name, base_dir)
    corpus, queries, qrels = GenericDataLoader(data_folder=str(data_path)).load(split=split)
    return corpus, queries, qrels


# --- Subsampling --------------------------------------------------------


@dataclass
class Subsample:
    corpus: Corpus
    queries: Queries
    qrels: Qrels
    seed: int

    def stats(self) -> str:
        n_rel = sum(len(rels) for rels in self.qrels.values())
        return (f"corpus={len(self.corpus)}  queries={len(self.queries)}  "
                f"qrel_rows={n_rel}  seed={self.seed}")

    def save(self, dir_path: Path) -> None:
        dir_path.mkdir(parents=True, exist_ok=True)
        with open(dir_path / "corpus.jsonl", "w", encoding="utf-8") as f:
            for did, doc in self.corpus.items():
                f.write(json.dumps({"_id": did, **doc}) + "\n")
        with open(dir_path / "queries.jsonl", "w", encoding="utf-8") as f:
            for qid, q in self.queries.items():
                f.write(json.dumps({"_id": qid, "text": q}) + "\n")
        with open(dir_path / "qrels.json", "w", encoding="utf-8") as f:
            json.dump(self.qrels, f)
        with open(dir_path / "meta.json", "w", encoding="utf-8") as f:
            json.dump({"seed": self.seed,
                       "n_corpus": len(self.corpus),
                       "n_queries": len(self.queries)}, f, indent=2)

    @classmethod
    def load(cls, dir_path: Path) -> "Subsample":
        corpus: Corpus = {}
        with open(dir_path / "corpus.jsonl", "r", encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                did = rec.pop("_id")
                corpus[did] = rec
        queries: Queries = {}
        with open(dir_path / "queries.jsonl", "r", encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                queries[rec["_id"]] = rec["text"]
        with open(dir_path / "qrels.json", "r", encoding="utf-8") as f:
            qrels = json.load(f)
        with open(dir_path / "meta.json", "r", encoding="utf-8") as f:
            meta = json.load(f)
        return cls(corpus=corpus, queries=queries, qrels=qrels, seed=meta["seed"])


def subsample(
    corpus: Corpus,
    queries: Queries,
    qrels: Qrels,
    n_queries: int,
    n_corpus: int,
    seed: int = 42,
) -> Subsample:
    """Sample n_queries queries; keep every qrel-positive doc; pad to n_corpus."""
    rng = random.Random(seed)

    # 1. pick queries that actually have qrels (otherwise eval is undefined)
    qids_with_qrels = [qid for qid in queries if qid in qrels and qrels[qid]]
    rng.shuffle(qids_with_qrels)
    sampled_qids = qids_with_qrels[:n_queries]
    sampled_queries = {qid: queries[qid] for qid in sampled_qids}
    sampled_qrels = {qid: qrels[qid] for qid in sampled_qids}

    # 2. union of all positive doc ids — must keep these or metrics break
    must_keep_ids = set()
    for rels in sampled_qrels.values():
        must_keep_ids.update(rels.keys())
    must_keep_ids = {did for did in must_keep_ids if did in corpus}

    # 3. fill the rest with random distractors
    other_ids = [did for did in corpus if did not in must_keep_ids]
    rng.shuffle(other_ids)
    n_extra = max(0, n_corpus - len(must_keep_ids))
    distractor_ids = other_ids[:n_extra]

    final_ids = must_keep_ids | set(distractor_ids)
    sampled_corpus = {did: corpus[did] for did in final_ids}

    return Subsample(
        corpus=sampled_corpus,
        queries=sampled_queries,
        qrels=sampled_qrels,
        seed=seed,
    )


# --- Indexing ----------------------------------------------------------


def corpus_to_chunks(corpus: Corpus) -> List[Chunk]:
    """Convert a BEIR corpus to one Chunk per passage.

    BEIR passages are already at retrieval granularity, so we deliberately
    skip our recursive splitter — splitting them further would force a
    chunk-to-doc max-pool step that adds noise without gain.
    """
    chunks: List[Chunk] = []
    for did, doc in corpus.items():
        title = doc.get("title", "") or ""
        body = doc.get("text", "") or ""
        # Title prefix is the BEIR convention; helps embedding quality.
        text = f"{title}. {body}".strip(". ").strip()
        chunks.append(
            Chunk(text=text, source_id=did, chunk_id=0,
                  metadata={"title": title})
        )
    return chunks


# --- Evaluation --------------------------------------------------------


def run_retrieval(
    pipeline: RagPipeline,
    queries: Queries,
    top_k: int = 10,
) -> Dict[str, Dict[str, float]]:
    """Returns BEIR-shaped results: {qid: {doc_id: score}}.

    Because we index one chunk per passage, chunk-level scores already are
    doc-level scores. If you change `corpus_to_chunks` to actually split,
    add a max-pool over `chunk.source_id` here.
    """
    results: Dict[str, Dict[str, float]] = {}
    for qid, q in tqdm(queries.items(), desc="retrieving"):
        hits = pipeline.retrieve(q, top_k=top_k)
        results[qid] = {h.chunk.source_id: float(h.score) for h in hits}
    return results


def evaluate_retrieval(
    qrels: Qrels,
    results: Dict[str, Dict[str, float]],
    k_values: List[int],
) -> Dict[str, float]:
    """Wrapper around BEIR's official evaluator.

    Returns a single flat dict like {"NDCG@10": 0.42, "Recall@10": 0.71, ...}
    for easy CSV/table writing.
    """
    from beir.retrieval.evaluation import EvaluateRetrieval

    evaluator = EvaluateRetrieval()
    ndcg, _map, recall, precision = evaluator.evaluate(qrels, results, k_values=k_values)
    mrr = evaluator.evaluate_custom(qrels, results, k_values=k_values, metric="mrr")

    flat: Dict[str, float] = {}
    for d in (ndcg, _map, recall, precision, mrr):
        flat.update({k: float(v) for k, v in d.items()})
    return flat
