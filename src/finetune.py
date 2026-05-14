"""Contrastive fine-tuning of the bi-encoder.

We train the MiniLM bi-encoder on (query, positive_passage) pairs drawn
from BEIR/NQ. The objective is `MultipleNegativesRankingLoss`: for a batch
of N pairs we treat the other N-1 positives as in-batch negatives, which
is the standard setup for dense retrieval since DPR
(Karpukhin et al., 2020).

Methodological note: the eval subset (200 queries, fixed seed) is excluded
from the training pool, so train and test query sets are disjoint and the
ablation comparison is honest.
"""
from __future__ import annotations

import random
from pathlib import Path
from typing import Iterable, List, Set, Tuple

from sentence_transformers import InputExample, SentenceTransformer, losses
from torch.utils.data import DataLoader

from .beir_eval import Corpus, Qrels, Queries


def build_training_pairs(
    corpus: Corpus,
    queries: Queries,
    qrels: Qrels,
    exclude_qids: Set[str],
    max_pairs: int,
    seed: int = 43,
    min_passage_chars: int = 20,
) -> List[Tuple[str, str]]:
    """Build (query, positive_passage) pairs for contrastive training.

    For each (qid, doc_id, rel>0) in the qrels, pair up the query text and
    the (title-prefixed) passage text. Queries in `exclude_qids` are
    skipped so we never train on something we will later evaluate on.
    """
    rng = random.Random(seed)
    pairs: List[Tuple[str, str]] = []

    for qid, rels in qrels.items():
        if qid in exclude_qids or qid not in queries:
            continue
        q_text = queries[qid]
        for did, rel in rels.items():
            if rel <= 0 or did not in corpus:
                continue
            doc = corpus[did]
            title = (doc.get("title") or "").strip()
            body = (doc.get("text") or "").strip()
            p_text = (f"{title}. {body}" if title else body).strip(". ").strip()
            if len(p_text) >= min_passage_chars:
                pairs.append((q_text, p_text))

    rng.shuffle(pairs)
    return pairs[:max_pairs]


def finetune_embedder(
    base_model: str,
    pairs: Iterable[Tuple[str, str]],
    output_dir: Path,
    epochs: int = 2,
    batch_size: int = 16,
    warmup_steps: int = 100,
) -> None:
    """Train and save a fine-tuned SentenceTransformer.

    Uses the legacy `model.fit()` API which is stable across
    sentence-transformers 2.x and 3.x. We pass `use_amp=False` because
    Apple Silicon CPUs don't benefit from autocast.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    model = SentenceTransformer(base_model)

    examples = [InputExample(texts=[q, p]) for q, p in pairs]
    if not examples:
        raise ValueError("No training pairs were provided.")
    dataloader = DataLoader(examples, shuffle=True, batch_size=batch_size)
    loss = losses.MultipleNegativesRankingLoss(model)

    model.fit(
        train_objectives=[(dataloader, loss)],
        epochs=epochs,
        warmup_steps=warmup_steps,
        output_path=str(output_dir),
        show_progress_bar=True,
        use_amp=False,
    )
