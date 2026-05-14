"""Trace a single query through every stage of the RAG pipeline.

Calls the same modules `RagPipeline.answer()` uses, but one at a time, so
each step is visible and timeable. Designed as a live demo / presentation
aid: run it, talk over each block as it prints.

Usage
-----
    # Default: UiA index, grounded prompt, no reranker
    python scripts/trace_query.py "What is the master in cybersecurity about?"

    # With reranker (top-50 bi-encoder pool reranked to top-5)
    python scripts/trace_query.py "..." --rerank

    # On the BEIR/NQ index (no need to type a question -- picks one from the eval set)
    python scripts/trace_query.py --index beir_nq

    # Compare grounded vs vanilla prompt on the same query
    python scripts/trace_query.py "..." --prompt vanilla

    # Skip the LLM call (just trace through retrieval + reranking + prompt build)
    python scripts/trace_query.py "..." --no-generate
"""
from __future__ import annotations

import argparse
import sys
import textwrap
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from src.embed import Embedder
from src.generator import OllamaGenerator
from src.pipeline import load_config
from src.prompts import PROMPT_VARIANTS, format_context
from src.reranker import CrossEncoderReranker
from src.retriever import FaissRetriever


# --- pretty printing -----------------------------------------------------
BAR = "=" * 72
SUBBAR = "-" * 72


def header(step: str, title: str) -> None:
    print(f"\n{BAR}\n[{step}] {title}\n{BAR}")


def wrap(text: str, width: int = 72, indent: str = "    ") -> str:
    return textwrap.fill(text, width=width, initial_indent=indent,
                         subsequent_indent=indent)


def section_summary(t0: float) -> None:
    print(f"    (took {time.perf_counter() - t0:.3f}s)")


# --- main ----------------------------------------------------------------
def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("question", nargs="?", default=None,
                   help="The query. If omitted, a default is used per --index.")
    p.add_argument("--index", choices=["uia", "beir_nq"], default="uia",
                   help="Which prebuilt FAISS index to load (indexes/<name>/).")
    p.add_argument("--top-k", type=int, default=5,
                   help="Number of chunks returned (and put in the prompt).")
    p.add_argument("--rerank", action="store_true",
                   help="Enable cross-encoder reranking (top-50 -> top-k).")
    p.add_argument("--rerank-pool", type=int, default=50,
                   help="Size of the bi-encoder candidate pool fed to the reranker.")
    p.add_argument("--prompt", choices=list(PROMPT_VARIANTS.keys()),
                   default="grounded",
                   help="Prompt template to use (see src/prompts.py).")
    p.add_argument("--no-generate", action="store_true",
                   help="Skip the LLM call; print the prompt and stop.")
    p.add_argument("--config", default=str(ROOT / "config.yaml"))
    args = p.parse_args()

    cfg = load_config(args.config)

    # Default queries per index for one-key demos
    if args.question is None:
        args.question = {
            "uia":     "What is the master in cybersecurity about?",
            "beir_nq": "who sang waiting on the world to change",
        }[args.index]

    print(f"\nQuestion: {args.question!r}")
    print(f"Index:    indexes/{args.index}/    "
          f"top_k={args.top_k}    rerank={args.rerank}    prompt={args.prompt}")

    # ------------------------------------------------------------------
    # Stage 1 - embed the query
    # ------------------------------------------------------------------
    header("1/5", "embed.py  -- turn the query into a 384-dim vector")
    t0 = time.perf_counter()
    embedder = Embedder(
        model_name=cfg["embedding"]["model_name"],
        batch_size=cfg["embedding"]["batch_size"],
        normalize=cfg["embedding"]["normalize"],
    )
    q_vec = embedder.embed_query(args.question)
    print(f"    model       = {cfg['embedding']['model_name']}")
    print(f"    dim         = {q_vec.shape[0]}")
    print(f"    L2 norm     = {np.linalg.norm(q_vec):.4f}   "
          f"(normalize_embeddings={cfg['embedding']['normalize']})")
    print(f"    first 8 dims= {np.array2string(q_vec[:8], precision=4, separator=', ')}")
    section_summary(t0)

    # ------------------------------------------------------------------
    # Stage 2 - retrieve top-N from FAISS
    # ------------------------------------------------------------------
    pool_k = args.rerank_pool if args.rerank else args.top_k
    header("2/5", f"retriever.py  -- FAISS IndexFlatIP, top-{pool_k}")
    t0 = time.perf_counter()
    retriever = FaissRetriever(embedder)
    retriever.load(ROOT / "indexes" / args.index)
    print(f"    n_vectors   = {retriever.index.ntotal:,}")
    print(f"    n_chunks    = {len(retriever.chunks):,}")
    candidates = retriever.search(args.question, top_k=pool_k)
    print(f"    returned    = {len(candidates)} chunks")
    print(SUBBAR)
    for r in candidates[: min(5, len(candidates))]:
        meta = r.chunk.metadata or {}
        title = meta.get("title") or r.chunk.source_id
        print(f"    [{r.rank+1}] score={r.score:.4f}  {title[:55]}")
        print(wrap((r.chunk.text or "").strip().replace("\n", " ")[:160] + "...",
                   width=68, indent="         "))
    if len(candidates) > 5:
        print(f"    ... ({len(candidates) - 5} more candidates not shown)")
    section_summary(t0)

    # ------------------------------------------------------------------
    # Stage 3 (optional) - cross-encoder reranking
    # ------------------------------------------------------------------
    if args.rerank:
        header("3/5", "reranker.py  -- cross-encoder rescore + resort")
        t0 = time.perf_counter()
        reranker = CrossEncoderReranker(cfg["retrieval"]["reranker_model"])
        results = reranker.rerank(args.question, candidates, top_k=args.top_k)
        print(f"    model       = {cfg['retrieval']['reranker_model']}")
        print(f"    in          = {len(candidates)} candidates "
              f"(bi-encoder scores)")
        print(f"    out         = {len(results)} candidates "
              f"(cross-encoder scores)")
        print(SUBBAR)
        # Show how the order changed
        prev_order = {id(c.chunk): c.rank for c in candidates}
        for r in results:
            old = prev_order.get(id(r.chunk), -1)
            arrow = f"(bi-rank {old+1} -> ce-rank {r.rank+1})" if old != r.rank \
                    else f"(unchanged at rank {r.rank+1})"
            meta = r.chunk.metadata or {}
            title = meta.get("title") or r.chunk.source_id
            print(f"    [{r.rank+1}] ce_score={r.score:.4f}  "
                  f"{title[:40]}  {arrow}")
        section_summary(t0)
    else:
        header("3/5", "reranker.py  -- SKIPPED (use --rerank to enable)")
        results = candidates[: args.top_k]

    # ------------------------------------------------------------------
    # Stage 4 - format the prompt
    # ------------------------------------------------------------------
    header("4/5", f"prompts.py  -- build the '{args.prompt}' prompt")
    t0 = time.perf_counter()
    system_msg, builder = PROMPT_VARIANTS[args.prompt]
    user_prompt = builder(args.question, results)
    print(f"    variant     = {args.prompt}")
    print(f"    system msg  ({len(system_msg)} chars):")
    print(wrap(system_msg, width=68))
    print(f"\n    user prompt ({len(user_prompt)} chars, "
          f"~{len(user_prompt)//4} tokens):")
    print(SUBBAR)
    # Print full prompt with leading indent so it's clearly delimited
    for line in user_prompt.split("\n"):
        print(f"    | {line}")
    print(SUBBAR)
    section_summary(t0)

    if args.no_generate:
        print("\n[skipped 5/5 -- --no-generate]\n")
        return

    # ------------------------------------------------------------------
    # Stage 5 - generate the answer
    # ------------------------------------------------------------------
    header("5/5", f"generator.py  -- {cfg['generator']['model']} "
                  f"@ temperature={cfg['generator']['temperature']}")
    t0 = time.perf_counter()
    gen = OllamaGenerator(
        model=cfg["generator"]["model"],
        temperature=cfg["generator"]["temperature"],
        num_ctx=cfg["generator"]["num_ctx"],
    )
    answer = gen.generate(user_prompt, system=system_msg)
    print(SUBBAR)
    for line in answer.split("\n"):
        print(f"    > {line}")
    print(SUBBAR)
    section_summary(t0)
    print()


if __name__ == "__main__":
    main()
