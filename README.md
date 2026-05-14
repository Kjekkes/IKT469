# IKT469 — Option 8: Retrieval-Augmented Generation

Semester assignment for IKT469. We build a small RAG system, evaluate it on a public benchmark (BEIR/NQ), then adapt the same pipeline to a custom corpus scraped from UiA's IKT course pages so the final chatbot can answer questions about IKT classes.

## Goals (from the assignment brief)

1. Understand the full RAG pipeline: collection, chunking, embedding, retrieval, prompting, response grounding.
2. Quantitatively evaluate retrieval and answer quality on BEIR/NQ.
3. Reuse the same pipeline on a small custom corpus (UiA IKT pages) to produce a working chatbot.
4. Compare design choices (chunk size, top-k, reranking, prompt template) and report findings.

## Stack

Chosen with a CPU-only laptop in mind. Everything runs locally, no API keys required.

| Layer            | Choice                                             | Why                                            |
| ---------------- | -------------------------------------------------- | ---------------------------------------------- |
| Orchestration    | LangChain                                          | Standard, lots of literature to cite           |
| Embeddings       | `sentence-transformers/all-MiniLM-L6-v2` (384-dim) | Fast on CPU, strong baseline on BEIR           |
| Vector store     | FAISS (in-memory, flat IP)                         | Zero-config, fine for sub-million chunks       |
| Reranker (opt.)  | `cross-encoder/ms-marco-MiniLM-L-6-v2`             | Optional ablation, also CPU-friendly           |
| Generator        | Ollama with `llama3.2:3b-instruct-q4_K_M`          | Quantized 3B model runs ~5-15 tok/s on CPU     |
| Scraper          | `requests` + `beautifulsoup4` + `trafilatura`      | Trafilatura strips boilerplate cleanly         |
| Benchmark loader | `beir` python package                              | Official BEIR loaders + qrels                  |
| Evaluation       | BEIR built-ins (Recall, nDCG, MRR) + custom RAGAS-lite | Retrieval + answer grounding              |

## Repository layout

```
IKT469/
├── README.md                  <- this file
├── requirements.txt
├── config.yaml                <- single source of truth for paths, model names, k, etc.
├── data/
│   ├── beir/                  <- downloaded BEIR/NQ subset (gitignored)
│   ├── uia_raw/               <- raw scraped HTML (gitignored)
│   └── uia_clean/             <- cleaned chunks, jsonl
├── indexes/                   <- FAISS indexes (gitignored)
├── src/
│   ├── __init__.py
│   ├── pipeline.py            <- chunk + embed + retrieve + generate, shared by both tracks
│   ├── chunking.py
│   ├── embed.py
│   ├── retriever.py
│   ├── generator.py           <- Ollama wrapper
│   ├── prompts.py
│   ├── beir_eval.py           <- BEIR/NQ benchmark driver
│   ├── uia_scrape.py          <- crawler for UiA IKT pages
│   └── chatbot.py             <- CLI chat over the UiA index
├── scripts/
│   ├── 01_build_beir_index.py
│   ├── 02_eval_beir.py
│   ├── 03_scrape_uia.py
│   ├── 04_build_uia_index.py
│   └── 05_run_chatbot.py
├── notebooks/                 <- exploratory analysis, plots for the report
└── report/
    └── report.tex             <- LaTeX source for the final PDF
```

## CPU-friendly scoping

NQ in BEIR has ~2.7M passages. We will **not** embed the full corpus on CPU. Plan:

- Cap the corpus to ~20k-50k passages (sampled so all qrel-relevant docs are kept).
- Cap evaluated queries to ~200-500.
- Use `all-MiniLM-L6-v2` (22M params, ~3 ms/sentence on CPU) — full subset embeds in <5 min.
- Generation step uses a 3B quantized model; we run it on at most 100 queries for the answer-quality eval, not on every query.

For the UiA track the corpus is tiny (a few hundred pages at most), so no scoping needed.

## Evaluation plan

**Retrieval (BEIR/NQ subset):** Recall@{1,5,10}, nDCG@10, MRR@10. Standard BEIR `EvaluateRetrieval`.

**Answer quality (BEIR/NQ subset, ~100 queries):**
- Exact / fuzzy match against the short answer.
- Faithfulness check (LLM-as-judge with Ollama on a small sample).

**Ablations to report:**
1. Chunk size: 128 / 256 / 512 tokens.
2. Top-k: 1 / 3 / 5 / 10.
3. With vs without cross-encoder reranking.
4. Prompt template: vanilla vs "answer only from context" vs CoT.
5. Embedding model: MiniLM vs MiniLM contrastively fine-tuned on a held-out BEIR/NQ subset (this is also the project's training step).

**UiA track:** small hand-built eval set (~20 questions about IKT courses with reference answers), reported as accuracy and a few qualitative examples.

## Milestones

| Week | Deliverable                                               |
| ---- | --------------------------------------------------------- |
| 1    | Skeleton, requirements, smoke test on toy corpus          |
| 2    | BEIR/NQ subset loaded, retrieval-only eval working        |
| 3    | Generator wired up via Ollama, end-to-end answers on NQ   |
| 4    | Ablations (chunk size, top-k, reranker)                   |
| 5    | UiA scraper + cleaned corpus + chatbot                    |
| 6    | Report draft, plots, final eval                           |
| 7    | Polish, submit PDF + GitHub link                          |

## What we copy vs what we contribute

Per the assignment, this needs to be explicit. Tracked in `report/attribution.md`:

- **Copied / reused:** LangChain primitives, BEIR loader, sentence-transformers checkpoints, Ollama runtime, the standard "answer from context" prompt pattern.
- **Our contribution:** the UiA scraper, the chunking/cleaning for UiA pages, the ablation harness, the eval set for UiA, the analysis and report.

## References (to expand in the report)

- Lewis et al., 2020. *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks.* NeurIPS.
- Karpukhin et al., 2020. *Dense Passage Retrieval for Open-Domain Question Answering.* EMNLP.
- Thakur et al., 2021. *BEIR: A Heterogeneous Benchmark for Zero-shot Evaluation of Information Retrieval Models.* NeurIPS.
- Gao et al., 2023. *Retrieval-Augmented Generation for Large Language Models: A Survey.* arXiv:2312.10997.
- Es et al., 2023. *RAGAS: Automated Evaluation of Retrieval Augmented Generation.* arXiv:2309.15217.
