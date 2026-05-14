# Attribution: copied vs contributed

The assignment requires us to be explicit about what is reused and what is
our own contribution. This file is the source of truth; the report's
Methodology section points here.

## Design inspirations

We did not fork any existing repository. Every file under `src/` and
`scripts/` was written from scratch for this project. However, the
architecture and many of the specific implementation patterns follow
established conventions documented elsewhere; we acknowledge those here
even though no code was copied directly.

| Component / pattern | Inspired by |
| --- | --- |
| Pipeline structure: chunk → embed → store → retrieve → format prompt → generate | LangChain RAG tutorials (`RetrievalQA` chain), HuggingFace RAG cookbook |
| Recursive character splitter usage in `src/chunking.py` | LangChain `RecursiveCharacterTextSplitter` documentation |
| Grounded prompt template ("Answer using only the context below; otherwise say 'I don't know'") in `src/prompts.py` | HuggingFace RAG cookbook prompt; standard pattern across RAG tutorials |
| BEIR loader + evaluator usage in `src/beir_eval.py` | BEIR project's `examples/retrieval/evaluation/dense/` scripts |
| Contrastive fine-tuning recipe in `src/finetune.py` (MultipleNegativesRankingLoss + `model.fit()`) | sentence-transformers training documentation; matches DPR training setup of \citet{karpukhin2020dpr} |
| Cross-encoder reranking (top-N from bi-encoder, re-score, re-sort) in `src/reranker.py` | sentence-transformers `CrossEncoder` documentation; standard since \citet{nogueira2019rerank} |
| Polite BFS web crawl + trafilatura body extraction in `src/uia_scrape.py` | Standard scraping practice; trafilatura usage from its documentation \citep{barbaresi2021trafilatura} |
| LLM-as-judge faithfulness scoring in `src/answer_eval.py` | RAGAS \citep{es2023ragas} (we use a single-integer simplification of its faithfulness sub-metric) |
| Subsample-and-evaluate pattern for BEIR on CPU | Common practice in BEIR-related tutorials when full corpus is impractical |

In short: the *what to build* is the standard RAG architecture from
\citet{lewis2020rag} and \citet{gao2023rag}; the *how to build each
component* follows the official documentation of LangChain,
sentence-transformers, BEIR, and trafilatura. The contributions are the
glue, the ablation framework, the UiA scraper, the eval harnesses, and
the analysis.

## Reused / external code and assets

| Component | Source | License | How we use it |
| --- | --- | --- | --- |
| `langchain`, `langchain-text-splitters` | LangChain project | MIT | Recursive character splitter, document abstractions |
| `sentence-transformers` | UKP Lab | Apache 2.0 | Bi-encoder and cross-encoder runtime |
| `all-MiniLM-L6-v2` checkpoint | Hugging Face hub | Apache 2.0 | Embedding model |
| `ms-marco-MiniLM-L-6-v2` checkpoint | Hugging Face hub | Apache 2.0 | Cross-encoder reranker |
| `faiss-cpu` | Meta AI | MIT | Vector index (`IndexFlatIP`) |
| `beir` | UKP Lab / Thakur et al. | Apache 2.0 | NQ loader and `EvaluateRetrieval` |
| `ollama` | Ollama project | MIT | Local LLM runtime |
| `llama3.2:3b-instruct-q4_K_M` | Meta / quantised by Ollama | Llama 3 license | Generator |
| `trafilatura` | Adrien Barbaresi | Apache 2.0 | HTML body extraction for the UiA scraper |
| `beautifulsoup4`, `lxml`, `requests` | various | MIT-ish | Crawling |
| `gradio` | Gradio team | Apache 2.0 | Chat GUI in `scripts/11_chatbot_gui.py` |
| Standard "answer only from the provided context" prompt pattern | Common practice | n/a | `src/prompts.py::SYSTEM_GROUNDED` |

## Our contribution

| File / module | What it is |
| --- | --- |
| `src/pipeline.py` | The `RagPipeline` glue layer (chunk + embed + retrieve + generate) |
| `src/embed.py`, `src/retriever.py`, `src/generator.py`, `src/chunking.py`, `src/prompts.py` | Thin wrappers + ergonomics around the libraries above |
| `src/reranker.py` | `CrossEncoderReranker` wrapper |
| `src/beir_eval.py` | BEIR loader + deterministic subsampler that preserves qrels, plus our run/eval driver |
| `src/ablations.py` | Ablation framework: configs, index variants, max-pool aggregation |
| `src/answer_eval.py` | LLM-as-judge faithfulness scorer (single-integer prompt) |
| `src/uia_scrape.py` | Polite robots-aware BFS crawler for UiA pages |
| `src/uia_eval.py` | UiA chatbot evaluation harness (source recall, keyword match, judge) |
| `src/finetune.py` | Contrastive fine-tuning utilities for the bi-encoder (training-pair builder + trainer wrapper) |
| `eval/uia_eval.yaml` | Hand-written question set with abstention tests |
| `scripts/00`–`11` (all numbered run scripts) | End-to-end drivers over the `src/` modules |
| `scripts/08_make_plots.py` | Plots |
| `scripts/10_finetune_embedder.py` | Trains MiniLM on a held-out BEIR/NQ subset; this is the project's training step |
| `scripts/11_chatbot_gui.py` | Gradio chat GUI wrapping `RagPipeline` with streaming + sources panel |
| `scripts/inspect_uia.py` | Utility CLI to browse/search the scraped UiA corpus (debugging aid) |
| `scripts/trace_query.py` | Single-query pipeline tracer: prints each stage (embed → retrieve → rerank → prompt → generate); presentation/demo aid |
| `report/report.tex` and the analysis | Our own work |

