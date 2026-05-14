"""Browser-based chatbot GUI for the RAG system.

Wraps `RagPipeline` in a Gradio `ChatInterface` so the system can be
demoed without a CLI. Tokens stream in as the LLM produces them and
each turn shows the retrieved sources below the answer.

Run:
    python scripts/11_chatbot_gui.py                    # default: UiA index
    python scripts/11_chatbot_gui.py --index uia
    python scripts/11_chatbot_gui.py --index beir_nq    # demo on the BEIR subset
    python scripts/11_chatbot_gui.py --share            # public Gradio URL
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterator, List

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import gradio as gr

from src.pipeline import RagPipeline, load_config
from src.prompts import PROMPT_VARIANTS
from src.retriever import Retrieved


def format_sources(retrieved: List[Retrieved]) -> str:
    """Render retrieved chunks as a Markdown bullet list with links."""
    if not retrieved:
        return ""
    lines = ["", "---", "**Sources**", ""]
    for r in retrieved:
        title = r.chunk.metadata.get("title") or r.chunk.source_id
        url = r.chunk.metadata.get("url")
        score = f"score {r.score:.3f}"
        if url:
            lines.append(f"- [{title}]({url}) — {score}")
        else:
            lines.append(f"- {title} — {score}")
    return "\n".join(lines)


def build_chat_fn(pipeline: RagPipeline):
    """Closure over the pipeline so Gradio doesn't need a globals lookup."""
    system_prompt, prompt_builder = PROMPT_VARIANTS["grounded"]
    generator = pipeline._ensure_generator()

    def chat(message: str, history) -> Iterator[str]:
        # Retrieve once per user turn -- we deliberately don't carry context
        # from prior turns, because this is a *retrieval*-augmented bot, not
        # a conversational memory. Each question is grounded fresh.
        retrieved = pipeline.retrieve(message)
        prompt = prompt_builder(message, retrieved)
        sources_md = format_sources(retrieved)

        # Stream the answer; append the sources block once generation ends.
        partial_answer = ""
        for partial in generator.stream_generate(prompt, system=system_prompt):
            partial_answer = partial
            yield partial_answer + sources_md
        # Final yield ensures the full answer + sources are present.
        yield partial_answer + sources_md

    return chat


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--index", default="uia",
                   help="Subdirectory under indexes/ to load (default: uia)")
    p.add_argument("--port", type=int, default=7860, help="Local port (default: 7860)")
    p.add_argument("--share", action="store_true",
                   help="Get a temporary public URL (Gradio tunnel, ~72h).")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    config = load_config(repo_root / "config.yaml")
    index_dir = repo_root / config["paths"]["index_dir"] / args.index
    if not index_dir.exists():
        print(f"No index at {index_dir}.")
        print("Build one first: scripts/04_build_uia_index.py "
              "or scripts/01_build_beir_index.py")
        sys.exit(1)

    print(f"Loading pipeline + index from {index_dir}...")
    pipeline = RagPipeline(config)
    pipeline.load_index(index_dir)
    print("Pipeline ready.")

    title = f"RAG Chatbot — {args.index}"
    description = (
        f"Retrieval-augmented chatbot. Index: `{args.index}`  "
        f"Embedder: `{config['embedding']['model_name']}`  "
        f"LLM: `{config['generator']['model']}`  "
        f"top-k: {config['retrieval']['top_k']}.\n\n"
        "Each answer is grounded in the retrieved passages shown below it."
    )

    examples = (
        ["What programmes does UiA offer in ICT?",
         "Where is the ICT department located?",
         "What language are the master courses taught in?"]
        if args.index == "uia"
        else ["Who wrote the song 'Imagine'?",
              "What is the capital of Australia?",
              "When did the Berlin Wall fall?"]
    )

    chat_fn = build_chat_fn(pipeline)
    # We pass only kwargs that exist across Gradio 3.x/4.x/5.x. The chat
    # function yields strings, which both the legacy "tuples" history
    # format and the modern "messages" format consume identically -- so
    # we don't need to specify `type=`. (If you upgrade to Gradio >=4.36
    # you can add `type="messages"` for the OpenAI-style history dicts.)
    demo = gr.ChatInterface(
        fn=chat_fn,
        title=title,
        description=description,
        examples=examples,
    )
    demo.queue()                    # required for streaming generators
    demo.launch(server_port=args.port, share=args.share, inbrowser=True)


if __name__ == "__main__":
    main()
