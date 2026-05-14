"""Glue layer: chunk + embed + retrieve + generate.

Both the BEIR/NQ benchmark driver and the UiA chatbot use this. Keeping it
narrow on purpose: each track owns its own data loading and evaluation, and
calls into RagPipeline.answer() for the actual RAG step.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import yaml

from .chunking import Chunk, chunk_documents
from .embed import Embedder
from .generator import OllamaGenerator
from .prompts import PROMPT_VARIANTS
from .retriever import FaissRetriever, Retrieved


@dataclass
class RagAnswer:
    question: str
    answer: str
    retrieved: List[Retrieved]


class RagPipeline:
    def __init__(self, config: dict):
        self.config = config
        self.embedder = Embedder(
            model_name=config["embedding"]["model_name"],
            batch_size=config["embedding"]["batch_size"],
            normalize=config["embedding"]["normalize"],
        )
        self.retriever = FaissRetriever(self.embedder)
        self.generator: Optional[OllamaGenerator] = None  # lazy

    # --- Index management -------------------------------------------------

    def build_index_from_documents(self, docs: List[dict]) -> List[Chunk]:
        chunks = chunk_documents(
            docs,
            chunk_size_tokens=self.config["chunking"]["size"],
            overlap_tokens=self.config["chunking"]["overlap"],
        )
        self.retriever.build(chunks)
        return chunks

    def save_index(self, dir_path: str | Path) -> None:
        self.retriever.save(dir_path)

    def load_index(self, dir_path: str | Path) -> None:
        self.retriever.load(dir_path)

    # --- Querying ---------------------------------------------------------

    def retrieve(self, question: str, top_k: Optional[int] = None) -> List[Retrieved]:
        k = top_k or self.config["retrieval"]["top_k"]
        return self.retriever.search(question, top_k=k)

    def _ensure_generator(self) -> OllamaGenerator:
        if self.generator is None:
            self.generator = OllamaGenerator(
                model=self.config["generator"]["model"],
                temperature=self.config["generator"]["temperature"],
                num_ctx=self.config["generator"]["num_ctx"],
            )
        return self.generator

    def answer(self, question: str, prompt_variant: str = "grounded") -> RagAnswer:
        return self.answer_with(question=question, prompt_variant=prompt_variant)

    def answer_with(
        self,
        question: str,
        top_k: Optional[int] = None,
        prompt_variant: str = "grounded",
    ) -> RagAnswer:
        """Like `answer`, but lets ablations override top_k per call."""
        results = self.retrieve(question, top_k=top_k)
        system, builder = PROMPT_VARIANTS[prompt_variant]
        prompt = builder(question, results)
        gen = self._ensure_generator()
        text = gen.generate(prompt, system=system)
        return RagAnswer(question=question, answer=text, retrieved=results)


# --- Convenience ---------------------------------------------------------


def load_config(path: str | Path = "config.yaml") -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)
