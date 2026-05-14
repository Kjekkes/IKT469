"""Text chunking utilities.

Char-based recursive splitting via LangChain. Token counts are approximate
(we use the rule of thumb ~4 chars/token). For a project at this scale that
is accurate enough; if we want exact token counts later we can swap in a
tokenizer-aware splitter.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List

from langchain_text_splitters import RecursiveCharacterTextSplitter


@dataclass
class Chunk:
    """A retrievable unit of text plus its provenance."""
    text: str
    source_id: str
    chunk_id: int
    metadata: dict


def make_splitter(chunk_size_tokens: int, overlap_tokens: int) -> RecursiveCharacterTextSplitter:
    """Build a splitter sized in approximate tokens (4 chars per token)."""
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size_tokens * 4,
        chunk_overlap=overlap_tokens * 4,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )


def chunk_documents(
    docs: Iterable[dict],
    chunk_size_tokens: int = 256,
    overlap_tokens: int = 32,
) -> List[Chunk]:
    """Chunk a list of documents.

    Each input doc must be a dict with at least 'id' and 'text'. Any other
    keys are forwarded into chunk metadata so we can trace a chunk back to
    its source URL/title at retrieval time.
    """
    splitter = make_splitter(chunk_size_tokens, overlap_tokens)
    chunks: List[Chunk] = []
    for doc in docs:
        pieces = splitter.split_text(doc["text"])
        for i, piece in enumerate(pieces):
            chunks.append(
                Chunk(
                    text=piece,
                    source_id=doc["id"],
                    chunk_id=i,
                    metadata={k: v for k, v in doc.items() if k not in {"id", "text"}},
                )
            )
    return chunks
