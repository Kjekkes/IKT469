"""Embedding wrapper.

Thin layer over sentence-transformers so the rest of the code only sees
`embed_texts(list[str]) -> np.ndarray`. We L2-normalize so cosine
similarity reduces to inner product, which lets us use FAISS IndexFlatIP.
"""
from __future__ import annotations

from typing import List

import numpy as np
from sentence_transformers import SentenceTransformer


class Embedder:
    def __init__(self, model_name: str, batch_size: int = 64, normalize: bool = True):
        self.model_name = model_name
        self.batch_size = batch_size
        self.normalize = normalize
        self.model = SentenceTransformer(model_name)
        self.dim = self.model.get_sentence_embedding_dimension()

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        vecs = self.model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=len(texts) > 256,
            convert_to_numpy=True,
            normalize_embeddings=self.normalize,
        )
        return vecs.astype("float32")

    def embed_query(self, query: str) -> np.ndarray:
        return self.embed_texts([query])[0]
