"""
embeddings.py
-------------
turn text (chunks and questions) into embedding vectors.
"""

from typing import List

import numpy as np
from sentence_transformers import SentenceTransformer

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


class EmbeddingError(Exception):
    """Raised when the embedding model cannot load or encode text."""


class EmbeddingModel:
    """Small wrapper around a SentenceTransformer model."""

    def __init__(self, model_name: str = MODEL_NAME):
        try:
            self.model = SentenceTransformer(model_name)
        except Exception as exc:
            raise EmbeddingError(
                "Could not load the embedding model. On the first run you need an "
                f"internet connection to download it. Details: {exc}"
            ) from exc

    def embed_texts(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """Return a float32 array of shape (number_of_texts, 384), L2-normalized."""
        if not texts:
            raise EmbeddingError("There is no text to embed.")
        try:
            vectors = self.model.encode(
                texts,
                batch_size=batch_size,
                convert_to_numpy=True,
                normalize_embeddings=True,   # length 1 -> inner product == cosine similarity
                show_progress_bar=False,
            )
        except Exception as exc:
            raise EmbeddingError(f"Embedding generation failed: {exc}") from exc
        return np.asarray(vectors, dtype="float32")

    def embed_query(self, question: str) -> np.ndarray:
        """Embed one question. Returns shape (1, 384)."""
        return self.embed_texts([question])