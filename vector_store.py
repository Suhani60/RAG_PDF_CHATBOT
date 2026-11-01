"""
vector_store.py
---------------
store embeddings in FAISS and search for similar chunks.
"""

from dataclasses import dataclass
from typing import List, Optional

import faiss
import numpy as np

from text_splitter import Chunk


class VectorStoreError(Exception):
    """Raised for invalid vector-store operations."""


@dataclass
class RetrievedChunk:
    """A search hit: the chunk and how similar it was to the question (0-1)."""
    chunk: Chunk
    score: float


class VectorStore:
    def __init__(self):
        self.index: Optional[faiss.Index] = None
        self.chunks: List[Chunk] = []

    @property
    def size(self) -> int:
        """Number of vectors stored."""
        return 0 if self.index is None else int(self.index.ntotal)

    def build(self, embeddings: np.ndarray, chunks: List[Chunk]) -> None:
        """Create a fresh index from chunk embeddings."""
        if embeddings.ndim != 2 or len(chunks) == 0:
            raise VectorStoreError("Embeddings must be a non-empty 2-D array.")
        if len(embeddings) != len(chunks):
            raise VectorStoreError("Number of embeddings and chunks must match.")

        vectors = np.ascontiguousarray(embeddings, dtype="float32")
        index = faiss.IndexFlatIP(vectors.shape[1])   # IP = inner product
        index.add(vectors)

        self.index = index
        self.chunks = list(chunks)   # row i of the index == chunks[i]

    def search(self, query_embedding: np.ndarray, top_k: int = 4) -> List[RetrievedChunk]:
        """Return the top_k most similar chunks, best first."""
        if self.index is None or self.index.ntotal == 0:
            raise VectorStoreError("The index is empty. Process a PDF first.")

        query = np.ascontiguousarray(query_embedding, dtype="float32").reshape(1, -1)
        if query.shape[1] != self.index.d:
            raise VectorStoreError(
                f"Question vector has {query.shape[1]} dimensions, index expects {self.index.d}."
            )

        k = max(1, min(top_k, self.index.ntotal))
        scores, ids = self.index.search(query, k)

        results: List[RetrievedChunk] = []
        for score, idx in zip(scores[0], ids[0]):
            if idx == -1:      # FAISS uses -1 for "no result"
                continue
            results.append(RetrievedChunk(chunk=self.chunks[idx], score=float(score)))
        return results