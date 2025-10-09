from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

import numpy as np

from app.schemas.document import TextChunk
from app.services.embedding_service import EmbeddingService


@dataclass
class ScoredChunk:
    chunk: TextChunk
    score: float


class InMemoryVectorStore:
    """Lightweight cosine-similarity vector store for prototypes."""

    def __init__(self, embedding_service: EmbeddingService) -> None:
        self.embedding_service = embedding_service
        self._chunks: List[TextChunk] = []
        self._matrix: np.ndarray | None = None

    def add_chunks(self, chunks: Sequence[TextChunk]) -> None:
        if not chunks:
            raise ValueError("No chunks supplied to vector store.")
        embeddings = self.embedding_service.embed_texts(
            chunk.content for chunk in chunks
        )
        self._matrix = self._normalize(np.array(embeddings, dtype=np.float32))
        self._chunks = list(chunks)

    def similarity_search(self, query: str, *, top_k: int) -> List[ScoredChunk]:
        if not self._chunks or self._matrix is None:
            raise RuntimeError("Vector store is empty. Populate it before querying.")
        query_embedding = self.embedding_service.embed_query(query)
        query_vector = self._normalize(np.array(query_embedding, dtype=np.float32))
        scores = self._matrix @ query_vector
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [
            ScoredChunk(chunk=self._chunks[idx], score=float(scores[idx]))
            for idx in top_indices
        ]

    @staticmethod
    def _normalize(matrix: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(matrix, axis=-1, keepdims=True)
        norms[norms == 0] = 1.0
        return matrix / norms
