from __future__ import annotations

from typing import Iterable, List

from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings

from app.core.config import get_settings


class EmbeddingService:
    """Wrapper around NVIDIA embedding models with sensible defaults."""

    def __init__(self) -> None:
        settings = get_settings()
        if not settings.nvidia_api_key:
            raise RuntimeError(
                "NVIDIA_API_KEY is not set. Provide a valid key to generate embeddings."
            )
        self._client = NVIDIAEmbeddings(
            model=settings.nvidia_embeddings_model,
            api_key=settings.nvidia_api_key,
            base_url=settings.nvidia_base_url,
        )

    def embed_texts(self, texts: Iterable[str]) -> List[List[float]]:
        return self._client.embed_documents(list(texts))

    def embed_query(self, text: str) -> List[float]:
        return self._client.embed_query(text)
