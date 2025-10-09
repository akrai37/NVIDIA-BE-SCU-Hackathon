from __future__ import annotations

from typing import List, Optional
from uuid import uuid4

from pydantic import BaseModel


class TextChunk(BaseModel):
    """Single chunk of extracted document text."""

    chunk_id: str
    content: str
    page_number: Optional[int] = None


class DocumentBundle(BaseModel):
    """Normalized representation of an uploaded document."""

    document_id: str
    title: str
    page_count: int
    total_characters: int
    chunks: List[TextChunk]

    @classmethod
    def from_chunks(
        cls,
        *,
        title: str,
        page_count: int,
        chunks: List[TextChunk],
    ) -> "DocumentBundle":
        return cls(
            document_id=str(uuid4()),
            title=title,
            page_count=page_count,
            total_characters=sum(len(chunk.content) for chunk in chunks),
            chunks=chunks,
        )
