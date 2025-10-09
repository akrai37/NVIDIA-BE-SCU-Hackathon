from __future__ import annotations

import io
from typing import List

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from pypdf import PdfReader

from app.core.config import get_settings
from app.schemas.document import DocumentBundle, TextChunk


class DocumentProcessor:
    """Handle parsing and chunking of uploaded documents."""

    def __init__(self) -> None:
        self.settings = get_settings()
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.settings.chunk_size,
            chunk_overlap=self.settings.chunk_overlap,
            add_start_index=True,
        )

    def process(self, *, file_bytes: bytes, filename: str) -> DocumentBundle:
        pages_text = self._extract_pdf_text(file_bytes)
        chunks = self._build_chunks(pages_text)
        title = filename.rsplit(".", 1)[0]
        return DocumentBundle.from_chunks(
            title=title,
            page_count=len(pages_text),
            chunks=chunks,
        )

    def _extract_pdf_text(self, file_bytes: bytes) -> List[str]:
        reader = PdfReader(io.BytesIO(file_bytes))
        pages_text: List[str] = []
        for page_index, page in enumerate(reader.pages):
            try:
                text = page.extract_text() or ""
            except Exception:  # pragma: no cover - edge cases in PDF parsing
                text = ""
            pages_text.append(text.strip())
        if not pages_text:
            raise ValueError("The uploaded PDF is empty or unreadable.")
        return pages_text

    def _build_chunks(self, pages_text: List[str]) -> List[TextChunk]:
        documents: List[Document] = []
        for page_number, page_text in enumerate(pages_text, start=1):
            if not page_text:
                continue
            documents.append(
                Document(
                    page_content=page_text,
                    metadata={
                        "page_number": page_number,
                    },
                )
            )
        if not documents:
            raise ValueError(
                "No readable text detected in the PDF. Consider enabling OCR upstream."
            )
        split_docs = self.splitter.split_documents(documents)

        chunks: List[TextChunk] = []
        for idx, split_doc in enumerate(split_docs):
            chunks.append(
                TextChunk(
                    chunk_id=f"chunk-{idx+1}",
                    content=split_doc.page_content,
                    page_number=split_doc.metadata.get("page_number"),
                )
            )
        return chunks
