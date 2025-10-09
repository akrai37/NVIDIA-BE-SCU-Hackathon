from __future__ import annotations

import io
from typing import List, Optional

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from pypdf import PdfReader
import pypdfium2 as pdfium
from PIL import Image

from app.core.config import get_settings
from app.schemas.document import DocumentBundle, TextChunk
from app.services.ocr_service import (
    NvidiaOcrService,
    OcrServiceProtocol,
    OcrUnavailableError,
)


class DocumentProcessor:
    """Handle parsing and chunking of uploaded documents."""

    def __init__(self, ocr_service: Optional[OcrServiceProtocol] = None) -> None:
        self.settings = get_settings()
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.settings.chunk_size,
            chunk_overlap=self.settings.chunk_overlap,
            add_start_index=True,
        )
        if ocr_service is not None:
            self.ocr_service = ocr_service
        elif self.settings.enable_ocr:
            try:
                self.ocr_service = NvidiaOcrService()
            except OcrUnavailableError:
                self.ocr_service = None
        else:
            self.ocr_service = None

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
        pdfium_doc: Optional[pdfium.PdfDocument] = None
        if self.ocr_service:
            try:
                pdfium_doc = pdfium.PdfDocument(io.BytesIO(file_bytes))
            except Exception:  # pragma: no cover - pdfium loading failures
                pdfium_doc = None

        pages_text: List[str] = []
        for page_index, page in enumerate(reader.pages):
            try:
                text = page.extract_text() or ""
            except Exception:  # pragma: no cover - edge cases in PDF parsing
                text = ""
            if not text.strip() and self.ocr_service and pdfium_doc is not None:
                text = self._ocr_page(pdfium_doc, page_index)
            pages_text.append(text.strip())
        if pdfium_doc is not None:
            pdfium_doc.close()
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

    def _ocr_page(self, pdfium_doc: pdfium.PdfDocument, page_index: int) -> str:
        if not self.ocr_service:
            return ""
        try:
            page = pdfium_doc[page_index]
            scale = float(self.settings.ocr_render_scale or 2.0)
            bitmap = page.render(scale=scale)  # type: ignore[arg-type]
            try:
                pil_image: Image.Image = bitmap.to_pil()
            finally:
                bitmap.close()
            return self.ocr_service.extract_text(pil_image).strip()
        except Exception:  # pragma: no cover - OCR failures should not crash pipeline
            return ""
