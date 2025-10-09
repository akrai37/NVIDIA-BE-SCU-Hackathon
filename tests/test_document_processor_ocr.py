from pathlib import Path

from fpdf import FPDF

from app.services.document_processor import DocumentProcessor


class _DummyOcrService:
    def __init__(self, text: str = "Scanned page text") -> None:
        self._text = text

    def extract_text(self, image) -> str:  # pragma: no cover - simple stub
        return self._text


def _make_blank_pdf(tmp_path: Path) -> bytes:
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(40, 10, " ")
    pdf_path = tmp_path / "blank.pdf"
    pdf.output(str(pdf_path))
    return pdf_path.read_bytes()


def test_document_processor_uses_ocr_for_blank_pages(tmp_path):
    pdf_bytes = _make_blank_pdf(tmp_path)
    processor = DocumentProcessor(ocr_service=_DummyOcrService())
    bundle = processor.process(file_bytes=pdf_bytes, filename="blank.pdf")
    combined_text = "".join(chunk.content for chunk in bundle.chunks)
    assert "Scanned page text" in combined_text