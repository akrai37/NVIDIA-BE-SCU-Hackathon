from datetime import datetime

from app.schemas.document import DocumentBundle, TextChunk
from app.services.analysis_service import DocumentAnalyzer
from app.schemas.api import InsightPriority


def make_bundle() -> DocumentBundle:
    chunks = [
        TextChunk(chunk_id="chunk-1", content="Deadline: 2025-12-31\nBudget: $10,000", page_number=1),
        TextChunk(chunk_id="chunk-2", content="Eligibility: Nonprofits with budgets < $250k", page_number=1),
    ]
    return DocumentBundle.from_chunks(title="testdoc", page_count=1, chunks=chunks)


def test_to_response_without_pdf():
    analyzer = DocumentAnalyzer()
    bundle = make_bundle()

    # Minimal fake payload resembling LLM output
    fake_payload = {
        "summary": "Test summary",
        "key_highlights": ["Highlight A", "Highlight B"],
        "categorized_insights": {
            "critical": [
                {"label": "Deadline", "description": "Deadline exists", "source_chunk_id": "chunk-1"},
                {"label": "Deadline", "description": "Duplicate entry", "source_chunk_id": "chunk-1"},
            ],
            "important": [
                {
                    "label": "Eligibility",
                    "description": "Eligibility noted",
                    "source_chunk_id": "chunk-2",
                }
            ],
            "informational": [],
        },
        "extracted_data": [{"name": "Budget", "value": "$10,000", "source_chunk_id": "chunk-1"}],
        "recommended_next_steps": [{"action": "Apply", "priority": "important", "rationale": "Because", "source_chunk_id": "chunk-1"}],
        "references": [],
    }

    resp = analyzer._to_response(bundle=bundle, scored_contexts={"summary": []}, payload=fake_payload)
    assert resp.document_id == bundle.document_id
    assert resp.title == "testdoc"
    assert resp.summary == "Test summary"
    assert len(resp.extracted_data) == 1
    assert resp.extracted_data[0].name == "Budget"
    assert resp.recommended_next_steps[0].action == "Apply"
    assert resp.categorized_chunks

    critical_chunks = [chunk for chunk in resp.categorized_chunks if chunk.key == InsightPriority.critical]
    assert len(critical_chunks) == 1
    assert critical_chunks[0].chunk_id == "chunk-1"
    assert "Deadline" in critical_chunks[0].data

    important_chunks = [chunk for chunk in resp.categorized_chunks if chunk.key == InsightPriority.important]
    assert len(important_chunks) == 1
    assert important_chunks[0].chunk_id == "chunk-2"

    processing = analyzer._to_processing_result(
        bundle=bundle,
        legacy=resp,
        payload=fake_payload,
        filename="testdoc.pdf",
        file_size=2048,
        content_type="application/pdf",
        uploaded_at=datetime(2025, 1, 1, 12, 0, 0),
    )

    assert processing.document.name == "testdoc.pdf"
    assert processing.document.size == 2048
    assert processing.summary.key_points == resp.key_highlights
    assert processing.actionable_steps
    assert processing.pipeline_status and len(processing.pipeline_status) == 10
    assert processing.extracted_data.deadlines
