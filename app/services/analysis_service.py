from __future__ import annotations

from typing import Any, Dict, List, Set, Tuple

from fastapi import HTTPException

from app.core.config import get_settings
from app.schemas.api import (
    CategorizedChunk,
    CategorizedInsights,
    DocumentAnalysisEnvelope,
    DocumentAnalysisResponse,
    ExtractedDataPoint,
    InsightItem,
    InsightPriority,
    RecommendedStep,
    SourceReference,
)
from app.schemas.document import DocumentBundle
from app.services.document_processor import DocumentProcessor
from app.services.embedding_service import EmbeddingService
from app.services.guidance_service import GuidanceService
from app.services.vector_store import InMemoryVectorStore, ScoredChunk


class DocumentAnalyzer:
    """Coordinates the full document understanding workflow."""

    def __init__(self) -> None:
        self.settings = get_settings()
        self.processor = DocumentProcessor()

    async def analyze(self, *, file_bytes: bytes, filename: str) -> DocumentAnalysisEnvelope:
        bundle = self._process_document(file_bytes=file_bytes, filename=filename)
        try:
            embedding_service = EmbeddingService()
            vector_store = InMemoryVectorStore(embedding_service)
            vector_store.add_chunks(bundle.chunks)
        except RuntimeError as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

        guidance_service = GuidanceService()

        queries = self._build_queries()
        scored_contexts = {
            label: vector_store.similarity_search(prompt, top_k=self.settings.top_k)
            for label, prompt in queries.items()
        }
        merged_context = self._build_context(scored_contexts)

        guidance_payload = await guidance_service.generate(
            context=merged_context,
            title=bundle.title,
            pages=bundle.page_count,
        )

        response = self._to_response(
            bundle=bundle, scored_contexts=scored_contexts, payload=guidance_payload
        )
        return DocumentAnalysisEnvelope(result=response)

    def _process_document(self, *, file_bytes: bytes, filename: str) -> DocumentBundle:
        try:
            return self.processor.process(file_bytes=file_bytes, filename=filename)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    def _build_queries(self) -> Dict[str, str]:
        return {
            "summary": "Summarise the overall intent, funding purpose, and strategic fit.",
            "deadlines": "List every deadline, reporting cadence, and submission window.",
            "eligibility": "Extract eligibility criteria and any compliance obligations.",
            "financials": "Extract all financial figures, budgets, match requirements, and disbursement logic.",
            "next_steps": "Recommend concrete next steps for a nonprofit operations team after reading this document.",
        }

    def _build_context(self, scored_contexts: Dict[str, List[ScoredChunk]]) -> str:
        lines: List[str] = []
        for label, scored_chunks in scored_contexts.items():
            lines.append(f"## {label.upper()}")
            for item in scored_chunks:
                chunk = item.chunk
                preview = chunk.content.replace("\n", " ")
                lines.append(
                    f"[{chunk.chunk_id}] page={chunk.page_number} score={item.score:.3f}: {preview}"
                )
            lines.append("")
        return "\n".join(lines)

    def _to_response(
        self,
        *,
        bundle: DocumentBundle,
        scored_contexts: Dict[str, List[ScoredChunk]],
        payload: Dict[str, Any],
    ) -> DocumentAnalysisResponse:
        categorized = payload.get("categorized_insights", {}) or {}
        extracted = payload.get("extracted_data", []) or []
        next_steps = payload.get("recommended_next_steps", []) or []
        references_payload = payload.get("references", []) or []

        key_highlights = payload.get("key_highlights", []) or []
        if isinstance(key_highlights, str):
            key_highlights = [key_highlights]

        categorized_insights = CategorizedInsights(
            critical=[InsightItem(**item) for item in categorized.get("critical", [])],
            important=[InsightItem(**item) for item in categorized.get("important", [])],
            informational=[InsightItem(**item) for item in categorized.get("informational", [])],
        )

        chunk_lookup = {chunk.chunk_id: chunk for chunk in bundle.chunks}
        score_lookup: Dict[str, float] = {}
        for scored_list in scored_contexts.values():
            for scored in scored_list:
                chunk_id = scored.chunk.chunk_id
                score_lookup[chunk_id] = max(score_lookup.get(chunk_id, float("-inf")), scored.score)

        categorized_chunks: List[CategorizedChunk] = []
        seen_pairs: Set[Tuple[InsightPriority, str]] = set()
        for priority_name, insight_items in categorized.items():
            try:
                priority = InsightPriority(priority_name)
            except ValueError:
                continue
            for item in insight_items or []:
                chunk_id = item.get("source_chunk_id")
                if not chunk_id:
                    continue
                if chunk_id not in chunk_lookup:
                    continue
                key = (priority, chunk_id)
                if key in seen_pairs:
                    continue
                seen_pairs.add(key)
                chunk = chunk_lookup[chunk_id]
                categorized_chunks.append(
                    CategorizedChunk(
                        key=priority,
                        chunk_id=chunk_id,
                        data=chunk.content,
                        page_number=chunk.page_number,
                        score=score_lookup.get(chunk_id),
                    )
                )

        extracted_points: List[ExtractedDataPoint] = []
        for item in extracted:
            name = item.get("name") or item.get("label") or "Unlabelled datapoint"
            value = item.get("value") or item.get("answer") or ""
            extracted_points.append(
                ExtractedDataPoint(
                    name=name,
                    value=value,
                    source_chunk_id=item.get("source_chunk_id"),
                )
            )

        recommended = [
            self._build_step(item)
            for item in next_steps
        ]

        references = []
        for ref in references_payload:
            try:
                score = float(ref.get("score", 0))
            except (TypeError, ValueError):
                score = 0.0
            chunk_id = str(ref.get("chunk_id", "unknown"))
            references.append(
                SourceReference(
                    chunk_id=chunk_id,
                    page_number=ref.get("page_number"),
                    score=score,
                    preview=ref.get("preview", ""),
                )
            )

        if not references:
            references = self._fallback_references(scored_contexts)

        return DocumentAnalysisResponse(
            document_id=bundle.document_id,
            title=bundle.title,
            page_count=bundle.page_count,
            summary=payload.get("summary", ""),
            key_highlights=key_highlights,
            categorized_insights=categorized_insights,
            extracted_data=extracted_points,
            recommended_next_steps=recommended,
            references=references,
            categorized_chunks=categorized_chunks,
        )

    def _build_step(self, payload: Dict[str, Any]) -> RecommendedStep:
        raw_priority = payload.get("priority", "informational")
        try:
            priority = InsightPriority(raw_priority)
        except ValueError:
            priority = InsightPriority.informational

        return RecommendedStep(
            action=payload.get("action", ""),
            priority=priority,
            rationale=payload.get("rationale"),
            due_date=payload.get("due_date"),
            owner=payload.get("owner"),
            source_chunk_id=payload.get("source_chunk_id"),
        )

    def _fallback_references(
        self, scored_contexts: Dict[str, List[ScoredChunk]]
    ) -> List[SourceReference]:
        reference_map: Dict[str, SourceReference] = {}
        for scored_list in scored_contexts.values():
            for item in scored_list:
                chunk = item.chunk
                if chunk.chunk_id in reference_map:
                    reference_map[chunk.chunk_id].score = max(
                        reference_map[chunk.chunk_id].score, item.score
                    )
                    continue
                reference_map[chunk.chunk_id] = SourceReference(
                    chunk_id=chunk.chunk_id,
                    page_number=chunk.page_number,
                    score=item.score,
                    preview=chunk.content[:200],
                )
        return list(reference_map.values())
