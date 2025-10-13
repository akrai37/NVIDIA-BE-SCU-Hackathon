from __future__ import annotations

from datetime import datetime
import re
from typing import Any, Dict, List, Optional, Set, Tuple

from fastapi import HTTPException

from app.core.config import get_settings
from app.schemas.api import (
    ActionableStep,
    CategorizedChunk,
    CategorizedInsights,
    DeadlineInfo,
    DocumentAnalysisEnvelope,
    DocumentAnalysisResponse,
    DocumentClassification,
    DocumentMetadata,
    DocumentSummary,
    ExtractedDataAggregate,
    ExtractedDataPoint,
    FinancialFigure,
    InsightItem,
    InsightPriority,
    PipelineStageStatus,
    ProcessingResult,
    RecommendedStep,
    SectionSummary,
    SourceReference,
)
from app.schemas.document import DocumentBundle
from app.services.document_processor import DocumentProcessor
from app.services.embedding_service import EmbeddingService
from app.services.guidance_service import GuidanceService
from app.services.vector_store import InMemoryVectorStore, ScoredChunk
from app.services.ocr_service import NvidiaOcrService, OcrUnavailableError


class DocumentAnalyzer:
    """Coordinates the full document understanding workflow."""

    def __init__(self) -> None:
        self.settings = get_settings()
        ocr_service = None
        self.ocr_service = None
        if self.settings.enable_ocr:
            try:
                self.ocr_service = NvidiaOcrService()
            except OcrUnavailableError:
                self.ocr_service = None
        self.processor = DocumentProcessor(ocr_service=self.ocr_service)

    async def analyze(
        self,
        *,
        file_bytes: bytes,
        filename: str,
        file_size: int,
        content_type: str,
    ) -> DocumentAnalysisEnvelope:
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

        # Create a chat session for follow-up questions
        session_id = guidance_service.create_session(
            document_id=bundle.document_id,
            document_context=merged_context,
        )

        legacy_response = self._to_response(
            bundle=bundle, scored_contexts=scored_contexts, payload=guidance_payload
        )
        uploaded_at = datetime.utcnow()
        processing_result = self._to_processing_result(
            bundle=bundle,
            legacy=legacy_response,
            payload=guidance_payload,
            filename=filename,
            file_size=file_size,
            content_type=content_type,
            uploaded_at=uploaded_at,
        )
        return DocumentAnalysisEnvelope(
            result=processing_result,
            legacy=legacy_response,
            session_id=session_id,  # Include session ID in response
        )

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
            important=[
                InsightItem(**item) for item in categorized.get("important", [])
            ],
            informational=[
                InsightItem(**item) for item in categorized.get("informational", [])
            ],
        )

        chunk_lookup = {chunk.chunk_id: chunk for chunk in bundle.chunks}
        score_lookup: Dict[str, float] = {}
        for scored_list in scored_contexts.values():
            for scored in scored_list:
                chunk_id = scored.chunk.chunk_id
                score_lookup[chunk_id] = max(
                    score_lookup.get(chunk_id, float("-inf")), scored.score
                )

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

        recommended = [self._build_step(item) for item in next_steps]

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

    def _to_processing_result(
        self,
        *,
        bundle: DocumentBundle,
        legacy: DocumentAnalysisResponse,
        payload: Dict[str, Any],
        filename: str,
        file_size: int,
        content_type: str,
        uploaded_at: datetime,
    ) -> ProcessingResult:
        document_meta = DocumentMetadata(
            id=legacy.document_id,
            name=filename,
            size=file_size,
            type=content_type,
            uploadedAt=uploaded_at,
        )

        classification = self._infer_classification(
            bundle=bundle, legacy=legacy, payload=payload
        )

        sections: List[SectionSummary] = []
        for priority_name, insights in (
            (InsightPriority.critical.value, legacy.categorized_insights.critical),
            (InsightPriority.important.value, legacy.categorized_insights.important),
            (
                InsightPriority.informational.value,
                legacy.categorized_insights.informational,
            ),
        ):
            for insight in insights:
                if not insight.label and not insight.description:
                    continue
                sections.append(
                    SectionSummary(
                        title=insight.label or "Insight",
                        content=insight.description,
                        importance=priority_name,
                    )
                )

        summary = DocumentSummary(
            title=bundle.title or filename,
            sections=sections,
            keyPoints=legacy.key_highlights,
        )

        data_aggregate = self._partition_extracted_data(legacy=legacy)

        actionable_steps = [
            ActionableStep(
                id=f"step-{index+1}",
                title=step.action or "Follow-up",
                description=step.rationale or step.action,
                priority=self._map_priority_to_client(step.priority),
                estimatedTime=self._derive_estimated_time(step),
                completed=False,
            )
            for index, step in enumerate(legacy.recommended_next_steps)
        ]

        pipeline_status = self._default_pipeline_status()

        return ProcessingResult(
            document=document_meta,
            classification=classification,
            summary=summary,
            extractedData=data_aggregate,
            actionableSteps=actionable_steps,
            pipelineStatus=pipeline_status,
        )

    def _infer_classification(
        self,
        *,
        bundle: DocumentBundle,
        legacy: DocumentAnalysisResponse,
        payload: Dict[str, Any],
    ) -> DocumentClassification:
        raw_category = payload.get("category") or payload.get("document_category")
        try:
            confidence = float(payload.get("category_confidence", 0.0))
        except (TypeError, ValueError):
            confidence = 0.0

        corpus = " ".join(
            filter(
                None,
                [
                    bundle.title,
                    legacy.summary,
                    " ".join(legacy.key_highlights),
                    " ".join(
                        insight.description
                        for insight in legacy.categorized_insights.critical
                    ),
                ],
            )
        ).lower()

        if not raw_category:
            if "grant" in corpus:
                raw_category = "Grant Application"
                confidence = max(confidence, 0.72)
            elif "compliance" in corpus:
                raw_category = "Compliance Document"
                confidence = max(confidence, 0.65)
            elif any(
                keyword in corpus for keyword in ["budget", "financial", "finance"]
            ):
                raw_category = "Financial Document"
                confidence = max(confidence, 0.6)
            else:
                raw_category = "General Document"
                confidence = max(confidence, 0.5)

        subcategories: List[str] = payload.get("subcategories") or []
        if not subcategories:
            for collection in (
                legacy.categorized_insights.critical,
                legacy.categorized_insights.important,
            ):
                for insight in collection:
                    if insight.label and insight.label not in subcategories:
                        subcategories.append(insight.label)
                        if len(subcategories) >= 5:
                            break
                if len(subcategories) >= 5:
                    break

        confidence = max(0.0, min(confidence, 1.0))
        return DocumentClassification(
            category=raw_category,
            confidence=confidence,
            subcategories=subcategories,
        )

    def _partition_extracted_data(
        self, legacy: DocumentAnalysisResponse
    ) -> ExtractedDataAggregate:
        deadlines: List[DeadlineInfo] = []
        eligibility: List[str] = []
        financials: List[FinancialFigure] = []

        for point in legacy.extracted_data:
            combined = " ".join(filter(None, [point.name, point.value])).strip()
            lowered = combined.lower()

            if not combined:
                continue

            if any(
                keyword in lowered
                for keyword in ["deadline", "due", "submission", "cut-off"]
            ):
                deadlines.append(
                    DeadlineInfo(
                        description=point.name or point.value,
                        date=point.value or point.name or "",
                        priority="high" if "deadline" in lowered else "medium",
                    )
                )
                continue

            if any(keyword in lowered for keyword in ["eligib", "must", "require"]):
                eligibility.append(point.value or point.name)
                continue

            amount = self._extract_numeric_amount(point.value)
            if amount is not None:
                currency = self._detect_currency(point.value)
                financials.append(
                    FinancialFigure(
                        label=point.name or "Figure",
                        amount=amount,
                        currency=currency,
                        context=point.value or point.name or "",
                    )
                )
                continue

        eligibility.extend(
            insight.description
            for insight in legacy.categorized_insights.important
            if insight.description and "eligib" in insight.description.lower()
        )

        if not deadlines:
            for insight in legacy.categorized_insights.critical:
                composed = " ".join(filter(None, [insight.label, insight.description]))
                lowered = composed.lower()
                if not composed:
                    continue
                if "deadline" not in lowered and "due" not in lowered:
                    continue
                inferred_date = self._extract_date_string(
                    insight.description or insight.label or ""
                )
                deadlines.append(
                    DeadlineInfo(
                        description=insight.description
                        or insight.label
                        or "Key deadline",
                        date=inferred_date
                        or (insight.label or insight.description or ""),
                        priority="high",
                    )
                )
                break

        eligibility = self._dedupe_preserve_order(eligibility)

        return ExtractedDataAggregate(
            deadlines=deadlines,
            eligibility=eligibility,
            financialFigures=financials,
        )

    @staticmethod
    def _map_priority_to_client(priority: InsightPriority) -> str:
        mapping = {
            InsightPriority.critical: "high",
            InsightPriority.important: "medium",
            InsightPriority.informational: "low",
        }
        return mapping.get(priority, "medium")

    @staticmethod
    def _derive_estimated_time(step: RecommendedStep) -> str:
        if step.due_date:
            return f"Due by {step.due_date}"
        if step.owner:
            return f"Owned by {step.owner}"
        return "TBD"

    def _default_pipeline_status(self) -> List[PipelineStageStatus]:
        stages = [
            ("upload", "File received"),
            ("parse", "PDF parsed"),
            ("ocr", "OCR skipped (native text extraction)"),
            ("chunk", "Chunks generated"),
            ("embed", "Embeddings computed"),
            ("store", "Vector store populated"),
            ("retrieve", "Relevant chunks retrieved"),
            ("re-rank", "Chunks ranked"),
            ("llm", "Guidance generated"),
            ("ui", "Response assembled"),
        ]

        return [
            PipelineStageStatus(
                stage=stage,
                status="completed",
                progress=100,
                message=message,
            )
            for stage, message in stages
        ]

    @staticmethod
    def _extract_numeric_amount(value: Optional[str]) -> Optional[float]:
        if not value:
            return None
        cleaned = value.replace(",", "")
        number_match = re.search(r"(-?\d+(?:\.\d+)?)", cleaned)
        if not number_match:
            return None
        try:
            return float(number_match.group(1))
        except ValueError:
            return None

    @staticmethod
    def _detect_currency(value: Optional[str]) -> str:
        if not value:
            return "USD"
        if "%" in value:
            return "%"
        for symbol in ("$", "€", "£"):
            if symbol in value:
                return symbol
        return "USD"

    @staticmethod
    def _extract_date_string(value: str) -> Optional[str]:
        if not value:
            return None
        patterns = [
            r"\b\d{4}-\d{2}-\d{2}\b",
            r"\b\d{1,2}/\d{1,2}/\d{2,4}\b",
            r"\b(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\s+\d{1,2}(?:,\s*\d{4})?\b",
        ]
        lowered_value = value.lower()
        for pattern in patterns:
            match = re.search(pattern, lowered_value, flags=re.IGNORECASE)
            if match:
                return match.group(0)
        return None

    @staticmethod
    def _dedupe_preserve_order(values: List[str]) -> List[str]:
        seen: Set[str] = set()
        result: List[str] = []
        for item in values:
            if not item:
                continue
            if item in seen:
                continue
            seen.add(item)
            result.append(item)
        return result

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
