from __future__ import annotations

from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field


class InsightPriority(str, Enum):
    critical = "critical"
    important = "important"
    informational = "informational"


class InsightItem(BaseModel):
    label: str = Field(..., description="Short title for the insight.")
    description: str = Field(..., description="One to two sentence explanation.")
    source_chunk_id: Optional[str] = Field(
        default=None, description="Chunk identifier associated with this insight."
    )


class CategorizedInsights(BaseModel):
    critical: List[InsightItem] = Field(default_factory=list)
    important: List[InsightItem] = Field(default_factory=list)
    informational: List[InsightItem] = Field(default_factory=list)


class ExtractedDataPoint(BaseModel):
    name: str
    value: str
    source_chunk_id: Optional[str] = None


class RecommendedStep(BaseModel):
    action: str
    priority: InsightPriority
    rationale: Optional[str] = None
    due_date: Optional[str] = None
    owner: Optional[str] = None
    source_chunk_id: Optional[str] = None


class SourceReference(BaseModel):
    chunk_id: str
    page_number: Optional[int] = None
    score: float
    preview: str


class CategorizedChunk(BaseModel):
    key: InsightPriority
    chunk_id: str
    data: str
    page_number: Optional[int] = None
    score: Optional[float] = None


class DocumentAnalysisResponse(BaseModel):
    document_id: str
    title: str
    page_count: int
    summary: str
    key_highlights: List[str]
    categorized_insights: CategorizedInsights
    extracted_data: List[ExtractedDataPoint]
    recommended_next_steps: List[RecommendedStep]
    references: List[SourceReference]
    categorized_chunks: List[CategorizedChunk] = Field(default_factory=list)


class GuidanceDebugPayload(BaseModel):
    """Optional debugging artefact that can be toggled on demand."""

    prompt_tokens: int
    completion_tokens: int
    total_cost_estimate_usd: Optional[float] = None


class DocumentAnalysisEnvelope(BaseModel):
    """Top-level response envelope so UI can evolve without breaking changes."""

    result: DocumentAnalysisResponse
    debug: Optional[GuidanceDebugPayload] = None
