from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field, ConfigDict, field_serializer


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


class DocumentMetadata(BaseModel):
    id: str
    name: str
    size: int
    type: str
    uploaded_at: datetime = Field(alias="uploadedAt")

    model_config = ConfigDict(populate_by_name=True)

    @field_serializer("uploaded_at", when_used="json")
    def _serialize_uploaded_at(self, value: datetime) -> str:
        return value.isoformat()


class DocumentClassification(BaseModel):
    category: str
    confidence: float
    subcategories: List[str] = Field(default_factory=list)


class SectionSummary(BaseModel):
    title: str
    content: str
    importance: str


class DocumentSummary(BaseModel):
    title: str
    sections: List[SectionSummary] = Field(default_factory=list)
    key_points: List[str] = Field(default_factory=list, alias="keyPoints")

    model_config = ConfigDict(populate_by_name=True)


class DeadlineInfo(BaseModel):
    description: str
    date: str
    priority: str


class FinancialFigure(BaseModel):
    label: str
    amount: float
    currency: str
    context: str


class ExtractedDataAggregate(BaseModel):
    deadlines: List[DeadlineInfo] = Field(default_factory=list)
    eligibility: List[str] = Field(default_factory=list)
    financial_figures: List[FinancialFigure] = Field(default_factory=list, alias="financialFigures")

    model_config = ConfigDict(populate_by_name=True)


class ActionableStep(BaseModel):
    id: str
    title: str
    description: str
    priority: str
    estimated_time: str = Field(alias="estimatedTime")
    completed: bool

    model_config = ConfigDict(populate_by_name=True)


class PipelineStageStatus(BaseModel):
    stage: str
    status: str
    progress: int = 0
    message: Optional[str] = None


class ProcessingResult(BaseModel):
    document: DocumentMetadata
    classification: DocumentClassification
    summary: DocumentSummary
    extracted_data: ExtractedDataAggregate = Field(alias="extractedData")
    actionable_steps: List[ActionableStep] = Field(default_factory=list, alias="actionableSteps")
    pipeline_status: List[PipelineStageStatus] = Field(default_factory=list, alias="pipelineStatus")

    model_config = ConfigDict(populate_by_name=True)


class GuidanceDebugPayload(BaseModel):
    """Optional debugging artefact that can be toggled on demand."""

    prompt_tokens: int
    completion_tokens: int
    total_cost_estimate_usd: Optional[float] = None


class DocumentAnalysisEnvelope(BaseModel):
    """Top-level response envelope so UI can evolve without breaking changes."""

    result: ProcessingResult
    legacy: Optional[DocumentAnalysisResponse] = None
    debug: Optional[GuidanceDebugPayload] = None
