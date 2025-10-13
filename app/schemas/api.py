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
    content: Optional[str] = Field(
        None, description="Full chunk content for highlighting"
    )
    category: Optional[str] = Field(
        None, description="Category: critical, important, or informational"
    )


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
    financial_figures: List[FinancialFigure] = Field(
        default_factory=list, alias="financialFigures"
    )

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
    actionable_steps: List[ActionableStep] = Field(
        default_factory=list, alias="actionableSteps"
    )
    pipeline_status: List[PipelineStageStatus] = Field(
        default_factory=list, alias="pipelineStatus"
    )

    model_config = ConfigDict(populate_by_name=True)


class GuidanceDebugPayload(BaseModel):
    """Optional debugging artefact that can be toggled on demand."""

    prompt_tokens: int
    completion_tokens: int
    total_cost_estimate_usd: Optional[float] = None


class ChatRequest(BaseModel):
    """Request model for chat endpoint."""

    session_id: str = Field(..., description="Session ID from document analysis")
    question: str = Field(..., description="User's question about the document")
    additional_context: Optional[str] = Field(
        None,
        description="Optional additional context from the document (e.g., specific lines or sections the user wants to reference)",
        example="From page 5: 'The grant deadline is December 31, 2025'",
    )


class ChatResponse(BaseModel):
    """Response model for chat endpoint."""

    answer: str = Field(..., description="AI-generated answer to the user's question")
    session_id: str = Field(..., description="Session ID for continued conversation")
    conversation_length: int = Field(
        ..., description="Number of messages in the conversation"
    )


class DateEvent(BaseModel):
    """Represents a date extracted from the document with context."""

    date: str = Field(..., description="ISO-8601 formatted date (YYYY-MM-DD)")
    event_type: str = Field(
        ..., description="Type of event: due, start, end, renewal, reporting, etc."
    )
    description: str = Field(..., description="Context about this date")
    page_number: Optional[int] = Field(
        None, description="Page number where this was found"
    )
    chunk_id: Optional[str] = Field(None, description="Chunk ID for highlighting")


class FinancialInfo(BaseModel):
    """Represents financial information extracted from the document."""

    amount: float = Field(..., description="Numerical amount")
    currency: str = Field(default="USD", description="Currency code (USD, EUR, etc.)")
    description: str = Field(..., description="Context about this financial figure")
    page_number: Optional[int] = Field(
        None, description="Page number where this was found"
    )
    chunk_id: Optional[str] = Field(None, description="Chunk ID for highlighting")


class QuantityInfo(BaseModel):
    """Represents quantities, percentages, counts, or durations."""

    value: float = Field(..., description="Numerical value")
    unit: Optional[str] = Field(
        None, description="Unit of measurement (days, %, hours, etc.)"
    )
    type: str = Field(..., description="Type: percentage, count, duration, etc.")
    description: str = Field(..., description="Context about this quantity")
    page_number: Optional[int] = Field(
        None, description="Page number where this was found"
    )
    chunk_id: Optional[str] = Field(None, description="Chunk ID for highlighting")


class ContactInfo(BaseModel):
    """Represents contact information extracted from the document."""

    name: Optional[str] = Field(None, description="Contact name")
    role: Optional[str] = Field(None, description="Role or title")
    email: Optional[str] = Field(None, description="Email address")
    phone: Optional[str] = Field(None, description="Phone number")
    page_number: Optional[int] = Field(
        None, description="Page number where this was found"
    )
    chunk_id: Optional[str] = Field(None, description="Chunk ID for highlighting")


class DocumentSummaryData(BaseModel):
    """Short summary of the document."""

    summary: str = Field(..., description="2-3 sentence summary of the document")
    bullet_points: List[str] = Field(
        default_factory=list, description="3-5 key bullet points"
    )


class StructuredExtraction(BaseModel):
    """All structured data extracted from the document."""

    summary: DocumentSummaryData
    dates: List[DateEvent] = Field(default_factory=list)
    financial: List[FinancialInfo] = Field(default_factory=list)
    quantities: List[QuantityInfo] = Field(default_factory=list)
    contacts: List[ContactInfo] = Field(default_factory=list)


class SimplifiedDocumentResponse(BaseModel):
    """Simplified document analysis response with only essential structured data."""

    document_id: str
    title: str
    page_count: int
    session_id: str = Field(..., description="Chat session ID for follow-up questions")
    structured_extraction: StructuredExtraction


class UnifiedDocumentAnalysis(BaseModel):
    """Unified document analysis response with clean structure."""

    document_id: str
    title: str
    page_count: int
    session_id: str = Field(..., description="Chat session ID for follow-up questions")

    # Document metadata
    document_name: str
    document_size: int
    document_type: str
    uploaded_at: datetime = Field(alias="uploadedAt")

    # Classification
    category: str
    confidence: float
    subcategories: List[str] = Field(default_factory=list)

    # Summary and insights
    summary: str
    key_highlights: List[str]
    categorized_insights: CategorizedInsights

    # Extracted structured data
    extracted_data: List[ExtractedDataPoint]
    recommended_next_steps: List[RecommendedStep]

    # Source references - single unified list with all chunks
    references: List[SourceReference]

    model_config = ConfigDict(populate_by_name=True)

    @field_serializer("uploaded_at", when_used="json")
    def _serialize_uploaded_at(self, value: datetime) -> str:
        return value.isoformat()


class DocumentAnalysisEnvelope(BaseModel):
    """Top-level response envelope so UI can evolve without breaking changes."""

    result: ProcessingResult
    session_id: Optional[str] = Field(
        None, description="Chat session ID for follow-up questions"
    )
    legacy: Optional[DocumentAnalysisResponse] = None
    debug: Optional[GuidanceDebugPayload] = None
