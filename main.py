from fastapi import FastAPI, File, HTTPException, UploadFile, Body
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict, Any

from app.core.config import get_settings
from app.schemas.api import (
    ChatRequest,
    ChatResponse,
    SimplifiedDocumentResponse,
    UnifiedDocumentAnalysis,
)
from app.services.analysis_service import DocumentAnalyzer
from app.services.guidance_service import GuidanceService


settings = get_settings()
app = FastAPI(
    title="NVIDIA Document Guidance API",
    version="0.1.0",
    description="""
## AI-Powered Document Analysis for Nonprofits

An intelligent assistant that classifies, summarizes, and guides nonprofits through grant and compliance documents using NVIDIA AI technology.

### Key Features

* 📄 **PDF Document Analysis**: Upload and analyze PDF documents
* 🎯 **Smart Classification**: Automatic document categorization with confidence scores
* 📝 **Intelligent Summarization**: Extract key points and executive summaries
* 💡 **Actionable Insights**: Get prioritized next steps and recommendations
* 📊 **Data Extraction**: Identify deadlines, eligibility criteria, and financial figures
* 🔍 **Source References**: Traceable insights with page references

### Quick Start

1. Upload a PDF document using the `/v1/documents/analyze` endpoint
2. Receive comprehensive analysis including classification, summary, and actionable steps
3. Use the extracted data to streamline your grant and compliance processes

### Authentication

Currently, no authentication is required for API access. NVIDIA API key is configured server-side.
    """,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_tags=[
        {
            "name": "general",
            "description": "General API information and health monitoring",
        },
        {
            "name": "documents",
            "description": "Document analysis and processing operations",
        },
        {
            "name": "chat",
            "description": "Chat and follow-up questions with session management",
        },
    ],
    contact={
        "name": "API Support",
        "url": "https://github.com/Anmol-tech/NVIDIA-BE",
        "email": "support@example.com",
    },
    license_info={
        "name": "MIT",
        "url": "https://opensource.org/licenses/MIT",
    },
)

# Allow local dev frontends by default.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

analyzer = DocumentAnalyzer()


@app.get(
    "/",
    tags=["general"],
    summary="Get API service information",
    response_description="Service metadata including name, version, and description",
)
async def root() -> dict[str, str]:
    """
    ## Service Information

    Returns basic metadata about the NVIDIA Document Guidance API service.

    **Returns:**
    - `service`: Name of the API service
    - `version`: Current version number
    - `description`: Brief service description
    """
    return {
        "service": app.title,
        "version": app.version,
        "description": app.description or "",
    }


@app.get(
    "/healthz",
    tags=["general"],
    summary="Health check endpoint",
    response_description="Current health status of the service",
)
async def healthcheck() -> dict[str, str]:
    """
    ## Health Check

    Monitors the operational status of the API service.

    **Status Values:**
    - `ok`: Service is fully operational with NVIDIA API key configured
    - `degraded`: Service is running but NVIDIA API key is missing or invalid

    Use this endpoint for:
    - Load balancer health checks
    - Monitoring and alerting
    - Service availability verification
    """
    status = "ok" if settings.nvidia_api_key else "degraded"
    return {"status": status}


@app.post(
    "/v1/documents/analyze",
    response_model=UnifiedDocumentAnalysis,
    summary="Analyze a PDF document with AI",
    tags=["documents"],
    response_description="Comprehensive document analysis with classification, summary, and actionable insights",
    responses={
        200: {
            "description": "Successfully analyzed document",
            "content": {
                "application/json": {
                    "example": {
                        "document_id": "doc_a1b2c3d4",
                        "title": "National Science Foundation Research Grant 2025",
                        "page_count": 24,
                        "session_id": "session_xyz123",
                        "document_name": "grant_application_2025.pdf",
                        "document_size": 1048576,
                        "document_type": "application/pdf",
                        "uploadedAt": "2025-10-12T14:30:00Z",
                        "category": "Grant Application",
                        "confidence": 0.95,
                        "subcategories": [
                            "Federal Grant",
                            "Research Funding",
                            "STEM Education",
                        ],
                        "summary": "A comprehensive guide for federal grant applications...",
                        "key_highlights": [
                            "Total funding available: $2.5M",
                            "Application deadline: December 31, 2025",
                            "Eligible organizations: 501(c)(3) nonprofits",
                        ],
                        "categorized_insights": {
                            "critical": [
                                {
                                    "label": "Application Deadline",
                                    "description": "Final submission must be completed by December 31, 2025",
                                    "source_chunk_id": "chunk-3",
                                }
                            ],
                            "important": [
                                {
                                    "label": "Eligibility Requirements",
                                    "description": "501(c)(3) status and 2 years operational history required",
                                    "source_chunk_id": "chunk-7",
                                }
                            ],
                            "informational": [],
                        },
                        "extracted_data": [
                            {
                                "name": "Maximum grant amount",
                                "value": "$250,000",
                                "source_chunk_id": "chunk-5",
                            }
                        ],
                        "recommended_next_steps": [
                            {
                                "action": "Verify eligibility requirements",
                                "priority": "critical",
                                "rationale": "Ensure organization meets all criteria",
                                "due_date": None,
                                "owner": None,
                                "source_chunk_id": "chunk-7",
                            }
                        ],
                        "references": [
                            {
                                "chunk_id": "chunk-3",
                                "page_number": 5,
                                "score": 0.95,
                                "preview": "Application deadline is December 31, 2025...",
                                "content": "The complete text content of this chunk for highlighting purposes. Application deadline is December 31, 2025 at 11:59 PM EST. Late submissions will not be accepted under any circumstances.",
                                "category": "critical",
                            },
                            {
                                "chunk_id": "chunk-7",
                                "page_number": 8,
                                "score": 0.88,
                                "preview": "Eligibility requirements include 501(c)(3) status...",
                                "content": "Eligibility requirements include current 501(c)(3) tax-exempt status and minimum 2 years of operational history.",
                                "category": "important",
                            },
                            {
                                "chunk_id": "chunk-12",
                                "page_number": 15,
                                "score": 0.75,
                                "preview": "Additional program information and contact details...",
                                "content": "For additional information about the program, please contact the grants office at grants@example.org",
                                "category": None,
                            },
                        ],
                    }
                }
            },
        },
        400: {
            "description": "Bad Request - Invalid or empty file",
            "content": {
                "application/json": {"example": {"detail": "Uploaded file is empty."}}
            },
        },
        415: {
            "description": "Unsupported Media Type - Non-PDF file uploaded",
            "content": {
                "application/json": {
                    "example": {"detail": "Only PDF files are supported at the moment."}
                }
            },
        },
    },
)
async def analyze_document(
    file: UploadFile = File(
        ...,
        description="PDF document to analyze (max recommended size: 10MB)",
    )
) -> UnifiedDocumentAnalysis:
    """
    ## Analyze PDF Document
    
    Upload and analyze a PDF document to receive comprehensive AI-powered insights tailored for nonprofit 
    organizations navigating grant applications and compliance requirements.
    
    ### Process Flow
    
    The analysis pipeline consists of the following stages:
    
    1. **📥 Upload & Validation**
       - Validates file type (PDF only)
       - Checks file integrity
       - Generates unique document ID
    
    2. **🔍 Content Extraction**
       - Extracts text from PDF pages
       - Preserves document structure
       - Handles multi-page documents
    
    3. **🤖 AI Classification**
       - Categorizes document type using NVIDIA AI
       - Assigns confidence scores
       - Identifies relevant subcategories
    
    4. **📝 Intelligent Summarization**
       - Generates executive summary
       - Extracts key highlights
       - Identifies important sections
    
    5. **📊 Data Extraction**
       - Identifies critical deadlines
       - Extracts eligibility criteria
       - Parses financial figures and amounts
    
    6. **💡 Guidance Generation**
       - Creates actionable next steps
       - Prioritizes tasks by urgency
       - Provides time estimates
    
    ### Request Parameters
    
    **file** (required): PDF file to analyze
    - **Type**: multipart/form-data
    - **Content-Type**: `application/pdf` or `application/octet-stream`
    - **Size**: Maximum 10MB recommended
    - **Format**: Valid PDF document
    
    ### Response Structure
    
    The API returns a comprehensive `DocumentAnalysisEnvelope` containing:
    
    #### 📄 Document Metadata
    - Unique identifier and upload timestamp
    - File name, size, and type
    
    #### 🎯 Classification
    - Primary category (e.g., "Grant Application", "Compliance Form")
    - Confidence score (0.0 to 1.0)
    - Relevant subcategories for refined categorization
    
    #### 📋 Summary
    - Extracted or inferred document title
    - Structured sections with importance levels
    - Key points highlighting critical information
    
    #### 📊 Extracted Data
    - **Deadlines**: Time-sensitive dates with priorities
    - **Eligibility**: Requirements and qualification criteria
    - **Financial Figures**: Amounts, currency, and contextual information
    
    #### ✅ Actionable Steps
    - Prioritized task list based on document content
    - Estimated completion time for each action
    - Detailed descriptions and rationale
    
    #### 🔄 Pipeline Status
    - Real-time processing stage information
    - Progress indicators for each stage
    - Status messages and completion tracking
    
    ### Example Usage
    
    **cURL:**
    ```bash
    curl -X POST "http://localhost:8000/v1/documents/analyze" \\
         -H "accept: application/json" \\
         -H "Content-Type: multipart/form-data" \\
         -F "file=@grant_application.pdf"
    ```
    
    **Python (httpx):**
    ```python
    import httpx
    
    with open("grant_application.pdf", "rb") as f:
        files = {"file": ("grant_application.pdf", f, "application/pdf")}
        response = httpx.post(
            "http://localhost:8000/v1/documents/analyze",
            files=files
        )
        result = response.json()
    ```
    
    **JavaScript (Fetch API):**
    ```javascript
    const formData = new FormData();
    formData.append('file', fileInput.files[0]);
    
    const response = await fetch('http://localhost:8000/v1/documents/analyze', {
      method: 'POST',
      body: formData
    });
    
    const result = await response.json();
    ```
    
    ### Error Handling
    
    The API returns appropriate HTTP status codes:
    
    - **200 OK**: Document successfully analyzed
    - **400 Bad Request**: Empty file or invalid content
    - **415 Unsupported Media Type**: Non-PDF file uploaded
    - **422 Unprocessable Entity**: Validation errors
    - **500 Internal Server Error**: Processing failure
    
    ### Performance Notes
    
    - Processing time varies based on document length (typically 5-30 seconds)
    - Longer documents may take additional time for comprehensive analysis
    - Results include source references for transparency and verification
    
    ### Best Practices
    
    1. **File Preparation**: Ensure PDFs are text-based (not scanned images) for optimal results
    2. **File Size**: Keep files under 10MB for faster processing
    3. **Error Handling**: Implement retry logic for transient failures
    4. **Rate Limiting**: Be mindful of API usage to avoid service disruption
    """
    if file.content_type not in {"application/pdf", "application/octet-stream"}:
        raise HTTPException(
            status_code=415, detail="Only PDF files are supported at the moment."
        )

    contents = await file.read()
    if not contents:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    filename = file.filename or "uploaded.pdf"

    return await analyzer.analyze(
        file_bytes=contents,
        filename=filename,
        file_size=len(contents),
        content_type=file.content_type or "application/octet-stream",
    )


@app.post(
    "/v1/documents/analyze/simplified",
    response_model=SimplifiedDocumentResponse,
    summary="Analyze a PDF document with simplified structured output",
    tags=["documents"],
    response_description="Simplified document analysis with structured extraction (summary, dates, financial, quantities, contacts)",
)
async def analyze_document_simplified(
    file: UploadFile = File(
        ...,
        description="PDF document to analyze (max recommended size: 10MB)",
    )
) -> SimplifiedDocumentResponse:
    """
    ## Analyze PDF Document (Simplified)
    
    Upload and analyze a PDF document to receive structured data extraction only.
    This endpoint returns a cleaner, simplified response focused on extracting specific data types.
    
    ### Extracted Data Types
    
    1. **📝 Summary**
       - 2-3 sentence document summary
       - 3-5 key bullet points
    
    2. **📅 Dates**
       - All dates in ISO-8601 format (YYYY-MM-DD)
       - Event types: due, start, end, renewal, reporting
       - Context and page references
    
    3. **💰 Financial Information**
       - Amounts with currency
       - Context and descriptions
       - Page references for highlighting
    
    4. **📊 Quantities**
       - Percentages, counts, durations
       - Units and types
       - Context and page references
    
    5. **👥 Contacts**
       - Names and roles
       - Email addresses and phone numbers
       - Page references
    
    ### Response Structure
    
    Returns a `SimplifiedDocumentResponse` containing:
    - `document_id`: Unique document identifier
    - `title`: Document title
    - `page_count`: Number of pages
    - `session_id`: Chat session ID for follow-up questions
    - `structured_extraction`: All extracted structured data with page references
    
    ### Example Usage
    
    **cURL:**
    ```bash
    curl -X POST "http://localhost:8000/v1/documents/analyze/simplified" \\
      -F "file=@grant_document.pdf"
    ```
    """
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400, detail="Only PDF files (.pdf) are supported."
        )

    contents = await file.read()
    if not contents:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    filename = file.filename or "uploaded.pdf"

    return await analyzer.analyze_simplified(
        file_bytes=contents,
        filename=filename,
        file_size=len(contents),
        content_type=file.content_type or "application/octet-stream",
    )


@app.get(
    "/v1/sessions/{session_id}",
    tags=["chat"],
    summary="Check session status",
    response_description="Session information and activity status",
)
async def get_session_status(session_id: str) -> dict:
    """
    ## Check Session Status

    Check if a session exists and get information about its activity status.
    Useful for debugging session expiration issues.

    ### Response

    Returns session information including:
    - Whether the session exists
    - When it was created
    - Last activity time
    - Number of messages in conversation
    - Time until expiration

    ### Example Usage

    ```bash
    curl -X GET "http://localhost:8000/v1/sessions/{session_id}"
    ```
    """
    guidance_service = GuidanceService()
    session_manager = guidance_service._session_manager

    session = session_manager.get_session(session_id)

    if not session:
        return {
            "exists": False,
            "session_id": session_id,
            "message": "Session not found or expired",
        }

    from datetime import datetime

    now = datetime.now()
    time_until_expiry = session_manager._timeout - (now - session.last_activity)

    return {
        "exists": True,
        "session_id": session_id,
        "document_id": session.document_id,
        "created_at": session.created_at.isoformat(),
        "last_activity": session.last_activity.isoformat(),
        "message_count": len(session.messages),
        "timeout_minutes": session_manager._timeout.total_seconds() / 60,
        "time_until_expiry_minutes": time_until_expiry.total_seconds() / 60,
        "is_active": time_until_expiry.total_seconds() > 0,
    }


@app.post(
    "/v1/chat",
    tags=["chat"],
    summary="Chat with document using session",
    response_description="Answer to user's question with session context",
    response_model=ChatResponse,
)
async def chat_with_document(
    request: ChatRequest = Body(
        ...,
        description="Chat request with session ID, question, and optional additional context",
    ),
) -> ChatResponse:
    """
    ## Chat with Document
    
    Ask follow-up questions about a previously analyzed document using the session ID.
    The user can optionally provide additional context (e.g., specific lines from the document) to focus the AI's attention.
    
    ### Process
    
    1. **Maintain Context**: Uses the session ID to access document context and conversation history
    2. **Additional Context**: Optionally accepts specific text excerpts from the document for focused analysis
    3. **Answer Questions**: Provides answers based on the document content and any additional context
    4. **Track History**: Maintains conversation history for contextual responses
    
    ### Request Body
    
    ```json
    {
      "session_id": "uuid-from-document-analysis",
      "question": "What are the eligibility requirements?",
      "additional_context": "From page 5: The grant is available to nonprofits with annual budgets under $1M"
    }
    ```
    
    ### Response
    
    ```json
    {
      "answer": "Based on the document...",
      "session_id": "same-session-id",
      "conversation_length": 3
    }
    ```
    
    ### Example Usage
    
    **Step 1**: Analyze a document and get session ID
    ```bash
    RESPONSE=$(curl -X POST "http://localhost:8000/v1/documents/analyze" \\
         -F "file=@grant.pdf")
    SESSION_ID=$(echo $RESPONSE | jq -r '.session_id')
    ```
    
    **Step 2**: Ask follow-up questions (basic)
    ```bash
    curl -X POST "http://localhost:8000/v1/chat" \\
         -H "Content-Type: application/json" \\
         -d '{
           "session_id": "'$SESSION_ID'",
           "question": "What is the deadline for this grant?"
         }'
    ```
    
    **Step 3**: Ask with additional context (focused)
    ```bash
    curl -X POST "http://localhost:8000/v1/chat" \\
         -H "Content-Type: application/json" \\
         -d '{
           "session_id": "'$SESSION_ID'",
           "question": "Can you explain this requirement in detail?",
           "additional_context": "From page 7: Organizations must demonstrate community impact through measurable outcomes"
         }'
    ```
    
    ### Features
    
    - 🔄 **Session Persistence**: Maintains context across multiple questions
    - 🧠 **Conversation Memory**: References previous Q&A in responses
    - 📄 **Document Context**: Always grounded in the analyzed document
    - 🎯 **Additional Context**: Optionally focus on specific document sections
    - ⏱️ **Auto-Cleanup**: Sessions expire after 30 minutes of inactivity
    
    ### Error Responses
    
    - **400**: Session ID required or no document context available
    - **404**: Session not found or expired
    - **500**: Processing error
    """
    try:
        guidance_service = GuidanceService()
        result = await guidance_service.chat(
            question=request.question,
            session_id=request.session_id,
            additional_context=request.additional_context,
        )
        return ChatResponse(**result)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete(
    "/v1/chat/session/{session_id}",
    tags=["chat"],
    summary="Clear chat session",
    response_description="Confirmation of session deletion",
)
async def clear_chat_session(session_id: str) -> Dict[str, str]:
    """
    ## Clear Chat Session

    Delete a chat session and its conversation history.

    ### When to Use

    - User wants to start fresh with the same document
    - Cleaning up after analysis is complete
    - Managing active sessions

    ### Response

    ```json
    {
      "message": "Session cleared successfully"
    }
    ```

    ### Example

    ```bash
    curl -X DELETE "http://localhost:8000/v1/chat/session/{session_id}"
    ```
    """
    try:
        guidance_service = GuidanceService()
        if guidance_service.clear_session(session_id):
            return {"message": "Session cleared successfully"}
        else:
            raise HTTPException(status_code=404, detail="Session not found")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
