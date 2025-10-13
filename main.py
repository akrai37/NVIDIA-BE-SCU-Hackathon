from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import get_settings
from app.schemas.api import DocumentAnalysisEnvelope
from app.services.analysis_service import DocumentAnalyzer


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
    response_model=DocumentAnalysisEnvelope,
    summary="Analyze a PDF document with AI",
    tags=["documents"],
    response_description="Comprehensive document analysis with classification, summary, and actionable insights",
    responses={
        200: {
            "description": "Successfully analyzed document",
            "content": {
                "application/json": {
                    "example": {
                        "result": {
                            "document": {
                                "id": "doc_a1b2c3d4",
                                "name": "grant_application_2025.pdf",
                                "size": 1048576,
                                "type": "application/pdf",
                                "uploadedAt": "2025-10-12T14:30:00Z",
                            },
                            "classification": {
                                "category": "Grant Application",
                                "confidence": 0.95,
                                "subcategories": [
                                    "Federal Grant",
                                    "Research Funding",
                                    "STEM Education",
                                ],
                            },
                            "summary": {
                                "title": "National Science Foundation Research Grant 2025",
                                "sections": [
                                    {
                                        "title": "Overview",
                                        "content": "Federal funding opportunity for STEM research...",
                                        "importance": "critical",
                                    }
                                ],
                                "keyPoints": [
                                    "Total funding available: $2.5M",
                                    "Application deadline: December 31, 2025",
                                    "Eligible organizations: 501(c)(3) nonprofits",
                                ],
                            },
                            "extractedData": {
                                "deadlines": [
                                    {
                                        "description": "Final application submission",
                                        "date": "2025-12-31",
                                        "priority": "critical",
                                    }
                                ],
                                "eligibility": [
                                    "501(c)(3) tax-exempt status required",
                                    "Minimum 2 years operational history",
                                ],
                                "financialFigures": [
                                    {
                                        "label": "Maximum grant amount",
                                        "amount": 250000,
                                        "currency": "USD",
                                        "context": "Per organization annual limit",
                                    }
                                ],
                            },
                            "actionableSteps": [
                                {
                                    "id": "step_001",
                                    "title": "Verify eligibility requirements",
                                    "description": "Review all eligibility criteria and gather supporting documentation",
                                    "priority": "critical",
                                    "estimatedTime": "2 hours",
                                    "completed": False,
                                }
                            ],
                            "pipelineStatus": [
                                {
                                    "stage": "extraction",
                                    "status": "completed",
                                    "progress": 100,
                                    "message": "Text extraction completed",
                                }
                            ],
                        }
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
) -> DocumentAnalysisEnvelope:
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
