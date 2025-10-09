from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import get_settings
from app.schemas.api import DocumentAnalysisEnvelope
from app.services.analysis_service import DocumentAnalyzer


settings = get_settings()
app = FastAPI(
    title="NVIDIA Document Guidance API",
    version="0.1.0",
    description="AI assistant that classifies, summarises, and guides nonprofits through grant and compliance documents.",
)

# Allow local dev frontends by default.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

analyzer = DocumentAnalyzer()


@app.get("/")
async def root() -> dict[str, str]:
    return {
        "service": app.title,
        "version": app.version,
        "description": app.description or "",
    }


@app.get("/healthz")
async def healthcheck() -> dict[str, str]:
    status = "ok" if settings.nvidia_api_key else "degraded"
    return {"status": status}


@app.post(
    "/v1/documents/analyze",
    response_model=DocumentAnalysisEnvelope,
    summary="Analyse an uploaded PDF and return structured insights",
)
async def analyze_document(file: UploadFile = File(...)) -> DocumentAnalysisEnvelope:
    if file.content_type not in {"application/pdf", "application/octet-stream"}:
        raise HTTPException(status_code=415, detail="Only PDF files are supported at the moment.")

    contents = await file.read()
    if not contents:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    filename = file.filename or "uploaded.pdf"

    return await analyzer.analyze(file_bytes=contents, filename=filename)
