# NVIDIA Document Guidance Backend

API backend for the NVIDIA Hackathon prototype that delivers real-time document understanding for nonprofits. Upload a PDF and the service classifies critical information, extracts deadlines and financial data, and suggests action steps using NVIDIA Nemotron models hosted on NIM.

## 🧱 Architecture overview

1. **Upload** – UI sends the PDF to `POST /v1/documents/analyze`.
2. **Parse** – `pypdf` streams text; if pages are image-only the backend renders them with `pypdfium2` and calls NVIDIA's OCR NIM to recover text.
3. **Chunk** – `RecursiveCharacterTextSplitter` slices context-aware segments (~900 chars, 150 overlap).
4. **Embed** – Chunks are embedded with NVIDIA's `embed-qa-4` model via NGC/NIM.
5. **Store & Retrieve** – A lightweight cosine-similarity in-memory vector store powers targeted lookups.
6. **Reason** – NVIDIA Nemotron (`llama-3.1-nemotron-70b-instruct` by default) turns retrieved evidence into structured JSON with critical/important/informational lanes.
7. **Respond** – FastAPI returns a typed response ready for the frontend to render guidance cards and next steps.

## 🎬 Demo

Watch a walkthrough of the document analysis pipeline in action:

https://github.com/user-attachments/assets/1ef399ab-19b4-4828-8110-70ff09abbec2


## ✅ Prerequisites

- Python 3.13+
- [`uv`](https://github.com/astral-sh/uv) for dependency management (or fall back to `pip`)
- NVIDIA API credentials with access to Nemotron NIM endpoints

Set the following environment variables before running the app:

| Variable | Purpose | Example |
| --- | --- | --- |
| `NVIDIA_API_KEY` | Bearer token for NVIDIA AI endpoints | `nvapi-xxxxxxxx` |
| `NVIDIA_BASE_URL` | (Optional) Override the default API gateway | `https://integrate.api.nvidia.com/v1` |
| `NVIDIA_EMBEDDING_MODEL` | (Optional) Embedding model name | `nvidia/embed-qa-4` |
| `NVIDIA_LLM_MODEL` | (Optional) Nemotron model name | `nvidia/llama-3.1-nemotron-70b-instruct` |
| `ENABLE_OCR` | Enable NVIDIA OCR fallback for scanned PDFs | `true` |
| `NVIDIA_OCR_MODEL` | (Optional) OCR NIM identifier | `nvidia/ocr-nvble-12b-vision` |
| `OCR_RENDER_SCALE` | (Optional) Render scale when rasterising pages for OCR | `2.0` |

## 🚀 Run locally

```bash
uv sync
uv run uvicorn main:app --reload --port 8000
```

The OpenAPI docs live at `http://localhost:8000/docs`.

## 🔌 API contract

### `POST /v1/documents/analyze`

- **Body**: `multipart/form-data` with a `file` field containing a PDF
- **Response**: Structured JSON envelope containing:
  - Document metadata (id, title, pages)
  - Executive summary + key highlights
  - Categorised insights (critical / important / informational)
  - Extracted data points (deadlines, figures, requirements)
  - Recommended next steps with priority and optional due dates
  - Source references (chunk id, page, similarity score)

Example request once the server is running:

```bash
curl -X POST \
  -H "Authorization: Bearer $NVIDIA_API_KEY" \
  -F "file=@samples/grant_brief.pdf" \
  http://localhost:8000/v1/documents/analyze
```

> The API key does not need to be sent with the request—the backend uses it when it calls NIM. The header above is only necessary if you place the service behind your own auth layer.

## 📄 Sample documents

A realistic grant template works best. To generate a quick sample for local testing:

1. Create a short document (Google Docs, Word, or Markdown) with headings for deadlines, eligibility, financials, and reporting.
2. Export it as PDF and save it under `samples/grant_brief.pdf`.
3. Call the endpoint using the `curl` example above.

If OCR is required (e.g., scanned PDFs), set `ENABLE_OCR=true` and ensure your NVIDIA account has access to the specified OCR NIM. The backend will render image-only pages, submit them to the OCR model, and merge the results with native text extraction. Without an API key or OCR access the service falls back to its previous behaviour and returns a descriptive error when no readable text is found.

## 🧠 Prompt + retrieval strategy

- Targeted semantic queries drive retrieval for summary, deadlines, eligibility, financials, and next steps.
- Retrieved chunks are merged into a single prompt block with chunk IDs, page numbers, and cosine similarity scores.
- The system prompt forces strict JSON output and category separation, making frontend rendering deterministic.

## 🔍 Observability hooks

- `GET /healthz` returns `ok` when the API key is present, otherwise `degraded` (still runs but model calls will fail).
- Responses optionally include a `debug` field (wired, but currently unused) for token usage—extend `DocumentAnalysisEnvelope` if cost tracking is needed.

## 🧪 Testing

Basic smoke tests can be run with `pytest` once you add test cases. During development you can exercise the pipeline with FastAPI's interactive docs or the provided `curl` snippet.

## 🗺️ Next steps

- Swap the in-memory store for a persistent vector DB (Milvus, pgvector, or Pinecone).
- Stream partial results to the UI for better perceived latency.
- Log rich telemetry (chunk spans, confidence) to help non-technical users trust the AI output.
