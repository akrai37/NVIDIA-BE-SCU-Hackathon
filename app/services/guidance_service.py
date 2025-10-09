from __future__ import annotations

import json
from typing import Any, Dict

from fastapi import HTTPException
from fastapi.concurrency import run_in_threadpool
from langchain.prompts import ChatPromptTemplate
from langchain_nvidia_ai_endpoints import ChatNVIDIA

from app.core.config import get_settings


_SYSTEM_PROMPT = """You are an expert document analyst helping small nonprofits understand complex grant, finance, and compliance documents. You must:
- return ONLY valid JSON with double quotes, no markdown fences.
- organise findings into critical, important, and informational categories.
- highlight deadlines, eligibility, financial requirements, and follow-up actions.
- include references back to the provided chunk IDs where possible.
If information is missing, respond with an empty array for that section."""

_USER_PROMPT = """Document context:
{context}

Document metadata:
- Title: {title}
- Total pages: {pages}

Provide a JSON object with the following shape:
{
  "summary": string,
  "key_highlights": string[],
  "categorized_insights": {
    "critical": InsightItem[],
    "important": InsightItem[],
    "informational": InsightItem[]
  },
  "extracted_data": ExtractedDataPoint[],
  "recommended_next_steps": RecommendedStep[],
  "references": SourceReference[]
}

Where InsightItem = {"label": string, "description": string, "source_chunk_id": string | null}
ExtractedDataPoint = {"name": string, "value": string, "source_chunk_id": string | null}
RecommendedStep = {"action": string, "priority": "critical" | "important" | "informational", "rationale": string | null, "due_date": string | null, "owner": string | null, "source_chunk_id": string | null}
SourceReference = {"chunk_id": string, "page_number": number | null, "score": number, "preview": string}
"""


class GuidanceService:
    """Generate structured guidance using NVIDIA Nemotron models via NIM."""

    def __init__(self) -> None:
        settings = get_settings()
        if not settings.nvidia_api_key:
            raise RuntimeError("NVIDIA_API_KEY is not configured for guidance generation.")
        self._client = ChatNVIDIA(
            model=settings.nvidia_llm_model,
            api_key=settings.nvidia_api_key,
            base_url=settings.nvidia_base_url,
            temperature=settings.guidance_temperature,
        )
        self._prompt = ChatPromptTemplate.from_messages(
            [
                ("system", _SYSTEM_PROMPT),
                ("user", _USER_PROMPT),
            ]
        )

    async def generate(self, *, context: str, title: str, pages: int) -> Dict[str, Any]:
        chain = self._prompt | self._client
        response = await run_in_threadpool(
            chain.invoke,
            {
                "context": context,
                "title": title,
                "pages": pages,
            },
        )
        content = self._coerce_content(response)

        try:
            return json.loads(content)
        except json.JSONDecodeError as exc:  # pragma: no cover - depends on model output
            raise HTTPException(
                status_code=502,
                detail="Model response could not be parsed as JSON.",
            ) from exc

    @staticmethod
    def _coerce_content(response: Any) -> str:
        if response is None:
            return "{}"

        if isinstance(response, str):
            return response

        content = getattr(response, "content", None)
        if isinstance(content, str):
            return content

        if isinstance(response, list):
            parts: list[str] = []
            for part in response:
                if isinstance(part, dict):
                    text = part.get("text") or part.get("content")
                    if text:
                        parts.append(str(text))
                        continue
                attr_content = getattr(part, "content", None)
                if attr_content:
                    parts.append(str(attr_content))
                    continue
                parts.append(str(part))
            return "".join(parts)

        if isinstance(response, dict):
            return json.dumps(response)

        if content is not None:
            return str(content)

        return str(response)
