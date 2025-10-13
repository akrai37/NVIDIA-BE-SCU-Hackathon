from __future__ import annotations

import json
from typing import Any, Dict, List, Optional
from urllib import response

from fastapi import HTTPException
from fastapi.concurrency import run_in_threadpool
from langchain.prompts import ChatPromptTemplate
from langchain_nvidia_ai_endpoints import ChatNVIDIA

from app.core.config import get_settings
from app.services.chat_session import ChatMessage, get_session_manager


_SYSTEM_PROMPT = """You are an expert document analyst helping small nonprofits understand complex grant, finance, and compliance documents. 

CRITICAL: You MUST return ONLY a valid JSON object. No markdown, no code fences, no explanatory text - just the raw JSON.

Requirements:
- Return ONLY valid JSON with double quotes for keys and string values
- Do NOT wrap the JSON in markdown code blocks (no ```json or ```)
- Do NOT include any text before or after the JSON
- Organize findings into critical, important, and informational categories
- Highlight deadlines, eligibility, financial requirements, and follow-up actions
- Include references back to the provided chunk IDs where possible
- If information is missing, use an empty array [] for that section

Your entire response should be parseable by JSON.parse()."""

_USER_PROMPT = """Document context:
{context}

Document metadata:
- Title: {title}
- Total pages: {pages}

Provide a JSON object with the following shape:
{{
  "summary": string,
  "key_highlights": string[],
  "categorized_insights": {{
    "critical": InsightItem[],
    "important": InsightItem[],
    "informational": InsightItem[]
  }},
  "extracted_data": ExtractedDataPoint[],
  "recommended_next_steps": RecommendedStep[],
  "references": SourceReference[]
}}

Where InsightItem = {{"label": string, "description": string, "source_chunk_id": string | null}}
ExtractedDataPoint = {{"name": string, "value": string, "source_chunk_id": string | null}}
RecommendedStep = {{"action": string, "priority": "critical" | "important" | "informational", "rationale": string | null, "due_date": string | null, "owner": string | null, "source_chunk_id": string | null}}
SourceReference = {{"chunk_id": string, "page_number": number | null, "score": number, "preview": string}}
"""

_CHAT_SYSTEM_PROMPT = """You are an expert document analyst helping nonprofits understand complex grant, finance, and compliance documents. You have access to the document content and can answer follow-up questions while maintaining conversation context.

CRITICAL: 
- Answer questions based on the provided document context
- When additional context is provided, pay special attention to it as it's what the user is specifically focused on
- Reference previous parts of the conversation when relevant
- Be concise but thorough
- Cite specific sections or page numbers when possible
- If you don't know something, say so clearly

Your responses should be helpful, accurate, and based on the document content."""

_CHAT_USER_PROMPT = """Document context:
{context}

{additional_context_section}

Previous conversation:
{chat_history}

User question: {question}

Please provide a helpful answer based on the document context{additional_context_instruction} and conversation history."""


class GuidanceService:
    """Generate structured guidance using NVIDIA Nemotron models via NIM."""

    def __init__(self) -> None:
        settings = get_settings()
        if not settings.nvidia_api_key:
            raise RuntimeError(
                "NVIDIA_API_KEY is not configured for guidance generation."
            )
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
        self._chat_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", _CHAT_SYSTEM_PROMPT),
                ("user", _CHAT_USER_PROMPT),
            ]
        )
        self._session_manager = get_session_manager()

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

        content = self._coerce_content(response.model_dump_json())

        try:
            return json.loads(content)
        except (
            json.JSONDecodeError
        ) as exc:  # pragma: no cover - depends on model output
            # Log the actual content for debugging
            print(f"Failed to parse JSON. Content: {content[:500]}...")
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

    async def chat(
        self,
        *,
        question: str,
        session_id: str,
        context: Optional[str] = None,
        additional_context: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Handle chat-based follow-up questions with session history.

        Args:
            question: User's question
            session_id: Chat session ID
            context: Optional document context (will use cached if not provided)
            additional_context: Optional additional context from user (e.g., specific lines from document)

        Returns:
            Dict containing answer and session info
        """
        # Get or validate session
        session = self._session_manager.get_session(session_id)
        if not session:
            raise HTTPException(
                status_code=404,
                detail=f"Session {session_id} not found or expired",
            )

        # Use provided context or cached context
        doc_context = context or session.document_context or ""
        if not doc_context:
            raise HTTPException(
                status_code=400,
                detail="No document context available for this session",
            )

        # Get conversation history
        history = self._session_manager.get_conversation_history(
            session_id, max_messages=10
        )
        chat_history = self._format_chat_history(history)

        # Add user message to history
        self._session_manager.add_message(session_id, "user", question)

        # Prepare additional context section
        additional_context_section = ""
        additional_context_instruction = ""
        if additional_context:
            additional_context_section = f"""User-provided additional context (pay special attention to this):
{additional_context}
"""
            additional_context_instruction = (
                ", especially the additional context provided"
            )

        # Generate response
        chain = self._chat_prompt | self._client
        response = await run_in_threadpool(
            chain.invoke,
            {
                "context": doc_context,
                "additional_context_section": additional_context_section,
                "additional_context_instruction": additional_context_instruction,
                "chat_history": chat_history,
                "question": question,
            },
        )

        # Extract answer
        answer = self._extract_answer(response)

        # Add assistant response to history
        self._session_manager.add_message(session_id, "assistant", answer)

        return {
            "answer": answer,
            "session_id": session_id,
            "conversation_length": len(session.messages),
        }

    def create_session(
        self,
        document_id: Optional[str] = None,
        document_context: Optional[str] = None,
    ) -> str:
        """
        Create a new chat session.

        Args:
            document_id: Optional document identifier
            document_context: Optional cached document context

        Returns:
            session_id: New session identifier
        """
        return self._session_manager.create_session(
            document_id=document_id,
            document_context=document_context,
        )

    def clear_session(self, session_id: str) -> bool:
        """
        Clear a chat session.

        Args:
            session_id: Session to clear

        Returns:
            True if session was found and cleared
        """
        return self._session_manager.clear_session(session_id)

    @staticmethod
    def _format_chat_history(messages: List[ChatMessage]) -> str:
        """Format chat history for the prompt."""
        if not messages:
            return "No previous conversation."

        history_parts = []
        for msg in messages:
            role = "User" if msg.role == "user" else "Assistant"
            # Truncate long messages
            content = msg.content
            if len(content) > 500:
                content = content[:500] + "..."
            history_parts.append(f"{role}: {content}")

        return "\n".join(history_parts)

    @staticmethod
    def _extract_answer(response: Any) -> str:
        """Extract answer text from LLM response."""
        if response is None:
            return "I apologize, but I couldn't generate a response."

        if isinstance(response, str):
            return response

        # Try to get content attribute
        content = getattr(response, "content", None)
        if isinstance(content, str):
            return content

        # Try dict-like access
        if isinstance(response, dict):
            return response.get("content", response.get("text", str(response)))

        # Last resort
        return str(response)
