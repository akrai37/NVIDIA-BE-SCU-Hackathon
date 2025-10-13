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

_STRUCTURED_EXTRACTION_SYSTEM_PROMPT = """You are an expert document analyst specializing in extracting structured data from grant, finance, and compliance documents.

CRITICAL: You MUST return ONLY a valid JSON object. No markdown, no code fences, no explanatory text - just the raw JSON.

Your task is to extract:
1. A short summary (2-3 sentences) with 3-5 key bullet points
2. All dates in ISO-8601 format (YYYY-MM-DD) with event types (due, start, end, renewal, reporting)
3. All financial information (amounts with currency)
4. All quantities (percentages, counts, durations)
5. All contact information (names, roles, emails, phones)

For each piece of extracted data, include the page_number and chunk_id where it was found for highlighting purposes.

Requirements:
- Return ONLY valid JSON with double quotes for keys and string values
- Do NOT wrap the JSON in markdown code blocks (no ```json or ```)
- Do NOT include any text before or after the JSON
- If information is missing for a category, use an empty array []
- Ensure all dates are in ISO-8601 format (YYYY-MM-DD)
- Include page_number and chunk_id references wherever possible

Your entire response should be parseable by JSON.parse()."""

_STRUCTURED_EXTRACTION_USER_PROMPT = """Document context with chunk IDs and page numbers:
{context}

Document metadata:
- Title: {title}
- Total pages: {pages}

Extract structured data and provide a JSON object with the following exact shape:
{{
  "summary": {{
    "summary": "2-3 sentence summary of the document",
    "bullet_points": ["key point 1", "key point 2", "key point 3"]
  }},
  "dates": [
    {{
      "date": "YYYY-MM-DD",
      "event_type": "due|start|end|renewal|reporting",
      "description": "Context about this date",
      "page_number": 1,
      "chunk_id": "chunk_id_here"
    }}
  ],
  "financial": [
    {{
      "amount": 10000.00,
      "currency": "USD",
      "description": "Context about this amount",
      "page_number": 1,
      "chunk_id": "chunk_id_here"
    }}
  ],
  "quantities": [
    {{
      "value": 50,
      "unit": "%",
      "type": "percentage|count|duration",
      "description": "Context about this quantity",
      "page_number": 1,
      "chunk_id": "chunk_id_here"
    }}
  ],
  "contacts": [
    {{
      "name": "John Doe",
      "role": "Program Officer",
      "email": "john@example.com",
      "phone": "+1-555-0100",
      "page_number": 1,
      "chunk_id": "chunk_id_here"
    }}
  ]
}}"""


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
            max_tokens=4096,  # Increase token limit to avoid truncation
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
        self._extraction_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", _STRUCTURED_EXTRACTION_SYSTEM_PROMPT),
                ("user", _STRUCTURED_EXTRACTION_USER_PROMPT),
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

        content = self._coerce_content(response)

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
        """Extract content from various response formats."""
        if response is None:
            return "{}"

        if isinstance(response, str):
            return response

        # Try dict-like access first (for the new response format)
        if isinstance(response, dict):
            content = response.get("content")
            if content:
                return str(content)
            return json.dumps(response)

        # Try attribute access (for object-like responses)
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

        if content is not None:
            return str(content)

        return str(response)

    @staticmethod
    def _clean_json_response(content: str) -> str:
        """Clean JSON response by removing markdown code fences."""
        content = content.strip()

        # Remove markdown code fences if present
        if content.startswith("```json"):
            content = content[7:]  # Remove ```json
        elif content.startswith("```"):
            content = content[3:]  # Remove ```

        if content.endswith("```"):
            content = content[:-3]  # Remove closing ```

        return content.strip()

    async def extract_structured_data(
        self, *, context: str, title: str, pages: int
    ) -> Dict[str, Any]:
        """
        Extract structured data from document context.

        Args:
            context: Document context with chunk IDs and page numbers
            title: Document title
            pages: Number of pages

        Returns:
            Dict containing structured extraction (summary, dates, financial, quantities, contacts)
        """
        chain = self._extraction_prompt | self._client
        response = await run_in_threadpool(
            chain.invoke,
            {
                "context": context,
                "title": title,
                "pages": pages,
            },
        )

        # Extract content from response
        content = self._coerce_content(response)
        content = self._clean_json_response(content)

        try:
            return json.loads(content)
        except json.JSONDecodeError as exc:
            # Log the actual content for debugging
            print(
                f"Failed to parse structured extraction JSON. Content: {content[:500]}..."
            )
            raise HTTPException(
                status_code=502,
                detail="Model response could not be parsed as JSON for structured extraction.",
            ) from exc

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
            Dict containing answer, session info, and references (chunks with page numbers)
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

        # Get references from stored chunks in session
        references = []
        if session.document_chunks:
            # Infer tag from question context
            question_lower = question.lower()
            inferred_tag = self._infer_tag_from_question(question_lower)
            
            # Build all references with scores
            all_references = []
            for chunk_id, chunk_data in session.document_chunks.items():
                content = chunk_data.get("content", "")
                
                # Try to infer more specific tag from chunk content if general tag
                tag = inferred_tag or self._infer_tag_from_content(content)
                
                all_references.append({
                    "chunk_id": chunk_id,
                    "page_number": chunk_data.get("page_number"),
                    "content": content,
                    "relevance_score": chunk_data.get("score"),
                    "tag": tag,
                })
            
            # Sort by relevance score (highest first) and limit to top 3
            all_references.sort(
                key=lambda x: x.get("relevance_score") or 0.0, 
                reverse=True
            )
            references = all_references[:3]  # Keep only top 3 most relevant

        return {
            "answer": answer,
            "session_id": session_id,
            "conversation_length": len(session.messages),
            "references": references,
        }

    def create_session(
        self,
        document_id: Optional[str] = None,
        document_context: Optional[str] = None,
        document_chunks: Optional[Dict[str, Dict]] = None,
    ) -> str:
        """
        Create a new chat session.

        Args:
            document_id: Optional document identifier
            document_context: Optional cached document context
            document_chunks: Optional mapping of chunk_id to chunk data

        Returns:
            session_id: New session identifier
        """
        return self._session_manager.create_session(
            document_id=document_id,
            document_context=document_context,
            document_chunks=document_chunks,
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

    @staticmethod
    def _extract_references_from_context(context: str) -> List[Dict[str, Any]]:
        """
        Extract chunk references from document context.
        
        Context format expected: "Chunk ID: chunk-X | Page: Y\nContent...\n\n"
        
        Args:
            context: Document context string with embedded chunk metadata
            
        Returns:
            List of reference dictionaries with chunk_id, page_number, content, relevance_score
        """
        import re
        
        references = []
        
        # Split context into chunks (assuming they're separated by double newlines)
        chunks = context.split('\n\n')
        
        for chunk_text in chunks:
            if not chunk_text.strip():
                continue
                
            # Try to extract chunk ID and page number from headers like:
            # "Chunk ID: chunk-5 | Page: 3"
            # or "[chunk-5] (Page 3)"
            # or just look for chunk-X pattern
            chunk_id_match = re.search(r'chunk-\d+', chunk_text, re.IGNORECASE)
            page_match = re.search(r'Page[:\s]+(\d+)', chunk_text, re.IGNORECASE)
            
            if chunk_id_match:
                chunk_id = chunk_id_match.group(0)
                page_number = int(page_match.group(1)) if page_match else None
                
                # Extract the actual content (remove metadata lines)
                content_lines = []
                for line in chunk_text.split('\n'):
                    # Skip lines that look like metadata
                    if not re.match(r'^(Chunk ID|Page|Score|\[chunk-|\s*$)', line, re.IGNORECASE):
                        content_lines.append(line)
                
                content = '\n'.join(content_lines).strip()
                
                if content:
                    references.append({
                        "chunk_id": chunk_id,
                        "page_number": page_number,
                        "content": content,
                        "relevance_score": None,  # Could be extracted if available
                    })
        
        return references

    @staticmethod
    def _infer_tag_from_question(question: str) -> Optional[str]:
        """
        Infer a tag/category from the user's question.
        
        Args:
            question: User's question in lowercase
            
        Returns:
            Tag string or None
        """
        # Define keyword mappings for different tags
        tag_keywords = {
            "deadline": ["deadline", "due date", "when is", "submission date", "cut-off", "by when"],
            "eligibility": ["eligible", "eligibility", "qualify", "requirements", "criteria", "who can"],
            "financial": ["budget", "cost", "funding", "money", "amount", "payment", "fee", "grant amount"],
            "requirement": ["require", "must", "need to", "necessary", "mandatory", "obligation"],
            "contact": ["contact", "email", "phone", "reach", "call", "address", "who to contact"],
            "process": ["how to", "steps", "procedure", "process", "apply", "submit"],
            "documentation": ["document", "form", "paperwork", "file", "attachment", "submit"],
            "reporting": ["report", "reporting", "update", "progress", "quarterly", "annual"],
        }
        
        # Check which tag keywords are present in the question
        for tag, keywords in tag_keywords.items():
            if any(keyword in question for keyword in keywords):
                return tag
        
        return None

    @staticmethod
    def _infer_tag_from_content(content: str) -> Optional[str]:
        """
        Infer a tag/category from chunk content.
        
        Args:
            content: Chunk content text
            
        Returns:
            Tag string or None
        """
        content_lower = content.lower()
        
        # Define keyword patterns for content-based tagging
        content_patterns = {
            "deadline": ["deadline", "due date", "submission date", "by", "before", "cut-off date"],
            "eligibility": ["eligible", "eligibility", "qualify", "must be", "requirement", "criteria"],
            "financial": ["$", "budget", "cost", "funding", "grant amount", "fee", "payment", "price"],
            "requirement": ["required", "must", "shall", "mandatory", "necessary", "need to"],
            "contact": ["contact", "@", "phone:", "email:", "call", "reach us", "tel:"],
            "process": ["step", "procedure", "process", "how to", "application process"],
            "documentation": ["document", "form", "attach", "submit", "upload", "provide"],
            "reporting": ["report", "reporting", "quarterly", "annual", "update", "progress report"],
        }
        
        # Count matches for each tag
        tag_scores = {}
        for tag, patterns in content_patterns.items():
            score = sum(1 for pattern in patterns if pattern in content_lower)
            if score > 0:
                tag_scores[tag] = score
        
        # Return tag with highest score, or None if no matches
        if tag_scores:
            return max(tag_scores.items(), key=lambda x: x[1])[0]
        
        return "general"  # Default tag for unclassified content

