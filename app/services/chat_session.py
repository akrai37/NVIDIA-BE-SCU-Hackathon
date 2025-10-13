"""Chat session management for conversation history and context retention."""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional

from pydantic import BaseModel, Field

# Import settings to get configurable timeout
from app.core.config import get_settings


class ChatMessage(BaseModel):
    """Represents a single message in the conversation."""

    role: str = Field(..., description="Message role: 'user' or 'assistant'")
    content: str = Field(..., description="Message content")
    timestamp: datetime = Field(default_factory=datetime.now)


class ChatSession(BaseModel):
    """Represents a chat session with conversation history."""

    session_id: str = Field(..., description="Unique session identifier")
    document_id: Optional[str] = Field(None, description="Associated document ID")
    messages: List[ChatMessage] = Field(default_factory=list)
    document_context: Optional[str] = Field(None, description="Cached document context")
    created_at: datetime = Field(default_factory=datetime.now)
    last_activity: datetime = Field(default_factory=datetime.now)
    metadata: Dict = Field(
        default_factory=dict, description="Additional session metadata"
    )


class ChatSessionManager:
    """Manages chat sessions with automatic cleanup and history tracking."""

    def __init__(self, session_timeout_minutes: int = 30):
        """
        Initialize the session manager.

        Args:
            session_timeout_minutes: Minutes of inactivity before session expires
        """
        self._sessions: Dict[str, ChatSession] = {}
        self._timeout = timedelta(minutes=session_timeout_minutes)

    def create_session(
        self,
        document_id: Optional[str] = None,
        document_context: Optional[str] = None,
        metadata: Optional[Dict] = None,
    ) -> str:
        """
        Create a new chat session.

        Args:
            document_id: Optional document ID to associate with session
            document_context: Optional cached document context
            metadata: Optional metadata to store with session

        Returns:
            session_id: Unique session identifier
        """
        session_id = str(uuid.uuid4())
        self._sessions[session_id] = ChatSession(
            session_id=session_id,
            document_id=document_id,
            document_context=document_context,
            metadata=metadata or {},
        )
        return session_id

    def get_session(self, session_id: str) -> Optional[ChatSession]:
        """
        Get a session by ID and update its last activity time.

        Args:
            session_id: Session identifier

        Returns:
            ChatSession if found and not expired, None otherwise
        """
        self._cleanup_expired_sessions()

        session = self._sessions.get(session_id)
        if session:
            session.last_activity = datetime.now()
        return session

    def add_message(
        self,
        session_id: str,
        role: str,
        content: str,
    ) -> bool:
        """
        Add a message to the session history.

        Args:
            session_id: Session identifier
            role: Message role ('user' or 'assistant')
            content: Message content

        Returns:
            True if message added successfully, False if session not found
        """
        session = self.get_session(session_id)
        if not session:
            return False

        session.messages.append(
            ChatMessage(
                role=role,
                content=content,
            )
        )
        return True

    def get_conversation_history(
        self,
        session_id: str,
        max_messages: Optional[int] = None,
    ) -> List[ChatMessage]:
        """
        Get conversation history for a session.

        Args:
            session_id: Session identifier
            max_messages: Maximum number of recent messages to return

        Returns:
            List of ChatMessage objects (most recent last)
        """
        session = self.get_session(session_id)
        if not session:
            return []

        messages = session.messages
        if max_messages is not None:
            messages = messages[-max_messages:]

        return messages

    def update_document_context(
        self,
        session_id: str,
        document_context: str,
    ) -> bool:
        """
        Update the cached document context for a session.

        Args:
            session_id: Session identifier
            document_context: New document context

        Returns:
            True if updated successfully, False if session not found
        """
        session = self.get_session(session_id)
        if not session:
            return False

        session.document_context = document_context
        return True

    def clear_session(self, session_id: str) -> bool:
        """
        Remove a session and its history.

        Args:
            session_id: Session identifier

        Returns:
            True if session was found and removed, False otherwise
        """
        if session_id in self._sessions:
            del self._sessions[session_id]
            return True
        return False

    def list_active_sessions(self) -> List[str]:
        """
        Get list of all active session IDs.

        Returns:
            List of session IDs
        """
        self._cleanup_expired_sessions()
        return list(self._sessions.keys())

    def get_session_count(self) -> int:
        """
        Get count of active sessions.

        Returns:
            Number of active sessions
        """
        self._cleanup_expired_sessions()
        return len(self._sessions)

    def _cleanup_expired_sessions(self):
        """Remove sessions that have exceeded the timeout period."""
        now = datetime.now()
        expired = [
            sid
            for sid, session in self._sessions.items()
            if now - session.last_activity > self._timeout
        ]
        for sid in expired:
            del self._sessions[sid]


# Global session manager instance
_session_manager: Optional[ChatSessionManager] = None


def get_session_manager() -> ChatSessionManager:
    """Get the global session manager instance with configured timeout."""
    global _session_manager
    if _session_manager is None:
        settings = get_settings()
        _session_manager = ChatSessionManager(
            session_timeout_minutes=settings.session_timeout_minutes
        )
    return _session_manager
