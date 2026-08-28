"""Database package — kernel only. Feature tables live in app/features/*/model.py.

`ChatMessage` / `ChatSession` / `DocumentVector` are re-exported here for one
release so `from database import ChatMessage` keeps working after the move to
`app.features.chat.model`. New code must import from the feature directly.
"""

# Legacy re-exports — deprecated, remove after one release.
from app.features.chat.model import ChatMessage, ChatSession
from database.schemas.document_vectors import DocumentVector

from .base import Base, PublicIdMixin, TimestampMixin

__all__ = [
    "Base",
    "ChatMessage",
    "ChatSession",
    "DocumentVector",
    "PublicIdMixin",
    "TimestampMixin",
]
