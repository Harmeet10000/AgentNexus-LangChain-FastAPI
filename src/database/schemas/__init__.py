"""Database schemas package — deprecated shim.

Canonical locations:
- ChatMessage / ChatSession → app.features.chat.model
- DocumentVector → this module (legacy, superseded by UnifiedDocument)
Kept for one release so `from database.schemas import ...` keeps working.
"""

from app.features.chat.model import ChatMessage, ChatSession

from .document_vectors import DocumentVector

__all__: list[str] = [
    "ChatMessage",
    "ChatSession",
    "DocumentVector",
]
