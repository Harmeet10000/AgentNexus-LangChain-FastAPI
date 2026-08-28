"""Chat feature — persistence lives here, not in database/schemas."""

from .model import ChatMessage, ChatSession

__all__ = ["ChatMessage", "ChatSession"]
