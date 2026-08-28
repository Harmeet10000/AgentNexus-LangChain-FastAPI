"""Deprecated shim — import from app.features.chat.model instead.

This file stays for one release so `from database.schemas.chat_messages import ...`
and `from database import ChatMessage` (old) keep working. New code must use
`from app.features.chat.model import ChatMessage, ChatSession`.
"""

from app.features.chat.model import ChatMessage, ChatSession

__all__ = ["ChatMessage", "ChatSession"]
