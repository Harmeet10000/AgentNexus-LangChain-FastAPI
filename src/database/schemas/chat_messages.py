"""Deprecated shim — import from app.features.chat.model instead.

This file stays for one release. Migrate callers to
`from app.features.agent_saul.adapters.persistence.models import ChatMessage, ChatSession`. New code must use
that import directly. `from database import ChatMessage` also still works via
`database/__init__.py` shim for the same window.
"""

from app.features.agent_saul.adapters.persistence.models import ChatMessage, ChatSession

__all__ = ["ChatMessage", "ChatSession"]
