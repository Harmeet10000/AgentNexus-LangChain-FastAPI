"""Database package — kernel only. Feature tables live in app/features/*/model.py.

`ChatMessage` / `ChatSession` / `DocumentVector` are re-exported here for one
release so `from database import ChatMessage` keeps working after the move to
`app.features.chat.model`. New code must import from the feature directly.
"""

from .base import Base, PublicIdMixin, TimestampMixin

__all__ = [
    "Base",
    "ChatMessage",
    "ChatSession",
    "DocumentVector",
    "PublicIdMixin",
    "TimestampMixin",
]


def __getattr__(name: str) -> type:
    """Lazy shim — avoids import cycle when app.features.chat.model imports Base.

    `from app.features.chat.model import ...` triggers `database` package load.
    If __init__.py eagerly imported ChatMessage, it would import chat.model
    before its classes are defined. Lazy getattr breaks the cycle.
    """
    if name in {"ChatMessage", "ChatSession"}:
        from app.features.chat.model import ChatMessage as _CM  # noqa: N814
        from app.features.chat.model import ChatSession as _CS  # noqa: N814

        return _CM if name == "ChatMessage" else _CS
    if name == "DocumentVector":
        from database.schemas.document_vectors import DocumentVector as _DV  # noqa: N814

        return _DV
    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)
