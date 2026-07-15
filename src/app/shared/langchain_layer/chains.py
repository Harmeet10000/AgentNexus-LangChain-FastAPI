"""Guardrail chain utilities."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Any

    from langchain_core.language_models import BaseChatModel
    from langchain_core.runnables import Runnable


def build_guardrail_chain(
    _model: BaseChatModel | None = None,
) -> Runnable[dict[str, Any], dict[str, Any]]:
    """Build a guardrail chain for content safety.

    This is a placeholder — replace with actual implementation when available.
    """
    # Import here to avoid circular imports
    from langchain_core.runnables import RunnablePassthrough

    return RunnablePassthrough()
