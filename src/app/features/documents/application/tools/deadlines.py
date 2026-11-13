"""Explicit Graphiti deadlines for the legal knowledge tools (harden 5.2).

Every knowledge-tool Graphiti read runs under
``DOCUMENTS_RETRIEVAL_GRAPHITI_TIMEOUT_SECONDS``. A deadline miss is a
structured unavailability (the tool answers ``ToolResult.unavailable_result``
or records the layer as unavailable) — never a hung agent step and never an
indistinguishable empty result. Transport failures already degrade inside the
BoundGraphitiService; this helper adds the missing wall-clock bound.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

from app.config import get_settings

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

GRAPHTI_DEADLINE_MISSED = object()


async def with_graphiti_deadline(
    _tool: str,
    call: Callable[[], Awaitable[Any]],
) -> tuple[Any, bool]:
    """Run one Graphiti read under the explicit deadline.

    Returns ``(result, deadline_missed)``: on a miss the result is the
    ``GRAPHTI_DEADLINE_MISSED`` sentinel and the caller answers with a
    structured unavailability. Any other transport failure propagates to the
    BoundGraphitiService degradation contract it already owns.
    """
    try:
        async with asyncio.timeout(get_settings().DOCUMENTS_RETRIEVAL_GRAPHITI_TIMEOUT_SECONDS):
            return await call(), False
    except TimeoutError:
        return GRAPHTI_DEADLINE_MISSED, True
    except Exception:
        raise
