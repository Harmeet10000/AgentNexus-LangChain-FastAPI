"""Producer-side trace propagation before broker publication."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Any

from opentelemetry import propagate

from app.observability.logging import logger


def _inject_trace_context(carrier: dict[str, Any], *, context: Any = None) -> None:
    """Inject the active W3C trace context into a Celery message carrier."""
    propagate.inject(carrier, context=context)


def log_task_published(
    sender: str | None = None,
    headers: dict[str, Any] | None = None,
    exchange: str | None = None,
    routing_key: str | None = None,
    **_: Any,
) -> None:
    if headers is not None:
        _inject_trace_context(headers)
    logger.bind(
        task=sender,
        task_id=(headers or {}).get("id"),
        exchange=exchange,
        routing_key=routing_key,
    ).info("Celery task published")
