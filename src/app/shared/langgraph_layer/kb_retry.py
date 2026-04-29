"""Small retry helpers for KB LangGraph nodes."""

from __future__ import annotations

from typing import TYPE_CHECKING

from tenacity import AsyncRetrying, retry_if_exception_type, stop_after_attempt, wait_none

from app.utils import logger

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable


class TransientExternalError(Exception):
    """Raised when an external dependency failure is safe to retry immediately."""


async def retry_immediate[T](
    operation: Callable[[], Awaitable[T]],
    *,
    label: str,
    attempts: int = 3,
) -> T:
    """Run an async operation through Tenacity with immediate three-attempt retry."""
    retryer = AsyncRetrying(
        stop=stop_after_attempt(attempts),
        wait=wait_none(),
        retry=retry_if_exception_type(Exception),
        reraise=True,
        before_sleep=lambda state: logger.bind(
            label=label,
            attempt=state.attempt_number,
            attempts=attempts,
        ).warning("kb_retry_immediate_retry", error=str(state.outcome.exception())),
    )
    try:
        async for attempt in retryer:
            with attempt:
                return await operation()
    except Exception as exc:
        msg = f"{label} failed after {attempts} immediate attempts"
        raise TransientExternalError(msg) from exc
    msg = f"{label} failed without an exception"
    raise TransientExternalError(msg)
