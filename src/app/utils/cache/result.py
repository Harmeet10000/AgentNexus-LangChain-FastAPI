"""Result adapters for Redis cache operations."""

from __future__ import annotations

from collections.abc import Awaitable, Callable  # noqa: TC003
from typing import TYPE_CHECKING

from returns.result import Failure, Success

from app.utils.exceptions import InfrastructureException
from app.utils.logger import logger

from .errors import CacheError

if TYPE_CHECKING:
    from .errors import CacheResult


async def run_cache_operation[T](
    operation: str,
    action: Callable[[], Awaitable[T]],
) -> CacheResult[T]:
    """Run one legacy cache operation and translate its transport failure to Result."""
    try:
        return Success(await action())
    except InfrastructureException as exc:
        detail = exc.detail if isinstance(exc.detail, str) else str(exc.detail)
        error = CacheError(
            message=f"Cache operation failed: {operation}",
            details={"operation": operation, "error": detail},
            source="redis_cache",
        )
        logger.bind(operation=operation, error=detail).warning("Cache operation returned failure")
        return Failure(error)


__all__ = ["run_cache_operation"]
