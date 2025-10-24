"""Logging decorators for functions and methods."""

import functools
from collections.abc import Callable
from typing import Any

from loguru import logger


def log_execution(func: Callable) -> Callable:
    """Decorator to log function execution with error handling."""

    @functools.wraps(func)
    @logger.catch(message=f"Error in {func.__name__}", reraise=True)
    async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
        logger.debug(f"Executing {func.__name__}")
        result = await func(*args, **kwargs)
        logger.success(f"Completed {func.__name__}")
        return result

    @functools.wraps(func)
    @logger.catch(message=f"Error in {func.__name__}", reraise=True)
    def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
        logger.debug(f"Executing {func.__name__}")
        result = func(*args, **kwargs)
        logger.success(f"Completed {func.__name__}")
        return result

    return async_wrapper if functools.iscoroutinefunction(func) else sync_wrapper


def log_performance(threshold_ms: float = 1000) -> Callable:
    """Decorator to log slow function execution."""

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            import time

            start = time.time()
            result = await func(*args, **kwargs)
            duration = (time.time() - start) * 1000
            if duration > threshold_ms:
                logger.warning(
                    f"{func.__name__} took {duration:.2f}ms (threshold: {threshold_ms}ms)"
                )
            return result

        @functools.wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            import time

            start = time.time()
            result = func(*args, **kwargs)
            duration = (time.time() - start) * 1000
            if duration > threshold_ms:
                logger.warning(
                    f"{func.__name__} took {duration:.2f}ms (threshold: {threshold_ms}ms)"
                )
            return result

        return async_wrapper if functools.iscoroutinefunction(func) else sync_wrapper

    return decorator
