"""Structured logging helpers for expected internal failures."""

from app.utils import execution_path, logger

from .errors import FeatureError


def log_expected_failure(error: FeatureError, *, operation: str | None = None) -> None:
    """Log an expected failure once at the ownership boundary."""
    # The `[]` default is load-bearing, for the same reason it is at
    # `middleware/global_exception_handler.py:54`: `execution_path` is set by the HTTP middleware
    # and is unset everywhere else — Celery tasks, LangGraph nodes, CLI entry points, tests. A
    # bare `.get()` raises `LookupError` there, which replaces the failure the caller was trying
    # to report with an unrelated one and loses the error entirely. This function is 46 callers
    # deep in code that is not all request-scoped.
    flow: str = " -> ".join(execution_path.get([]))
    logger.bind(
        error_code=error.code,
        retryable=error.retryable,
        source=error.source,
        operation=operation,
        flow=flow,
        details=error.details,
    ).warning(error.message)
