"""Structured logging helpers for expected internal failures."""

from app.utils.logger import execution_path, logger

from .errors import AppError


def log_expected_failure(error: AppError, *, operation: str | None = None) -> None:
    """Log an expected failure once at the ownership boundary."""
    flow = " -> ".join(execution_path.get())
    logger.bind(
        error_code=error.code,
        retryable=error.retryable,
        source=error.source,
        operation=operation,
        flow=flow,
        details=error.details,
    ).warning(error.message)
