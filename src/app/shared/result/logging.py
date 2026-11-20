"""Expected-failure logging at application ownership boundaries."""

from app.kernel.errors import FeatureError
from app.observability.logging import execution_path, logger


def log_expected_failure(error: FeatureError, *, operation: str | None = None) -> None:
    """Log an expected failure once without assuming request scope."""
    flow = " -> ".join(execution_path.get([]))
    logger.bind(
        error_code=error.code,
        retryable=error.retryable,
        source=error.source,
        operation=operation,
        flow=flow,
        details=error.details,
    ).warning(error.message)
