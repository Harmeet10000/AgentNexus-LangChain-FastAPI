"""Internal Result conventions for expected recoverable failures."""

from .errors import (
    AppError,
    ConflictAppError,
    ExternalServiceAppError,
    InfrastructureAppError,
    NotFoundAppError,
    ValidationAppError,
)
from .logging import log_expected_failure
from .mappers import app_error_to_exception
from .types import AppFutureResult, AppResult

__all__ = [
    "AppError",
    "AppFutureResult",
    "AppResult",
    "ConflictAppError",
    "ExternalServiceAppError",
    "InfrastructureAppError",
    "NotFoundAppError",
    "ValidationAppError",
    "app_error_to_exception",
    "log_expected_failure",
]
