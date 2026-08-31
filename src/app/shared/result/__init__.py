"""Internal Result conventions for expected recoverable failures."""

from .errors import (
    STATUS_BY_KIND,
    AppError,
    ConflictAppError,
    ErrorKind,
    ExternalServiceAppError,
    FeatureError,
    InfrastructureAppError,
    NotFoundAppError,
    ValidationAppError,
    http_status_for_kind,
)
from .logging import log_expected_failure
from .mappers import app_error_to_exception
from .render import render_result
from .types import AppResult

__all__ = [
    "STATUS_BY_KIND",
    "AppError",
    "AppResult",
    "ConflictAppError",
    "ErrorKind",
    "ExternalServiceAppError",
    "FeatureError",
    "InfrastructureAppError",
    "NotFoundAppError",
    "ValidationAppError",
    "app_error_to_exception",
    "http_status_for_kind",
    "log_expected_failure",
    "render_result",
]
