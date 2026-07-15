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
from .patterns import (
    bind_success,
    map_success,
    match_app_result,
    match_result,
    match_result_or_raise,
    try_unwrap_failure,
    try_unwrap_success,
    unwrap_app_failure,
    unwrap_app_result_or_raise,
    unwrap_app_success,
    unwrap_failure,
    unwrap_success,
)
from .types import AppResult

__all__ = [
    "AppError",
    "AppResult",
    "ConflictAppError",
    "ExternalServiceAppError",
    "InfrastructureAppError",
    "NotFoundAppError",
    "ValidationAppError",
    "app_error_to_exception",
    "bind_success",
    "log_expected_failure",
    "map_success",
    "match_app_result",
    "match_result",
    "match_result_or_raise",
    "try_unwrap_failure",
    "try_unwrap_success",
    "unwrap_app_failure",
    "unwrap_app_result_or_raise",
    "unwrap_app_success",
    "unwrap_failure",
    # Pattern matching helpers
    "unwrap_success",
]
