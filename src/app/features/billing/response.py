"""Helpers for the service -> router response boundary.

Services return ``http_error(...)`` envelopes on expected failures (the
project's RESULT-PATTERN.md standard). Routers pass those error envelopes
through untouched instead of double-wrapping them with ``http_response``.
"""

from typing import Any, overload

from app.shared.result import (
    AppError,
    ConflictAppError,
    ExternalServiceAppError,
    InfrastructureAppError,
    NotFoundAppError,
    ValidationAppError,
    log_expected_failure,
)
from app.utils import APIResponse, http_error, http_response

# Service methods return either a fully-formed error envelope on expected
# failures, or the success value. Routers pass both through `bill_response`.
type ServiceResult[T] = APIResponse[Any] | T


def status_for(error: AppError) -> int:
    """Map an AppError kind to an HTTP status code."""
    if isinstance(error, ValidationAppError):
        return 422
    if isinstance(error, NotFoundAppError):
        return 404
    if isinstance(error, ConflictAppError):
        return 409
    if isinstance(error, ExternalServiceAppError):
        return 502
    if isinstance(error, InfrastructureAppError):
        return 503 if error.retryable else 500
    return 500


def failure_envelope(error: AppError, *, operation: str) -> APIResponse[Any]:
    """Convert a repository Failure into the standard http_error envelope."""
    log_expected_failure(error, operation=operation)
    return http_error(
        message=error.message,
        status_code=status_for(error),
        error_code=error.code,
        data=error.details,
    )


@overload
def bill_response[T](message: str, result: APIResponse[Any] | T) -> APIResponse[T]: ...


@overload
def bill_response(message: str, result: object) -> APIResponse[Any]: ...


def bill_response(message: str, result: APIResponse[Any] | object) -> APIResponse[Any]:
    """Pass through a service error envelope, otherwise wrap success data."""
    if isinstance(result, APIResponse):
        return result
    return http_response(message=message, data=result)
