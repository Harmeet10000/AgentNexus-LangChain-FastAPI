"""HTTP renderer for typed Results — ADR-004."""

from __future__ import annotations

from typing import TYPE_CHECKING

from fastapi import Response
from returns.result import Failure

from app.utils.http_response import http_error, http_response

from .errors import http_status_for_kind

if TYPE_CHECKING:
    from typing import Any

    from returns.result import Result

    from app.utils.response_type import APIResponse

    from .errors import FeatureError


def render_result[T](
    result: Result[T, FeatureError],
    response: Response,
    message: str = "Success",
    success_status: int = 200,
) -> APIResponse[T] | APIResponse[Any]:
    """Render a Result to the standard envelope, setting the transport status.

    On Success: sets response.status_code to success_status and returns success envelope.
    On Failure: derives status from error.kind (and retryable for INFRASTRUCTURE),
                sets response.status_code, and returns error envelope with the
                error's class-constant code. Callers cannot override failure status.
    """
    if isinstance(result, Failure):
        error = result.failure()
        status = http_status_for_kind(error.kind, retryable=error.retryable)
        response.status_code = status
        return http_error(
            message=error.message,
            status_code=status,
            data=error.details,
            error_code=error.code.value,
        )
    # Success path
    response.status_code = success_status
    return http_response(message=message, data=result.unwrap(), status_code=success_status)
