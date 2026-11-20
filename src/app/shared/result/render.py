"""HTTP envelope models and Result-to-response rendering."""

from __future__ import annotations

from enum import StrEnum
from typing import TYPE_CHECKING, Any

from fastapi import Response
from fastapi.utils import is_body_allowed_for_status_code
from pydantic import BaseModel, ConfigDict, Field
from returns.result import Failure

from app.config import get_settings
from app.kernel.errors import ErrorKind
from app.observability.logging import request_state

if TYPE_CHECKING:
    from collections.abc import Mapping

    from returns.result import Result

    from app.kernel.errors import FeatureError


STATUS_BY_KIND: dict[ErrorKind, int] = {
    ErrorKind.VALIDATION: 422,
    ErrorKind.NOT_FOUND: 404,
    ErrorKind.CONFLICT: 409,
    ErrorKind.AUTHENTICATION: 401,
    ErrorKind.AUTHORIZATION: 403,
    ErrorKind.EXTERNAL_SERVICE: 502,
    ErrorKind.INFRASTRUCTURE: 500,
}


def http_status_for_kind(kind: ErrorKind, *, retryable: bool = False) -> int:
    """Map a domain failure classification to its HTTP status."""
    if kind is ErrorKind.INFRASTRUCTURE:
        return 503 if retryable else 500
    return STATUS_BY_KIND[kind]


class RequestMeta(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        populate_by_name=True,
        serialize_by_alias=True,
    )

    ip: str | None = None
    method: str | None = None
    url: str | None = None
    correlation_id: str | None = Field(default=None, serialization_alias="correlationId")


class ErrorDetail(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        populate_by_name=True,
        serialize_by_alias=True,
    )

    code: str
    message: str
    data: Any = None
    trace: str | None = None
    inner_error: str | None = Field(default=None, serialization_alias="innerError")
    flow: str | None = None


class APIResponse[T](BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True, serialize_by_alias=True)

    success: bool = True
    status_code: int = Field(default=200, serialization_alias="statusCode")
    request: RequestMeta
    message: str = "Success"
    data: T | None = None
    error: ErrorDetail | None = None


APIResponse.model_rebuild(force=True)


def _serialize_data(data: Any) -> Any:
    if isinstance(data, BaseModel):
        return data.model_dump(mode="json")
    if isinstance(data, list):
        return [_serialize_data(item) for item in data]
    return data


def _build_request_meta(correlation_id: str | None = None) -> RequestMeta:
    settings = get_settings()
    context = request_state.get({})
    ip = context.get("ip")
    if settings.ENVIRONMENT.lower() == "production":
        ip = None
    return RequestMeta(
        ip=ip,
        method=context.get("method"),
        url=context.get("url"),
        correlation_id=(
            correlation_id if correlation_id is not None else context.get("request_id")
        ),
    )


def http_response[T](
    message: str,
    data: T | None = None,
    status_code: int = 200,
) -> APIResponse[T]:
    return APIResponse[T](
        success=True,
        status_code=status_code,
        request=_build_request_meta(),
        message=message,
        data=data,
        error=None,
    )


def http_error(
    message: str,
    status_code: int = 400,
    data: Any = None,
    *,
    error_code: str = "ERROR",
    trace: str | None = None,
    inner_error: str | None = None,
    flow: str | None = None,
    correlation_id: str | None = None,
) -> APIResponse[Any]:
    return APIResponse[Any](
        success=False,
        status_code=status_code,
        request=_build_request_meta(correlation_id=correlation_id),
        message=message,
        data=None,
        error=ErrorDetail(
            code=error_code,
            message=message,
            data=_serialize_data(data),
            trace=trace,
            inner_error=inner_error,
            flow=flow,
        ),
    )


class _ExceptionError:
    kind = ErrorKind.INFRASTRUCTURE
    code = "INTERNAL_SERVER_ERROR"
    retryable = False

    def __init__(self, message: str) -> None:
        self.message = message
        self.details = None


def _error_code_value(code: StrEnum | str) -> str:
    return code.value if isinstance(code, StrEnum) else code


def render_result[T](
    result: Result[T, FeatureError],
    response: Response,
    message: str = "Success",
    success_status: int = 200,
    *,
    failure_status: int | None = None,
    failure_code: StrEnum | str | None = None,
    failure_data: Any = None,
    trace: str | None = None,
    flow: str | None = None,
    correlation_id: str | None = None,
) -> APIResponse[T] | APIResponse[Any]:
    if isinstance(result, Failure):
        error = result.failure()
        status = failure_status or http_status_for_kind(error.kind, retryable=error.retryable)
        response.status_code = status
        return http_error(
            message=error.message,
            status_code=status,
            data=failure_data if failure_data is not None else error.details,
            error_code=_error_code_value(failure_code or error.code),
            trace=trace,
            flow=flow,
            correlation_id=correlation_id,
        )
    response.status_code = success_status
    return http_response(message=message, data=result.unwrap(), status_code=success_status)


def render_exception(
    *,
    message: str,
    status_code: int,
    error_code: StrEnum | str,
    data: Any = None,
    headers: Mapping[str, str] | None = None,
    trace: str | None = None,
    flow: str | None = None,
    correlation_id: str | None = None,
) -> Response:
    response = Response(status_code=status_code, headers=headers)
    payload = render_result(
        Failure(_ExceptionError(message)),
        response,
        failure_status=status_code,
        failure_code=error_code,
        failure_data=data,
        trace=trace,
        flow=flow,
        correlation_id=correlation_id,
    )
    if not is_body_allowed_for_status_code(response.status_code):
        return response
    return Response(
        content=payload.model_dump_json(),
        status_code=response.status_code,
        headers=headers,
        media_type="application/json",
    )
