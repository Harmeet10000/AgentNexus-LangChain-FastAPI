"""HTTP renderer for typed Results — ADR-004."""

from __future__ import annotations

from enum import StrEnum
from typing import TYPE_CHECKING, Any

from fastapi import Response
from fastapi.utils import is_body_allowed_for_status_code
from pydantic import BaseModel, ConfigDict, Field
from returns.result import Failure

from app.config import get_settings
from app.utils.logger import request_state

from .errors import http_status_for_kind

if TYPE_CHECKING:
    from collections.abc import Mapping
    from typing import Self

    from returns.result import Result

    from .errors import FeatureError


class RequestMeta(BaseModel):
    """Request context echoed back in API responses."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        populate_by_name=True,
        serialize_by_alias=True,
    )

    ip: str | None = Field(default=None)
    method: str | None = Field(default=None)
    url: str | None = Field(default=None)
    correlation_id: str | None = Field(default=None, serialization_alias="correlationId")


class ErrorDetail(BaseModel):
    """Normalized error payload for non-success responses."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        populate_by_name=True,
        serialize_by_alias=True,
    )

    code: str
    message: str
    data: dict[str, object] | list[object] | str | None = Field(default=None)
    trace: str | None = Field(default=None)
    inner_error: str | None = Field(default=None, serialization_alias="innerError")
    flow: str | None = Field(default=None)


class APIResponse[T](BaseModel):
    """Default API response envelope for all HTTP handlers."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True, serialize_by_alias=True)

    success: bool = Field(default=True)
    status_code: int = Field(default=200, serialization_alias="statusCode")
    request: RequestMeta
    message: str = Field(default="Success")
    data: T | None = Field(default=None)
    error: ErrorDetail | None = Field(default=None)


class HealthStatus(StrEnum):
    """Health status of a dependency."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


class DependencyHealth(BaseModel):
    """Health status of a single dependency."""

    model_config = ConfigDict(extra="forbid", frozen=True, serialize_by_alias=True)

    name: str
    status: HealthStatus
    latency_ms: float = Field(default=0.0)
    message: str | None = None

    @classmethod
    def ok(cls, name: str, latency_ms: float = 0.0) -> Self:
        return cls(name=name, status=HealthStatus.HEALTHY, latency_ms=latency_ms)

    @classmethod
    def fail(cls, name: str, message: str, latency_ms: float = 0.0) -> Self:
        return cls(name=name, status=HealthStatus.UNHEALTHY, latency_ms=latency_ms, message=message)

    @classmethod
    def degraded(cls, name: str, message: str, latency_ms: float = 0.0) -> Self:
        return cls(name=name, status=HealthStatus.DEGRADED, latency_ms=latency_ms, message=message)


class HealthResponse(BaseModel):
    """Response body for GET /health."""

    model_config = ConfigDict(extra="forbid", frozen=True, serialize_by_alias=True)

    status: HealthStatus
    version: str = "1.0.0"
    git_sha: str = "unknown"
    build_date: str = "unknown"
    dependencies: list[DependencyHealth] = Field(default_factory=list)


APIResponse.model_rebuild(force=True)


def _serialize_data(data: Any) -> Any:
    """Normalize payloads for error metadata and mixed response payloads."""
    if isinstance(data, BaseModel):
        return data.model_dump(mode="json")
    if isinstance(data, list):
        return [_serialize_data(item) for item in data]
    return data


def _build_request_meta() -> RequestMeta:
    """Build request metadata from the current request context."""
    settings = get_settings()
    ctx = request_state.get({})

    ip = ctx.get("ip")
    if settings.ENVIRONMENT.lower() == "production":
        ip = None

    return RequestMeta(
        ip=ip,
        method=ctx.get("method"),
        url=ctx.get("url"),
        correlation_id=ctx.get("request_id"),
    )


def http_response[T](
    message: str,
    data: T | None = None,
    status_code: int = 200,
) -> APIResponse[T]:
    """Create standardized HTTP success response using ContextVar."""
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
) -> APIResponse[Any]:
    """Create standardized HTTP error response using ContextVar."""
    return APIResponse[Any](
        success=False,
        status_code=status_code,
        request=_build_request_meta(),
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
    kind = "infrastructure"
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
) -> APIResponse[T] | APIResponse[Any]:
    """Render a Result to the standard envelope, setting the transport status.

    On Success: sets response.status_code to success_status and returns success envelope.
    On Failure: derives status from error.kind (and retryable for INFRASTRUCTURE),
                sets response.status_code, and returns an error envelope. Optional
                failure metadata is used by the exception transport adapter.
    """
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
        )
    # Success path
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
) -> Response:
    """Render an exception failure through :func:`render_result` for HTTP transport."""
    response = Response(status_code=status_code, headers=headers)
    payload = render_result(
        Failure(_ExceptionError(message)),
        response,
        failure_status=status_code,
        failure_code=error_code,
        failure_data=data,
        trace=trace,
        flow=flow,
    )

    if not is_body_allowed_for_status_code(response.status_code):
        return response

    return Response(
        content=payload.model_dump_json(),
        status_code=response.status_code,
        headers=headers,
        media_type="application/json",
    )
