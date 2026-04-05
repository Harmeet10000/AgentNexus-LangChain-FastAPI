from typing import Any, TypeVar

from pydantic import BaseModel

from app.config import Environment, get_settings
from app.shared.response_type import APIResponse, ErrorDetail, RequestMeta
from app.utils.logger import request_state

T = TypeVar("T")


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
    ctx = request_state.get()

    ip = ctx.get("ip")
    if settings.ENVIRONMENT == Environment.PRODUCTION:
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
