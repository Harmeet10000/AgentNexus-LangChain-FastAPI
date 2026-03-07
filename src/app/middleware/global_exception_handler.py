import traceback
from typing import Any

from fastapi import Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import ORJSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.config import get_settings
from app.utils import APIException, execution_path, logger


def _build_request_context(request: Request) -> dict[str, Any]:
    """Extract relevant request information."""
    settings = get_settings()
    ctx = {
        "method": request.method,
        "url": str(request.url),
        "correlationId": getattr(request.state, "correlation_id", "unknown"),
    }
    if settings.ENVIRONMENT != "production" and request.client:
        ctx["clientIp"] = request.client.host
    return ctx


def _build_error_response(
    status_code: int,
    message: str,
    error_code: str,
    data: Any | None = None,
    request_ctx: dict[str, Any] | None = None,
    trace: str | None = None,
    inner_error: str | None = None,
    flow: str | None = None,  # Added flow parameter
) -> dict[str, Any]:
    """Unified error shape — used for both known and unknown errors."""
    error_detail: dict[str, Any] = {
        "code": error_code,
        "message": message,
    }

    if data is not None:
        error_detail["data"] = data

    response: dict[str, Any] = {
        "success": False,
        "statusCode": status_code,
        "error": error_detail,
        "request": request_ctx or {},
    }

    settings = get_settings()
    if settings.ENVIRONMENT != "production":
        if trace:
            error_detail["trace"] = trace
        if inner_error:
            error_detail["innerError"] = inner_error
        if flow:
            error_detail["flow"] = flow  # Expose the execution path to the client in dev

    return response


async def global_exception_handler(request: Request, exc: Exception) -> ORJSONResponse:
    settings = get_settings()
    request_ctx = _build_request_context(request)

    # Extract the current function chain from our ContextVar
    current_flow = " -> ".join(execution_path.get())

    # ────────────────────────────────────────────────
    # 1. Custom APIException family
    # ────────────────────────────────────────────────
    if isinstance(exc, APIException):
        status_code = exc.status_code
        error_code = exc.error_code
        message = (
            exc.detail.get("message", str(exc.detail))
            if isinstance(exc.detail, dict)
            else str(exc.detail)
        )
        data = exc.detail.get("data") if isinstance(exc.detail, dict) else None

        response_body = _build_error_response(
            status_code=status_code,
            message=message,
            error_code=error_code,
            data=data,
            request_ctx=request_ctx,
            flow=current_flow,
        )

        # Loguru's context already has reqId, method, url, and layer.
        # We only bind what is specific to this exact error.
        log_call = logger.bind(error_code=error_code, status_code=status_code)
        if status_code < 500:
            log_call.warning(message)
        else:
            log_call.error(message)

        return ORJSONResponse(status_code=status_code, content=response_body)

    # ────────────────────────────────────────────────
    # 2. Pydantic / FastAPI validation errors (422)
    # ────────────────────────────────────────────────
    if isinstance(exc, RequestValidationError):
        status_code = status.HTTP_422_UNPROCESSABLE_ENTITY
        error_code = "VALIDATION_ERROR"
        message = "Request validation failed"

        validation_errors = [
            {
                "field": " → ".join(map(str, err["loc"])),
                "message": err["msg"],
                "type": err["type"],
            }
            for err in exc.errors()
        ]

        response_body = _build_error_response(
            status_code=status_code,
            message=message,
            error_code=error_code,
            data={"errors": validation_errors},
            request_ctx=request_ctx,
            flow=current_flow,
        )

        logger.bind(status_code=status_code, validation_errors=validation_errors[:3]).warning(
            message
        )

        return ORJSONResponse(status_code=status_code, content=response_body)

    # ────────────────────────────────────────────────
    # 3. Plain HTTPException / Starlette exceptions
    # ────────────────────────────────────────────────
    if isinstance(exc, StarletteHTTPException):
        status_code = exc.status_code
        error_code = f"HTTP_{status_code}"
        message = exc.detail if isinstance(exc.detail, str) else "HTTP error"

        response_body = _build_error_response(
            status_code=status_code,
            message=message,
            error_code=error_code,
            request_ctx=request_ctx,
            flow=current_flow,
        )

        log_call = logger.bind(status_code=status_code)
        if status_code < 500:
            log_call.warning(message)
        else:
            log_call.error(message)

        return ORJSONResponse(status_code=status_code, content=response_body, headers=exc.headers)

    # ────────────────────────────────────────────────
    # 4. Catch-all — unexpected server errors (500)
    # ────────────────────────────────────────────────
    status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
    error_code = "INTERNAL_SERVER_ERROR"
    message = "An unexpected error occurred"

    trace = traceback.format_exc() if settings.ENVIRONMENT != "production" else None

    response_body = _build_error_response(
        status_code=status_code,
        message=message,
        error_code=error_code,
        request_ctx=request_ctx,
        trace=trace,
        flow=current_flow,
    )

    # Loguru's .exception() automatically formats the traceback and exc_info
    # We bind the flow so it is explicitly indexed in the JSON file output
    # Inside the Catch-all (500) block:
    exc_type = type(exc).__name__
    last_function = execution_path.get()[-1] if execution_path.get() else "unknown_layer"

    dynamic_message = f"Unhandled {exc_type} crashed in {last_function}"

    logger.bind(
        status_code=status_code, error_code=error_code, crashed_at_flow=current_flow
    ).exception(dynamic_message)

    return ORJSONResponse(status_code=status_code, content=response_body)
