import traceback
from typing import Any

from fastapi import Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import ORJSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.config.settings import get_settings
from app.utils.exceptions import APIException  # ← your base class
from app.utils.logger import logger


def _build_request_context(request: Request) -> dict[str, Any]:
    """Extract relevant request information (safe for logging & response)."""
    settings = get_settings()
    ctx = {
        "method": request.method,
        "url": str(request.url),
        "correlationId": getattr(request.state, "correlation_id", "unknown"),
    }
    # Only include client IP in non-production environments
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

    # Development & staging extras (never in production)
    settings = get_settings()
    if settings.ENVIRONMENT != "production":
        if trace:
            error_detail["trace"] = trace
        if inner_error:
            error_detail["innerError"] = inner_error

    return response


async def global_exception_handler(request: Request, exc: Exception) -> ORJSONResponse:
    """
    Unified global exception handler.

    Handles:
    - APIException family (custom business/validation/auth errors)
    - RequestValidationError (Pydantic / query / body validation)
    - StarletteHTTPException (plain HTTPException or raised by dependencies)
    - All other unexpected exceptions → 500
    """
    settings = get_settings()
    request_ctx = _build_request_context(request)
    correlation_id = request_ctx["correlationId"]

    # ────────────────────────────────────────────────
    # 1. Custom APIException family (business/validation/auth errors)
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
        )

        if status_code < 500:
            logger.warning(
                f"[{correlation_id}] {error_code} - {message}",
                status_code=status_code,
                method=request_ctx["method"],
                url=request_ctx["url"],
                error_code=error_code,
            )
        else:
            logger.error(
                f"[{correlation_id}] {error_code} - {message}",
                status_code=status_code,
                method=request_ctx["method"],
                url=request_ctx["url"],
                error_code=error_code,
            )

        return ORJSONResponse(status_code=status_code, content=response_body)

    # ────────────────────────────────────────────────
    # 2. Pydantic / FastAPI validation errors (422)
    # ────────────────────────────────────────────────
    if isinstance(exc, RequestValidationError):
        status_code = status.HTTP_422_UNPROCESSABLE_ENTITY
        error_code = "VALIDATION_ERROR"
        message = "Request validation failed"

        # Format errors in a client-friendly way
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
        )

        logger.warning(
            f"[{correlation_id}] VALIDATION_ERROR - {message}",
            status_code=status_code,
            method=request_ctx["method"],
            url=request_ctx["url"],
            validation_errors=validation_errors[:3],
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
        )

        if status_code < 500:
            logger.warning(
                f"[{correlation_id}] {error_code} - {message}",
                status_code=status_code,
                method=request_ctx["method"],
                url=request_ctx["url"],
            )
        else:
            logger.error(
                f"[{correlation_id}] {error_code} - {message}",
                status_code=status_code,
                method=request_ctx["method"],
                url=request_ctx["url"],
            )

        return ORJSONResponse(
            status_code=status_code, content=response_body, headers=exc.headers
        )

    # ────────────────────────────────────────────────
    # 4. Catch-all — unexpected server errors (500)
    # ────────────────────────────────────────────────
    status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
    error_code = "INTERNAL_SERVER_ERROR"
    message = "An unexpected error occurred"

    # In production: minimal info
    # In dev/staging: include traceback
    trace = traceback.format_exc() if settings.ENVIRONMENT != "production" else None

    response_body: dict[str, Any] = _build_error_response(
        status_code=status_code,
        message=message,
        error_code=error_code,
        request_ctx=request_ctx,
        trace=trace,
    )

    # Always log full exception in production (critical!)
    logger.error(
        f"[{correlation_id}] {error_code} - Unhandled exception",
        status_code=status_code,
        method=request_ctx["method"],
        url=request_ctx["url"],
        traceback=trace,
        exc_info=True,
    )

    return ORJSONResponse(status_code=status_code, content=response_body)
