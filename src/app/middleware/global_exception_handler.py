import traceback

from fastapi import Request, status
from fastapi.exceptions import HTTPException, RequestValidationError
from fastapi.responses import ORJSONResponse

from app.config.settings import get_settings
from app.utils.exceptions import APIException
from app.utils.logger import logger


async def global_exception_handler(request: Request, exc: Exception) -> ORJSONResponse:
    """
    Unified exception handler for all errors.
    Handles: APIException, RequestValidationError, HTTPException, and unexpected errors.
    """
    settings = get_settings()
    correlation_id = getattr(request.state, "correlation_id", "unknown")
    is_production = settings.ENVIRONMENT == "production"

    # Build base request info (reused across all error types)
    request_info = {
        "method": request.method,
        "url": str(request.url),
        "correlationId": correlation_id,
    }

    # Include IP only in non-production
    if not is_production and request.client:
        request_info["ip"] = request.client.host

    # Determine error type and build error response
    if isinstance(exc, APIException):
        # Custom API exceptions
        error_obj = {
            "name": exc.name,
            "success": False,
            "statusCode": exc.status_code,
            "request": request_info,
            "message": exc.message,
            "data": exc.data,
        }

        # Add trace for 5xx errors in non-production
        if exc.status_code >= 500 and not is_production:
            error_obj["trace"] = traceback.format_exc()

        log_level = "error" if exc.status_code >= 500 else "warning"

    elif isinstance(exc, RequestValidationError):
        # Pydantic validation errors (422)
        errors = [
            {
                "field": ".".join(str(x) for x in error["loc"][1:]),
                "message": error["msg"],
                "type": error["type"],
            }
            for error in exc.errors()
        ]

        error_obj = {
            "name": "ValidationError",
            "success": False,
            "statusCode": status.HTTP_422_UNPROCESSABLE_ENTITY,
            "request": request_info,
            "message": "Validation failed",
            "data": {"errors": errors},
        }
        log_level = "warning"

    elif isinstance(exc, HTTPException):
        # FastAPI HTTP exceptions
        error_obj = {
            "name": "HTTPException",
            "success": False,
            "statusCode": exc.status_code,
            "request": request_info,
            "message": exc.detail if isinstance(exc.detail, str) else "HTTP error",
            "data": exc.detail if not isinstance(exc.detail, str) else None,
        }
        log_level = "error" if exc.status_code >= 500 else "warning"

    else:
        # Unexpected errors (500)
        error_obj = {
            "name": type(exc).__name__,
            "success": False,
            "statusCode": status.HTTP_500_INTERNAL_SERVER_ERROR,
            "request": request_info,
            "message": "An unexpected error occurred" if is_production else str(exc),
            "data": None,
        }

        # Add trace in non-production
        if not is_production:
            error_obj["trace"] = traceback.format_exc()

        log_level = "error"

    # Log the error with appropriate level
    log_func = getattr(logger, log_level)
    log_func(
        f"[{correlation_id}] {error_obj['name']}: {error_obj['message']}",
        status_code=error_obj["statusCode"],
        method=request_info["method"],
        url=request_info["url"],
    )

    return ORJSONResponse(
        status_code=error_obj["statusCode"],
        content=error_obj,
    )
