"""Standardized HTTP error response utility."""

import os

from fastapi import HTTPException, Request
from loguru import logger

from app.shared.enums import SOMETHING_WENT_WRONG, Environment


def http_error(
    message: str = SOMETHING_WENT_WRONG,
    status_code: int = 500,
    request: Request | None = None,
    exc: Exception | None = None,
) -> HTTPException:
    """
    Create standardized HTTP error response.

    Args:
        message: Error message
        status_code: HTTP status code
        request: FastAPI request object
        exc: Original exception

    Returns:
        HTTPException with standardized error format
    """

    error_obj = {
        "name": exc.__class__.__name__ if exc else "Error",
        "success": False,
        "statusCode": status_code,
        "request": {
            "ip": request.client.host if request and request.client else None,
            "method": request.method if request else None,
            "url": str(request.url) if request else None,
            "correlationId": getattr(request.state, "correlation_id", None)
            if request
            else None,
        },
        "message": message,
        "data": None,
        "trace": {"error": str(exc)} if exc else None,
    }

    # Log error
    logger.error("CONTROLLER_ERROR", extra={"meta": error_obj})

    # Remove sensitive data in production
    if os.getenv("ENVIRONMENT") == Environment.PRODUCTION:
        error_obj["trace"] = None

    return HTTPException(status_code=status_code, detail=error_obj)
