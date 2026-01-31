from app.config.settings import get_settings
import os
from typing import Any

from fastapi import Request
from fastapi.responses import ORJSONResponse
from loguru import logger

from app.config.enums import Environment


def http_response(
    message: str,
    data: Any = None,
    status_code: int = 200,
    request: Request | None = None,
) -> ORJSONResponse:
    """
    Create standardized HTTP success response.

    Args:
        message: Response message
        data: Response data
        status_code: HTTP status code
        request: FastAPI request object

    Returns:
        JSONResponse with standardized format
    """

    response = {
        "success": True,
        "statusCode": status_code,
        "request": {
            "ip": request.client.host if request and request.client else None,
            "method": request.method if request else None,
            "url": str(request.url) if request else None,
            "correlationId": (
                getattr(request.state, "correlation_id", None) if request else None
            ),
        },
        "message": message,
        "data": data,
    }

    # Remove sensitive data in production
    if os.getenv("ENVIRONMENT") == Environment.PRODUCTION:
        response["request"]["ip"] = None

    # Log response
    logger.info(
        "CONTROLLER_RESPONSE",
        response=response,
    )

    return ORJSONResponse(status_code=status_code, content=response)


def http_error(
    message: str,
    status_code: int = 400,
    data: Any = None,
    request: Request | None = None,
) -> ORJSONResponse:
    """
    Create standardized HTTP error response.

    Args:
        message: Error message
        status_code: HTTP status code
        data: Additional error data
        request: FastAPI request object

    Returns:
        ORJSONResponse with standardized error format
    """
    settings = get_settings()

    response = {
        "success": False,
        "statusCode": status_code,
        "request": {
            "ip": request.client.host if request and request.client else None,
            "method": request.method if request else None,
            "url": str(request.url) if request else None,
            "correlationId": (
                getattr(request.state, "correlation_id", None) if request else None
            ),
        },
        "message": message,
        "data": data,
    }

    # Remove sensitive data in production
    if settings.environment == Environment.PRODUCTION:
        response["request"]["ip"] = None

    logger.error(
        "CONTROLLER_ERROR",
        status_code=status_code,
        message=message,
        response=response,
    )

    return ORJSONResponse(status_code=status_code, content=response)
