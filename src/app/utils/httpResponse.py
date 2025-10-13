"""Standardized HTTP response utility."""
from typing import Any, Optional
from fastapi import Request
from fastapi.responses import JSONResponse
from loguru import logger
from app.shared.enums import Environment
from app.middleware.server_middleware import get_correlation_id
import os


def http_response(
    message: str,
    data: Any = None,
    status_code: int = 200,
    request: Optional[Request] = None,
) -> JSONResponse:
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
    correlation_id = get_correlation_id() if request else None

    response = {
        "success": True,
        "statusCode": status_code,
        "request": {
            "ip": request.client.host if request and request.client else None,
            "method": request.method if request else None,
            "url": str(request.url) if request else None,
            "correlationId": correlation_id,
        },
        "message": message,
        "data": data,
    }

    # Remove sensitive data in production
    if os.getenv("ENVIRONMENT") == Environment.PRODUCTION:
        response["request"]["ip"] = None

    # Log response
    logger.info("CONTROLLER_RESPONSE", extra={"meta": response})

    return JSONResponse(status_code=status_code, content=response)
