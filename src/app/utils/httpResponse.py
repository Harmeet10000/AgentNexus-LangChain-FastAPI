import os
from typing import Any

from fastapi import Request
from fastapi.responses import JSONResponse
from loguru import logger

from app.shared.enums import Environment


def http_response(
    message: str,
    data: Any = None,
    status_code: int = 200,
    request: Request | None = None,
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

    return JSONResponse(status_code=status_code, content=response)
