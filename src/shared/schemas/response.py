"""Standardized HTTP response utilities."""

from typing import Any, Optional, Dict
from fastapi import Request
from fastapi.responses import JSONResponse
from core.config.settings import settings
from core.config.logging_config import LoggerAdapter
from shared.utils.id_generator import generate_correlation_id

logger = LoggerAdapter(__name__)


class StandardResponse:
    """Standardized HTTP response utility for consistent API responses."""

    @staticmethod
    def success(
        request: Request,
        status_code: int,
        message: str,
        data: Any = None,
        correlation_id: Optional[str] = None,
    ) -> JSONResponse:
        """Create standardized success response."""

        # Generate correlation ID if not provided
        if not correlation_id:
            correlation_id = getattr(
                request.state, "correlation_id", generate_correlation_id()
            )

        response_data = {
            "success": True,
            "status_code": status_code,
            "request": {
                "ip": request.client.host if request.client else None,
                "method": request.method,
                "url": str(request.url),
                "correlation_id": correlation_id,
            },
            "message": message,
            "data": data,
        }

        # Remove sensitive info in production
        if settings.is_production:
            del response_data["request"]["ip"]

        # Log the response
        logger.info("CONTROLLER_RESPONSE", extra={"response": response_data})

        return JSONResponse(status_code=status_code, content=response_data)

    @staticmethod
    def error(
        request: Request,
        status_code: int,
        message: str,
        error_details: Optional[Dict[str, Any]] = None,
        correlation_id: Optional[str] = None,
    ) -> JSONResponse:
        """Create standardized error response."""

        # Generate correlation ID if not provided
        if not correlation_id:
            correlation_id = getattr(
                request.state, "correlation_id", generate_correlation_id()
            )

        response_data = {
            "success": False,
            "status_code": status_code,
            "request": {
                "ip": request.client.host if request.client else None,
                "method": request.method,
                "url": str(request.url),
                "correlation_id": correlation_id,
            },
            "message": message,
            "error": error_details,
        }

        # Remove sensitive info in production
        if settings.is_production:
            del response_data["request"]["ip"]
            # Remove detailed error info in production
            if error_details and not settings.debug:
                response_data["error"] = "Internal server error"

        # Log the error response
        logger.error("CONTROLLER_ERROR_RESPONSE", extra={"response": response_data})

        return JSONResponse(status_code=status_code, content=response_data)


# Convenience functions for common use cases
def http_success(
    request: Request, status_code: int = 200, message: str = "Success", data: Any = None
) -> JSONResponse:
    """Convenience function for success responses."""
    return StandardResponse.success(request, status_code, message, data)


def http_error(
    request: Request,
    error: Exception,
    status_code: int = 500,
    error_details: Optional[Dict[str, Any]] = None,
) -> JSONResponse:
    """Convenience function for error responses."""
    message = str(error) if error else "Internal server error"
    return StandardResponse.error(request, status_code, message, error_details)
