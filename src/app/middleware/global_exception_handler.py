import os
import traceback

from fastapi import Request, status
from fastapi.exceptions import HTTPException, RequestValidationError
from fastapi.responses import JSONResponse

from app.utils.exceptions import APIException
from app.utils.logger import logger


async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """
    Single unified exception handler for ALL errors
    Handles: APIException, ValidationError, HTTPException, and unexpected errors
    """

    correlation_id = getattr(request.state, "correlation_id", None)
    is_production = os.getenv("ENV", "").lower() == "production"

    # Determine error type and build error object
    if isinstance(exc, APIException):
        # Custom API exceptions
        error_obj = {
            "name": exc.name,
            "success": False,
            "statusCode": exc.status_code,
            "request": {
                "ip": request.client.host if request.client else None,
                "method": request.method,
                "url": str(request.url),
                "correlationId": correlation_id,
            },
            "message": exc.message,
            "data": exc.data,
            "trace": {"error": traceback.format_exc()}
            if exc.status_code >= 500
            else None,
        }
        log_type = "CONTROLLER_ERROR"

    elif isinstance(exc, RequestValidationError):
        # Pydantic validation errors (422)
        errors = []
        for error in exc.errors():
            errors.append(
                {
                    "field": ".".join(str(x) for x in error["loc"][1:]),
                    "message": error["msg"],
                    "type": error["type"],
                }
            )

        error_obj = {
            "name": "ValidationError",
            "success": False,
            "statusCode": status.HTTP_422_UNPROCESSABLE_ENTITY,
            "request": {
                "ip": request.client.host if request.client else None,
                "method": request.method,
                "url": str(request.url),
                "correlationId": correlation_id,
            },
            "message": "Validation failed",
            "data": {"errors": errors},
            "trace": None,
        }
        log_type = "VALIDATION_ERROR"

    elif isinstance(exc, HTTPException):
        # FastAPI HTTP exceptions
        error_obj = {
            "name": "HTTPException",
            "success": False,
            "statusCode": exc.status_code,
            "request": {
                "ip": request.client.host if request.client else None,
                "method": request.method,
                "url": str(request.url),
                "correlationId": correlation_id,
            },
            "message": exc.detail if isinstance(exc.detail, str) else "HTTP error",
            "data": None,
            "trace": None,
        }
        log_type = "HTTP_ERROR"

    else:
        # Unexpected errors (500)
        error_obj = {
            "name": type(exc).__name__,
            "success": False,
            "statusCode": status.HTTP_500_INTERNAL_SERVER_ERROR,
            "request": {
                "ip": request.client.host if request.client else None,
                "method": request.method,
                "url": str(request.url),
                "correlationId": correlation_id,
            },
            "message": "Something went wrong",
            "data": None,
            "trace": {"error": traceback.format_exc()},
        }
        log_type = "UNHANDLED_ERROR"

    # Log the error
    logger.error(
        log_type,
        error_name=error_obj["name"],
        status_code=error_obj["statusCode"],
        method=error_obj["request"]["method"],
        url=error_obj["request"]["url"],
        correlation_id=correlation_id,
        message=error_obj["message"],
    )

    # Remove sensitive data in production
    if is_production:
        if "ip" in error_obj["request"]:
            del error_obj["request"]["ip"]
        if error_obj.get("trace"):
            del error_obj["trace"]

    return JSONResponse(status_code=error_obj["statusCode"], content=error_obj)
