from typing import Any

from fastapi.responses import ORJSONResponse
from pydantic import BaseModel

from app.config import Environment, get_settings
from app.utils.logger import request_state


def _serialize_data(data: Any) -> Any:
    """Safely extract data from Pydantic models for ORJSON compatibility."""
    if isinstance(data, BaseModel):
        # mode="json" ensures dates/UUIDs are stringified correctly for ORJSON
        return data.model_dump(mode="json")
    elif isinstance(data, list):
        return [_serialize_data(item) for item in data]
    return data

def http_response(
    message: str,
    data: Any = None,
    status_code: int = 200,
) -> ORJSONResponse:
    """Create standardized HTTP success response using ContextVar."""
    settings = get_settings()
    ctx = request_state.get()

    ip = ctx.get("ip")
    # Remove sensitive data in production
    if settings.ENVIRONMENT == Environment.PRODUCTION:
        ip = None

    response = {
        "success": True,
        "statusCode": status_code,
        "request": {
            "ip": ip,
            "method": ctx.get("method"),
            "url": ctx.get("url"),
            # Match the key you set in your middleware ("request_id" or "correlation_id")
            "correlationId": ctx.get("request_id"),
        },
        "message": message,
        "data": _serialize_data(data),
    }

    return ORJSONResponse(status_code=status_code, content=response)


def http_error(
    message: str,
    status_code: int = 400,
    data: Any = None,
) -> ORJSONResponse:
    """Create standardized HTTP error response using ContextVar."""
    settings = get_settings()
    ctx = request_state.get()

    ip = ctx.get("ip")
    if settings.ENVIRONMENT == Environment.PRODUCTION:
        ip = None

    response = {
        "success": False,
        "statusCode": status_code,
        "request": {
            "ip": ip,
            "method": ctx.get("method"),
            "url": ctx.get("url"),
            "correlationId": ctx.get("request_id"),
        },
        "message": message,
        "data": _serialize_data(data),
    }

    return ORJSONResponse(status_code=status_code, content=response)
