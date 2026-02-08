"""Health check controller."""

import time
from typing import Any

from fastapi import Request

from app.utils.httpResponse import http_response

from .service import (
    check_database,
    check_disk,
    check_memory,
    check_redis,
    get_application_health,
    get_system_health,
)


async def self_info(request: Request) -> Any:
    """Get basic server information."""
    server_info: dict[str, Any] = {
        "server": request.app.title or "unknown",
        "version": request.app.version or "unknown",
        "client": request.client.host if request.client else "unknown",
        "timestamp": time.time(),
    }

    return http_response(
        message="Server information retrieved",
        data=server_info,
        status_code=200,
        request=request,
    )


async def health_check(request: Request) -> Any:
    """
    Comprehensive health check endpoint.
    Checks MongoDB, Redis, memory, and disk health.
    """
    # Get MongoDB client from app.state
    mongo_client = getattr(request.app.state, "mongo_client", None)

    # Get Redis client
    redis_client = None
    try:
        redis_client = getattr(request.app.state, "redis", None)
    except RuntimeError:
        redis_client = None

    # Run async checks for external services
    database_check = {"status": "unknown", "state": "not_configured"}
    if mongo_client:
        try:
            database_check = await check_database(mongo_client)
        except Exception as e:
            database_check = {"status": "unhealthy", "state": "error", "error": str(e)}

    redis_check = {"status": "unknown", "state": "not_configured"}
    if redis_client:
        try:
            redis_check = await check_redis(redis_client)
        except Exception as e:
            redis_check = {"status": "unhealthy", "state": "error", "error": str(e)}

    # Synchronous checks
    memory_check = check_memory()
    disk_check = check_disk()

    # Determine overall health status
    all_checks = [database_check, redis_check, memory_check, disk_check]
    overall_status = "healthy"

    if any(check.get("status") == "unhealthy" for check in all_checks):
        overall_status = "unhealthy"
    elif any(check.get("status") == "warning" for check in all_checks):
        overall_status = "degraded"

    health_data = {
        "status": overall_status,
        "timestamp": time.time(),
        "application": get_application_health(),
        "system": get_system_health(),
        "checks": {
            "database": database_check,
            "redis": redis_check,
            "memory": memory_check,
            "disk": disk_check,
        },
    }

    # Return appropriate status code based on health
    status_code = 200 if overall_status == "healthy" else 503

    return http_response(
        message=f"Health check: {overall_status}",
        data=health_data,
        status_code=status_code,
        request=request,
    )
