from typing import Any

from fastapi import Request

from app.connections import mongodb as mongo_db
from app.connections.redis import create_redis_client
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
    server_info: dict[str, Any] = {
        "server": request.app.title or "unknown",
        "container": request.client.host if request.client else "unknown",
        "timestamp": request.scope.get("time", None) or None,
    }

    return http_response(
        message="Success",
        data=server_info,
        status_code=200,
        request=request,
    )


async def health_check(request: Request) -> Any:
    # Gather checks — use app-level clients if available
    db_client = getattr(mongo_db, "mongodb_client", None)
    redis_client = None
    try:
        redis_client = get_redis_client()
    except RuntimeError:
        redis_client = None

    # run async checks for external services using helper functions
    database_check = {"status": "unknown"}
    redis_check = {"status": "unknown"}
    try:
        if db_client:
            database_check = await check_database(db_client)
    except Exception as e:
        database_check = {"status": "unhealthy", "error": str(e)}

    try:
        if redis_client:
            redis_check = await check_redis(redis_client)
    except Exception as e:
        redis_check = {"status": "unhealthy", "error": str(e)}

    checks = {
        "database": database_check,
        "redis": redis_check,
        "memory": check_memory(),
        "disk": check_disk(),
    }

    health_data = {
        "application": get_application_health(),
        "system": get_system_health(),
        "timestamp": __import__("time").time(),
        "checks": checks,
    }

    return http_response(
        message="Success",
        data=health_data,
        status_code=200,
        request=request,
    )
