"""Health check controller."""

import time
from typing import Any

from fastapi import Request

from app.utils import http_response

from .service import (
    check_celery,
    check_disk,
    check_memory,
    check_mongodb,
    check_neo4j,
    check_postgres,
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

    # Get Postgres session factory from app.state
    session_local = getattr(request.app.state, "db_session_local", None)

    # Get Neo4j driver from app.state
    neo4j_driver = getattr(request.app.state, "neo4j_driver", None)

    # Get optional Celery app from app.state
    celery_app = getattr(request.app.state, "celery", None)

    # Run async checks for external services
    database_check = {"status": "unknown", "state": "not_configured"}
    if mongo_client:
        try:
            database_check = await check_mongodb(mongo_client)
        except Exception as e:
            database_check = {"status": "unhealthy", "state": "error", "error": str(e)}

    redis_check = {"status": "unknown", "state": "not_configured"}
    if redis_client:
        try:
            redis_check = await check_redis(redis_client)
        except Exception as e:
            redis_check = {"status": "unhealthy", "state": "error", "error": str(e)}

    postgres_check = {"status": "unknown", "state": "not_configured"}
    if session_local:
        try:
            async with session_local() as session:
                postgres_check = await check_postgres(session)
        except Exception as e:
            postgres_check = {"status": "unhealthy", "state": "error", "error": str(e)}

    neo4j_check = {"status": "unknown", "state": "not_configured"}
    if neo4j_driver:
        try:
            neo4j_check = await check_neo4j(neo4j_driver)
        except Exception as e:
            neo4j_check = {"status": "unhealthy", "state": "error", "error": str(e)}

    celery_check = check_celery(celery_app)

    # Synchronous checks
    memory_check = check_memory()
    disk_check = check_disk()

    # Determine overall health status
    all_checks = [
        database_check,
        redis_check,
        postgres_check,
        neo4j_check,
        celery_check,
        memory_check,
        disk_check,
    ]
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
            "postgres": postgres_check,
            "neo4j": neo4j_check,
            "celery": celery_check,
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
