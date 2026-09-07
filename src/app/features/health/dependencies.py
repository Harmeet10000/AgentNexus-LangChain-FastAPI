"""Dependency wiring for health feature."""

from fastapi import Request

from .service import HealthService


def get_health_service(request: Request) -> HealthService:
    """Build the health service from resources initialized during application startup."""
    state = request.app.state
    return HealthService(
        mongo_client=getattr(state, "mongo_client", None),
        redis_client=getattr(state, "redis", None),
        postgres_session_factory=getattr(state, "db_session_local", None),
        neo4j_driver=getattr(state, "neo4j_driver", None),
        celery_app=getattr(state, "celery", None),
        graph_memory_client=getattr(state, "graphiti", None),
        cognee_config=getattr(state, "cognee_config", None),
    )
