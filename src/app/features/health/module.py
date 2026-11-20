"""Health slice composition descriptor and process-local service factory."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

from app.shared.module import Module

if TYPE_CHECKING:
    from typing import Any

# Composition imports below are intentionally inside the factory (D5).
# ruff: noqa: PLC0415


def _routes() -> tuple[Any, ...]:
    """Load HTTP routes only when an API composition root asks for routes."""
    from .adapters.inbound.http.router import deep_health_router, router

    return (router, deep_health_router)


def get_health_service(request: Any) -> Any:
    """Build the application service from startup-owned app state.

    Driver imports intentionally live inside this factory. Importing the
    descriptor for worker or scheduler discovery therefore remains cheap and
    independent of the API's database clients.
    """
    from app.config import get_settings
    from app.features.health.adapters.outbound.celery.probe import CeleryProbe
    from app.features.health.adapters.outbound.mongodb.probe import MongoDBProbe
    from app.features.health.adapters.outbound.neo4j.probe import (
        AgentMemoryProbe,
        GraphitiProbe,
        Neo4jProbe,
    )
    from app.features.health.adapters.outbound.postgres.probe import PostgresProbe
    from app.features.health.adapters.outbound.redis.probe import RedisProbe
    from app.features.health.adapters.outbound.system.probe import (
        DiskProbe,
        MemoryProbe,
        SystemProbe,
    )
    from app.features.health.application.service import HealthService

    state = request.app.state
    settings = get_settings()
    start_time = time.time()
    return HealthService(
        probes=(
            MongoDBProbe(getattr(state, "mongo_client", None)),
            RedisProbe(getattr(state, "redis", None)),
            PostgresProbe(getattr(state, "db_session_local", None)),
            Neo4jProbe(getattr(state, "neo4j_driver", None)),
            GraphitiProbe(getattr(state, "graphiti", None)),
            AgentMemoryProbe(
                getattr(state, "cognee_config", None),
                getattr(state, "neo4j_driver", None),
            ),
            CeleryProbe(getattr(state, "celery", None)),
            MemoryProbe(),
            DiskProbe(),
            SystemProbe(settings.ENVIRONMENT, start_time),
        ),
        environment=settings.ENVIRONMENT,
        start_time=start_time,
    )


MODULE = Module(name="health", routes=(_routes,))
