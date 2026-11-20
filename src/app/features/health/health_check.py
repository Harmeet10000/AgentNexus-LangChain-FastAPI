"""Compatibility imports for deep probes now owned by outbound adapters."""

from app.features.health.adapters.out.mongodb.probe import check_mongodb
from app.features.health.adapters.out.neo4j.probe import (
    check_cognee,
    check_graphiti,
    check_neo4j,
    check_neo4j_plugins,
)
from app.features.health.adapters.out.postgres.probe import check_postgres
from app.features.health.adapters.out.redis.probe import check_redis

ALL_PROBES = [
    check_postgres,
    check_redis,
    check_mongodb,
    check_neo4j,
    check_neo4j_plugins,
    check_graphiti,
    check_cognee,
]

__all__ = [
    "ALL_PROBES",
    "check_cognee",
    "check_graphiti",
    "check_mongodb",
    "check_neo4j",
    "check_neo4j_plugins",
    "check_postgres",
    "check_redis",
]
