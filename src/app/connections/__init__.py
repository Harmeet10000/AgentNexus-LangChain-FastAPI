"""Database connection dependencies."""

from app.connections.mongodb import create_mongo_client
from app.connections.redis import create_redis_client


__all__ = [
    "create_mongo_client",
    "create_redis_client",
    "get_db",
    "get_redis",
]
