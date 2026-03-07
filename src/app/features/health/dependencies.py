"""Dependency wiring for health feature."""

from celery import Celery
from fastapi import Depends, Request
from motor.motor_asyncio import AsyncIOMotorClient
from neo4j import AsyncDriver
from redis.asyncio import Redis
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from .service import HealthService


def get_health_mongodb_client(request: Request) -> AsyncIOMotorClient | None:
    return getattr(request.app.state, "mongo_client", None)


def get_health_redis_client(request: Request) -> Redis | None:
    return getattr(request.app.state, "redis", None)


def get_health_postgres_session_factory(
    request: Request,
) -> async_sessionmaker[AsyncSession] | None:
    return getattr(request.app.state, "db_session_local", None)


def get_health_neo4j_driver(request: Request) -> AsyncDriver | None:
    return getattr(request.app.state, "neo4j_driver", None)


def get_health_celery_app(request: Request) -> Celery | None:
    return getattr(request.app.state, "celery", None)


def get_health_service(
    mongo_client: AsyncIOMotorClient | None = Depends(get_health_mongodb_client),
    redis_client: Redis | None = Depends(get_health_redis_client),
    postgres_session_factory: async_sessionmaker[AsyncSession] | None = Depends(
        get_health_postgres_session_factory
    ),
    neo4j_driver: AsyncDriver | None = Depends(get_health_neo4j_driver),
    celery_app: Celery | None = Depends(get_health_celery_app),
) -> HealthService:
    return HealthService(
        mongo_client=mongo_client,
        redis_client=redis_client,
        postgres_session_factory=postgres_session_factory,
        neo4j_driver=neo4j_driver,
        celery_app=celery_app,
    )

