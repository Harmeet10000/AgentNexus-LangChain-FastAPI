from fastapi import Depends, Request
from fastapi.security import HTTPBearer
from motor.motor_asyncio import AsyncIOMotorClient
from neo4j import Driver
from redis.asyncio import Redis
from sqlalchemy.ext.asyncio import AsyncSession

from app.config.settings import get_settings
from app.connections import get_mongodb, get_neo4j_driver, get_postgres_db, get_redis

security = HTTPBearer()
settings = get_settings()


def get_health_mongodb_client(db=Depends(get_mongodb)) -> AsyncIOMotorClient:
    return db.client


def get_health_redis_client(redis=Depends(get_redis)) -> Redis:
    return redis


async def get_health_postgres_session(
    session: AsyncSession = Depends(get_postgres_db),
) -> AsyncSession:
    return session


async def get_health_neo4j_driver(
    driver: Driver = Depends(get_neo4j_driver),
) -> Driver:
    return driver


def get_health_celery_app(request: Request):
    return getattr(request.app.state, "celery", None)
