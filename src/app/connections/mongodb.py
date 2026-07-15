"""MongoDB connection and database management."""

from typing import Any

from beanie import init_beanie
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase
from starlette.requests import HTTPConnection


async def create_mongo_client(
    uri: str, db_name: str, document_models: list[type]
) -> tuple[AsyncIOMotorClient[Any], AsyncIOMotorDatabase[Any]]:
    """
    Initialize database connection using Beanie's recommended approach.
    """
    client = AsyncIOMotorClient[Any](
        host=uri,  # Connection pool
        maxPoolSize=10,
        minPoolSize=2,
        maxIdleTimeMS=30_000,
        # Timeouts
        serverSelectionTimeoutMS=5_000,
        socketTimeoutMS=45_000,
        # Read / write behavior (use string for readPreference with Motor)
        readPreference="secondaryPreferred",
        readConcernLevel="majority",
        w="majority",
        journal=True,
        wTimeoutMS=5_000,
        # Quality-of-life defaults
        retryReads=True,
        retryWrites=True,
        tz_aware=True,
    )
    database: AsyncIOMotorDatabase[Any] = client[db_name]

    await init_beanie(
        database=database,  # type: ignore
        document_models=document_models,  # type: ignore
    )

    return client, database


async def get_mongodb(connection: HTTPConnection) -> AsyncIOMotorDatabase[Any]:
    return connection.app.state.db
