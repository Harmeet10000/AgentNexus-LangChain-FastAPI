"""MongoDB connection and database management."""

from beanie import init_beanie
from fastapi import Request
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase


async def create_mongo_client(uri: str, db_name: str, document_models: list):
    """
    Initialize database connection using Beanie's recommended approach.
    """
    client = AsyncIOMotorClient(
        uri,  # Connection pool
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
    database = client[db_name]

    await init_beanie(
        database=database,
        document_models=document_models,
    )

    return client, database


def get_db(request: Request) -> AsyncIOMotorDatabase:
    return request.app.state.db
