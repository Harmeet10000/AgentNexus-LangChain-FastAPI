"""MongoDB connection and database management."""

from motor.motor_asyncio import AsyncIOMotorClient

# from pymongo.read_concern import ReadConcern
from pymongo.write_concern import WriteConcern


def create_mongo_client(uri: str) -> AsyncIOMotorClient:
    """
    Create and return a configured AsyncIOMotorClient.
    The client is thread-safe and intended to live for the entire process lifetime.
    """
    return AsyncIOMotorClient(
        uri,
        # Connection pool
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
