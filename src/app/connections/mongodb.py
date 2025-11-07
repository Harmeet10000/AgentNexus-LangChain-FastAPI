"""MongoDB connection and database management."""

from typing import Any

from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase
from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError

from app.core.settings import get_settings
from app.utils.logger import logger

# Global MongoDB client and database instances
mongodb_client: AsyncIOMotorClient | None = None
mongodb: AsyncIOMotorDatabase | None = None


async def connect_to_mongodb() -> bool:
    """Create MongoDB connection with production-grade options."""
    global mongodb_client, mongodb

    try:

        # MongoDB connection options matching JS configuration
        mongo_options: dict[str, Any] = {
            "maxPoolSize": 10,
            "minPoolSize": 2,
            "maxIdleTimeMS": 30000,
            "serverSelectionTimeoutMS": 5000,
            "socketTimeoutMS": 45000,
            "readPreference": "secondaryPreferred",
            "w": "majority",
            "journal": True,
            "wtimeoutMS": 5000,
            "readConcernLevel": "majority",
        }

        mongodb_client = AsyncIOMotorClient(get_settings().MONGODB_URL, **mongo_options)

        # Verify connection
        await mongodb_client.admin.command("ping")

        mongodb = mongodb_client[get_settings().MONGODB_DATABASE]

        # Get connection info
        server_info = await mongodb_client.server_info()
        host = mongodb_client.address[0] if mongodb_client.address else "unknown"

        logger.info(
            f"MongoDB Connected: {host}",
            readyState=1,
            poolSize=mongo_options["maxPoolSize"],
            version=server_info.get("version", "unknown"),
        )


        return True

    except (ConnectionFailure, ServerSelectionTimeoutError) as e:
        logger.error("MongoDB client error", error=str(e))
        raise



async def close_mongodb_connection() -> None:
    """Close MongoDB connection."""
    global mongodb_client, mongodb

    if mongodb_client:
        logger.warning("MongoDB connection closing")
        mongodb_client.close()
        mongodb_client = None
        mongodb = None
        logger.info("MongoDB connection closed")
