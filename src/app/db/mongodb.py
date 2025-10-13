"""MongoDB connection and database management."""

from typing import Optional

from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase

from src.core.config.logging_config import LoggerAdapter
from src2.app.core.settings import settings

logger = LoggerAdapter(__name__)

# Global MongoDB client and database instances
mongodb_client: Optional[AsyncIOMotorClient] = None
mongodb: Optional[AsyncIOMotorDatabase] = None


async def connect_to_mongodb() -> None:
    """Create MongoDB connection."""
    global mongodb_client, mongodb

    try:
        logger.info("Connecting to MongoDB", url=settings.mongodb_url)

        mongodb_client = AsyncIOMotorClient(
            settings.mongodb_url,
            maxPoolSize=50,
            minPoolSize=10,
            serverSelectionTimeoutMS=5000,
        )

        # Verify connection
        await mongodb_client.admin.command("ping")

        mongodb = mongodb_client[settings.mongodb_database]

        logger.info("Successfully connected to MongoDB")

    except Exception as e:
        logger.error("Failed to connect to MongoDB", error=str(e))
        raise


async def close_mongodb_connection() -> None:
    """Close MongoDB connection."""
    global mongodb_client

    if mongodb_client:
        logger.info("Closing MongoDB connection")
        mongodb_client.close()
        mongodb_client = None
        logger.info("MongoDB connection closed")
