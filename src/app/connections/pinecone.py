"""Pinecone vector store service."""

from pinecone import Pinecone, ServerlessSpec

from app.core.settings import get_settings
from app.utils.logger import logger

# Global Pinecone client and index
pinecone_client: Pinecone | None = None
pinecone_index = None


def initialize_pinecone() -> None:
    """Initialize Pinecone client and index."""
    global pinecone_client, pinecone_index

    try:
        logger.info("Initializing Pinecone", index_name=get_settings().PINECONE_INDEX_NAME)

        # Initialize Pinecone client
        pinecone_client = Pinecone(api_key=get_settings().PINECONE_API_KEY)

        # Check if index exists
        existing_indexes = pinecone_client.list_indexes().names()

        if get_settings().PINECONE_INDEX_NAME not in existing_indexes:
            logger.info(
                f"Creating Pinecone index: {get_settings().PINECONE_INDEX_NAME}",
                dimension=get_settings().PINECONE_DIMENSION,
                metric=get_settings().PINECONE_METRIC,
            )
            pinecone_client.create_index(
                name=get_settings().PINECONE_INDEX_NAME,
                dimension=get_settings().PINECONE_DIMENSION,
                metric=get_settings().PINECONE_METRIC,
                spec=ServerlessSpec(
                    cloud="aws",
                    region="us-east-1",  # Change based on your preference
                ),
            )

        # Get index reference
        pinecone_index = pinecone_client.Index(get_settings().PINECONE_INDEX_NAME)

        # Get index stats
        stats = pinecone_index.describe_index_stats()
        logger.info(
            "Pinecone initialized successfully",
            index_name=get_settings().PINECONE_INDEX_NAME,
            total_vectors=stats.get("total_vector_count", 0),
            dimension=stats.get("dimension", get_settings().PINECONE_DIMENSION),
        )

    except Exception as e:
        logger.error("Failed to initialize Pinecone", error=str(e), index_name=get_settings().PINECONE_INDEX_NAME)
        raise


class VectorStoreService:
    """Service for vector store operations."""

    def __init__(self):
        """Initialize vector store service."""


# Create global instance
vector_store_service = VectorStoreService()
