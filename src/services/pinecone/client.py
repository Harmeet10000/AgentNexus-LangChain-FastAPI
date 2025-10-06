"""Pinecone vector store service."""

import os
from typing import List, Dict, Any, Optional, Tuple
from uuid import uuid4

from pinecone import Pinecone, ServerlessSpec
from langchain_community.vectorstores import Pinecone as LangchainPinecone
from langchain.schema import Document

from src.core.config.settings import settings
from src.core.config.logging_config import LoggerAdapter
from src.core.cache.redis_client import redis_cache

logger = LoggerAdapter(__name__)

# Global Pinecone client and index
pinecone_client: Optional[Pinecone] = None
pinecone_index = None


def initialize_pinecone():
    """Initialize Pinecone client and index."""
    global pinecone_client, pinecone_index

    try:
        logger.info("Initializing Pinecone")

        # Initialize Pinecone client
        pinecone_client = Pinecone(
            api_key=settings.pinecone_api_key
        )

        # Check if index exists
        existing_indexes = pinecone_client.list_indexes().names()

        if settings.pinecone_index_name not in existing_indexes:
            logger.info(f"Creating Pinecone index: {settings.pinecone_index_name}")
            pinecone_client.create_index(
                name=settings.pinecone_index_name,
                dimension=settings.pinecone_dimension,
                metric=settings.pinecone_metric,
                spec=ServerlessSpec(
                    cloud='aws',
                    region='us-east-1'  # Change based on your preference
                )
            )

        # Get index reference
        pinecone_index = pinecone_client.Index(settings.pinecone_index_name)

        # Get index stats
        stats = pinecone_index.describe_index_stats()
        logger.info(f"Pinecone index stats: {stats}")

        logger.info("Pinecone initialized successfully")

    except Exception as e:
        logger.error("Failed to initialize Pinecone", error=str(e))
        raise


class VectorStoreService:
    """Service for vector store operations."""

    def __init__(self):
        """Initialize vector store service."""
        from src.services.langchain.gemini_service import gemini_service

        self.embedding_model = gemini_service.embedding_model
        self.vectorstore = None

        if pinecone_index:
            self.vectorstore = LangchainPinecone(
                pinecone_index,
                self.embedding_model.embed_query,
                "text",
                namespace="default"
            )

    async def add_documents(
        self,
        documents: List[Dict[str, Any]],
        namespace: str = "default",
        batch_size: int = 100
    ) -> List[str]:
        """Add documents to vector store."""
        try:
            if not self.vectorstore:
                raise RuntimeError("Vector store not initialized")

            # Convert to LangChain Document objects
            lc_documents = []
            for doc in documents:
                lc_doc = Document(
                    page_content=doc.get("content", ""),
                    metadata=doc.get("metadata", {})
                )
                lc_documents.append(lc_doc)

            # Generate IDs if not provided
            ids = [str(uuid4()) for _ in range(len(lc_documents))]

            # Add documents in batches
            all_ids = []
            for i in range(0, len(lc_documents), batch_size):
                batch_docs = lc_documents[i:i + batch_size]
                batch_ids = ids[i:i + batch_size]

                # Create vectorstore for specific namespace
                namespace_vectorstore = LangchainPinecone(
                    pinecone_index,
                    self.embedding_model.embed_query,
                    "text",
                    namespace=namespace
                )

                # Add documents
                added_ids = await namespace_vectorstore.aadd_documents(
                    documents=batch_docs,
                    ids=batch_ids
                )
                all_ids.extend(added_ids)

                logger.debug(f"Added batch {i//batch_size + 1} to vector store")

            logger.info(f"Added {len(all_ids)} documents to vector store")
            return all_ids

        except Exception as e:
            logger.error("Failed to add documents", error=str(e))
            raise

    async def similarity_search(
        self,
        query: str,
        k: int = 4,
        namespace: str = "default",
        filter: Optional[Dict[str, Any]] = None,
        use_cache: bool = True
    ) -> List[Dict[str, Any]]:
        """Perform similarity search."""
        try:
            if not self.vectorstore:
                raise RuntimeError("Vector store not initialized")

            # Check cache
            cache_key = f"vector:search:{namespace}:{query}:{k}"
            if use_cache:
                cached_results = await redis_cache.get(cache_key)
                if cached_results:
                    logger.info("Returning cached search results")
                    return cached_results

            # Create vectorstore for specific namespace
            namespace_vectorstore = LangchainPinecone(
                pinecone_index,
                self.embedding_model.embed_query,
                "text",
                namespace=namespace
            )

            # Perform search
            results = await namespace_vectorstore.asimilarity_search_with_score(
                query=query,
                k=k,
                filter=filter
            )

            # Format results
            formatted_results = [
                {
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                    "score": float(score)
                }
                for doc, score in results
            ]

            # Cache results
            if use_cache:
                await redis_cache.set(cache_key, formatted_results, ttl=1800)

            logger.info(f"Found {len(formatted_results)} similar documents")
            return formatted_results

        except Exception as e:
            logger.error("Failed to perform similarity search", error=str(e))
            raise

    async def similarity_search_with_relevance_scores(
        self,
        query: str,
        k: int = 4,
        namespace: str = "default",
        score_threshold: float = 0.5
    ) -> List[Dict[str, Any]]:
        """Perform similarity search with relevance score filtering."""
        try:
            results = await self.similarity_search(query, k * 2, namespace)  # Get more results

            # Filter by score threshold
            filtered_results = [
                result for result in results
                if result["score"] >= score_threshold
            ][:k]

            logger.info(f"Filtered to {len(filtered_results)} results above threshold {score_threshold}")
            return filtered_results

        except Exception as e:
            logger.error("Failed to perform filtered similarity search", error=str(e))
            raise

    async def hybrid_search(
        self,
        query: str,
        k: int = 4,
        namespace: str = "default",
        metadata_filter: Optional[Dict[str, Any]] = None,
        keyword_filter: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Perform hybrid search combining vector and keyword search."""
        try:
            # Vector search
            vector_results = await self.similarity_search(
                query=query,
                k=k * 2,  # Get more results for filtering
                namespace=namespace,
                filter=metadata_filter
            )

            # Filter by keywords if provided
            if keyword_filter:
                filtered_results = []
                for result in vector_results:
                    if keyword_filter.lower() in result["content"].lower():
                        filtered_results.append(result)

                results = filtered_results[:k]
            else:
                results = vector_results[:k]

            logger.info(f"Hybrid search returned {len(results)} results")
            return results

        except Exception as e:
            logger.error("Failed to perform hybrid search", error=str(e))
            raise

    async def delete_by_ids(
        self,
        ids: List[str],
        namespace: str = "default"
    ) -> bool:
        """Delete vectors by IDs."""
        try:
            if not pinecone_index:
                raise RuntimeError("Pinecone index not initialized")

            pinecone_index.delete(
                ids=ids,
                namespace=namespace
            )

            logger.info(f"Deleted {len(ids)} vectors from namespace {namespace}")
            return True

        except Exception as e:
            logger.error("Failed to delete vectors", error=str(e))
            return False

    async def delete_by_metadata(
        self,
        metadata_filter: Dict[str, Any],
        namespace: str = "default"
    ) -> bool:
        """Delete vectors by metadata filter."""
        try:
            if not pinecone_index:
                raise RuntimeError("Pinecone index not initialized")

            pinecone_index.delete(
                filter=metadata_filter,
                namespace=namespace
            )

            logger.info(f"Deleted vectors with filter {metadata_filter} from namespace {namespace}")
            return True

        except Exception as e:
            logger.error("Failed to delete by metadata", error=str(e))
            return False

    async def update_metadata(
        self,
        id: str,
        metadata: Dict[str, Any],
        namespace: str = "default"
    ) -> bool:
        """Update metadata for a vector."""
        try:
            if not pinecone_index:
                raise RuntimeError("Pinecone index not initialized")

            pinecone_index.update(
                id=id,
                set_metadata=metadata,
                namespace=namespace
            )

            logger.info(f"Updated metadata for vector {id}")
            return True

        except Exception as e:
            logger.error("Failed to update metadata", error=str(e))
            return False

    def get_index_stats(self) -> Dict[str, Any]:
        """Get index statistics."""
        try:
            if not pinecone_index:
                raise RuntimeError("Pinecone index not initialized")

            stats = pinecone_index.describe_index_stats()
            return stats

        except Exception as e:
            logger.error("Failed to get index stats", error=str(e))
            return {}


# Create global instance
vector_store_service = VectorStoreService()
