"""Vector repository implementation."""

from typing import List, Dict, Any, Optional
from fastapi import Depends

from core.config.logging_config import LoggerAdapter
from services.pinecone.client import vector_store_service

logger = LoggerAdapter(__name__)


class VectorRepository:
    """Repository for vector store operations."""
    
    def __init__(self):
        """Initialize vector repository."""
        self.vector_store = vector_store_service
    
    async def add_documents(
        self,
        documents: List[Dict[str, Any]],
        namespace: str = "default",
        batch_size: int = 100
    ) -> List[str]:
        """Add documents to vector store."""
        try:
            return await self.vector_store.add_documents(
                documents=documents,
                namespace=namespace,
                batch_size=batch_size
            )
        except Exception as e:
            logger.error("Failed to add documents to vector store", error=str(e))
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
            return await self.vector_store.similarity_search(
                query=query,
                k=k,
                namespace=namespace,
                filter=filter,
                use_cache=use_cache
            )
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
            return await self.vector_store.similarity_search_with_relevance_scores(
                query=query,
                k=k,
                namespace=namespace,
                score_threshold=score_threshold
            )
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
            return await self.vector_store.hybrid_search(
                query=query,
                k=k,
                namespace=namespace,
                metadata_filter=metadata_filter,
                keyword_filter=keyword_filter
            )
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
            return await self.vector_store.delete_by_ids(ids, namespace)
        except Exception as e:
            logger.error("Failed to delete vectors by IDs", error=str(e))
            raise
    
    async def delete_by_metadata(
        self,
        metadata_filter: Dict[str, Any],
        namespace: str = "default"
    ) -> bool:
        """Delete vectors by metadata filter."""
        try:
            return await self.vector_store.delete_by_metadata(metadata_filter, namespace)
        except Exception as e:
            logger.error("Failed to delete vectors by metadata", error=str(e))
            raise
    
    async def update_metadata(
        self,
        id: str,
        metadata: Dict[str, Any],
        namespace: str = "default"
    ) -> bool:
        """Update metadata for a vector."""
        try:
            return await self.vector_store.update_metadata(id, metadata, namespace)
        except Exception as e:
            logger.error("Failed to update vector metadata", error=str(e))
            raise
    
    async def get_index_stats(self) -> Dict[str, Any]:
        """Get index statistics."""
        try:
            return self.vector_store.get_index_stats()
        except Exception as e:
            logger.error("Failed to get index stats", error=str(e))
            raise


def get_vector_repository() -> VectorRepository:
    """Dependency to get vector repository."""
    return VectorRepository()