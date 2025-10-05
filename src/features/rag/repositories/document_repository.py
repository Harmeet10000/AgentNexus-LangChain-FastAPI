"""Document repository implementation."""

from typing import Dict, Any, List, Optional
from datetime import datetime
from fastapi import Depends

from core.database.mongodb import get_database
from core.config.logging_config import LoggerAdapter

logger = LoggerAdapter(__name__)


class DocumentRepository:
    """Repository for document metadata operations."""
    
    def __init__(self, db):
        """Initialize document repository."""
        self.db = db
        self.documents = db.documents
        self.chunks = db.document_chunks
    
    async def save_document(self, document_data: Dict[str, Any]) -> str:
        """Save document metadata to database."""
        try:
            document_doc = {
                **document_data,
                "created_at": datetime.utcnow(),
                "updated_at": datetime.utcnow(),
                "status": "processed"
            }
            
            result = await self.documents.insert_one(document_doc)
            
            logger.info(f"Saved document metadata with ID: {result.inserted_id}")
            return str(result.inserted_id)
            
        except Exception as e:
            logger.error("Failed to save document metadata", error=str(e))
            raise
    
    async def get_document(self, document_id: str) -> Optional[Dict[str, Any]]:
        """Get document metadata by ID."""
        try:
            from bson import ObjectId
            
            document = await self.documents.find_one({"_id": ObjectId(document_id)})
            
            if document:
                document["_id"] = str(document["_id"])
                logger.info(f"Retrieved document metadata for ID: {document_id}")
            
            return document
            
        except Exception as e:
            logger.error("Failed to get document metadata", error=str(e))
            raise
    
    async def list_documents(
        self,
        namespace: Optional[str] = None,
        limit: int = 50,
        skip: int = 0
    ) -> List[Dict[str, Any]]:
        """List documents with optional filtering."""
        try:
            query = {}
            if namespace:
                query["namespace"] = namespace
            
            cursor = self.documents.find(query).skip(skip).limit(limit).sort("created_at", -1)
            documents = await cursor.to_list(length=limit)
            
            # Convert ObjectId to string
            for doc in documents:
                doc["_id"] = str(doc["_id"])
            
            logger.info(f"Retrieved {len(documents)} documents")
            return documents
            
        except Exception as e:
            logger.error("Failed to list documents", error=str(e))
            raise
    
    async def delete_document(self, document_id: str) -> bool:
        """Delete document metadata."""
        try:
            from bson import ObjectId
            
            result = await self.documents.delete_one({"_id": ObjectId(document_id)})
            
            logger.info(f"Deleted document metadata for ID: {document_id}")
            return result.deleted_count > 0
            
        except Exception as e:
            logger.error("Failed to delete document metadata", error=str(e))
            raise
    
    async def update_document_status(
        self, 
        document_id: str, 
        status: str,
        error_message: Optional[str] = None
    ) -> bool:
        """Update document processing status."""
        try:
            from bson import ObjectId
            
            update_data = {
                "status": status,
                "updated_at": datetime.utcnow()
            }
            
            if error_message:
                update_data["error_message"] = error_message
            
            result = await self.documents.update_one(
                {"_id": ObjectId(document_id)},
                {"$set": update_data}
            )
            
            logger.info(f"Updated document status for ID: {document_id} to {status}")
            return result.modified_count > 0
            
        except Exception as e:
            logger.error("Failed to update document status", error=str(e))
            raise
    
    async def save_chunks(
        self, 
        document_id: str, 
        chunks: List[Dict[str, Any]]
    ) -> List[str]:
        """Save document chunks metadata."""
        try:
            chunk_docs = []
            for i, chunk in enumerate(chunks):
                chunk_doc = {
                    "document_id": document_id,
                    "chunk_index": i,
                    "content": chunk.get("content", ""),
                    "metadata": chunk.get("metadata", {}),
                    "created_at": datetime.utcnow()
                }
                chunk_docs.append(chunk_doc)
            
            result = await self.chunks.insert_many(chunk_docs)
            
            chunk_ids = [str(id) for id in result.inserted_ids]
            logger.info(f"Saved {len(chunk_ids)} chunks for document {document_id}")
            return chunk_ids
            
        except Exception as e:
            logger.error("Failed to save document chunks", error=str(e))
            raise
    
    async def get_document_chunks(self, document_id: str) -> List[Dict[str, Any]]:
        """Get chunks for a document."""
        try:
            cursor = self.chunks.find({"document_id": document_id}).sort("chunk_index", 1)
            chunks = await cursor.to_list(length=None)
            
            # Convert ObjectId to string
            for chunk in chunks:
                chunk["_id"] = str(chunk["_id"])
            
            logger.info(f"Retrieved {len(chunks)} chunks for document {document_id}")
            return chunks
            
        except Exception as e:
            logger.error("Failed to get document chunks", error=str(e))
            raise
    
    async def get_document_stats(self) -> Dict[str, Any]:
        """Get document statistics."""
        try:
            total_docs = await self.documents.count_documents({})
            total_chunks = await self.chunks.count_documents({})
            
            # Get documents by namespace
            pipeline = [
                {"$group": {"_id": "$namespace", "count": {"$sum": 1}}}
            ]
            namespace_stats = await self.documents.aggregate(pipeline).to_list(length=None)
            
            # Get documents by status
            status_pipeline = [
                {"$group": {"_id": "$status", "count": {"$sum": 1}}}
            ]
            status_stats = await self.documents.aggregate(status_pipeline).to_list(length=None)
            
            stats = {
                "total_documents": total_docs,
                "total_chunks": total_chunks,
                "by_namespace": {stat["_id"]: stat["count"] for stat in namespace_stats},
                "by_status": {stat["_id"]: stat["count"] for stat in status_stats}
            }
            
            logger.info("Retrieved document statistics")
            return stats
            
        except Exception as e:
            logger.error("Failed to get document statistics", error=str(e))
            raise


def get_document_repository(db = Depends(get_database)) -> DocumentRepository:
    """Dependency to get document repository."""
    return DocumentRepository(db)