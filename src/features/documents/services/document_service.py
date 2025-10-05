"""Document service implementation."""

import tempfile
import os
from typing import List, Dict, Any
from datetime import datetime
from fastapi import Depends, HTTPException, UploadFile, BackgroundTasks
from shared.utils.id_generator import generate_document_id

from features.documents.api.schemas import (
    DocumentInfo, DocumentUploadRequest, DocumentProcessRequest,
    DocumentSearchRequest, DocumentSearchResponse, DocumentDeleteResponse,
    EntityExtractionResponse
)
from features.documents.repositories.document_repository import DocumentRepository, get_document_repository
from features.rag.repositories.vector_repository import VectorRepository, get_vector_repository
from core.config.logging_config import LoggerAdapter
from core.cache.redis_client import redis_cache
from services.docling.processor import document_processor
from services.document_intelligence.advanced_processor import advanced_document_processor

logger = LoggerAdapter(__name__)


class DocumentService:
    """Service for document operations."""
    
    def __init__(
        self, 
        document_repo: DocumentRepository,
        vector_repo: VectorRepository
    ):
        """Initialize document service."""
        self.document_repo = document_repo
        self.vector_repo = vector_repo
    
    async def upload_document(
        self,
        file: UploadFile,
        upload_request: DocumentUploadRequest,
        background_tasks: BackgroundTasks
    ) -> DocumentInfo:
        """Upload a document for processing."""
        try:
            # Validate file type
            file_extension = file.filename.split(".")[-1].lower()
            supported_formats = ["pdf", "txt", "docx", "xlsx", "pptx", "md", "html", "csv", "json"]
            
            if file_extension not in supported_formats:
                raise HTTPException(
                    status_code=400,
                    detail=f"Unsupported file type: {file_extension}. Supported formats: {', '.join(supported_formats)}"
                )
            
            # Generate document ID using nanoid
            document_id = generate_document_id("doc")
            
            # Save uploaded file
            with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_extension}") as tmp:
                content = await file.read()
                tmp.write(content)
                tmp_path = tmp.name
            
            # Store document info in cache
            doc_info = {
                "document_id": document_id,
                "filename": file.filename,
                "upload_time": datetime.now().isoformat(),
                "status": "uploaded",
                "file_path": tmp_path,
                "namespace": upload_request.namespace
            }
            
            await redis_cache.set(f"document:{document_id}", doc_info, ttl=86400)  # 24 hours
            
            # Save to repository
            await self.document_repo.save_document({
                "document_id": document_id,
                "filename": file.filename,
                "namespace": upload_request.namespace,
                "status": "uploaded",
                "file_extension": file_extension,
                "chunk_size": upload_request.chunk_size,
                "chunk_overlap": upload_request.chunk_overlap
            })
            
            if upload_request.process_immediately:
                # Process in background
                background_tasks.add_task(
                    self._process_document_background,
                    document_id,
                    tmp_path,
                    upload_request.namespace,
                    upload_request.chunk_size,
                    upload_request.chunk_overlap
                )
                doc_info["status"] = "processing"
            
            logger.info(f"Document uploaded: {document_id}")
            
            return DocumentInfo(**doc_info)
            
        except Exception as e:
            logger.error("Failed to upload document", error=str(e))
            raise HTTPException(status_code=500, detail=str(e))
    
    async def _process_document_background(
        self,
        document_id: str,
        file_path: str,
        namespace: str,
        chunk_size: int,
        chunk_overlap: int
    ):
        """Background task to process document."""
        try:
            # Update status
            await self.document_repo.update_document_status(document_id, "processing")
            
            # Process document
            chunks = await document_processor.process_document(
                file_path,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            )
            
            # Add metadata
            for chunk in chunks:
                chunk["metadata"]["document_id"] = document_id
                chunk["metadata"]["namespace"] = namespace
            
            # Add to vector store
            vector_ids = await self.vector_repo.add_documents(chunks, namespace=namespace)
            
            # Update document info
            doc_info = await redis_cache.get(f"document:{document_id}")
            if doc_info:
                doc_info["status"] = "processed"
                doc_info["chunks"] = len(chunks)
                doc_info["vector_ids"] = vector_ids
                await redis_cache.set(f"document:{document_id}", doc_info, ttl=86400)
            
            # Update repository
            await self.document_repo.update_document_status(document_id, "processed")
            await self.document_repo.save_chunks(document_id, chunks)
            
            logger.info(f"Document processed: {document_id}, {len(chunks)} chunks")
            
        except Exception as e:
            logger.error(f"Failed to process document {document_id}", error=str(e))
            # Update status to failed
            await self.document_repo.update_document_status(document_id, "failed", str(e))
            
            doc_info = await redis_cache.get(f"document:{document_id}")
            if doc_info:
                doc_info["status"] = "failed"
                doc_info["error"] = str(e)
                await redis_cache.set(f"document:{document_id}", doc_info, ttl=86400)
        finally:
            # Clean up temp file
            if os.path.exists(file_path):
                os.unlink(file_path)
    
    async def get_document_status(self, document_id: str) -> DocumentInfo:
        """Get document processing status."""
        try:
            doc_info = await redis_cache.get(f"document:{document_id}")
            
            if not doc_info:
                # Try to get from repository
                doc_data = await self.document_repo.get_document(document_id)
                if not doc_data:
                    return None
                
                doc_info = {
                    "document_id": doc_data["document_id"],
                    "filename": doc_data["filename"],
                    "upload_time": doc_data["created_at"].isoformat(),
                    "status": doc_data["status"],
                    "namespace": doc_data.get("namespace")
                }
            
            return DocumentInfo(**doc_info)
            
        except Exception as e:
            logger.error("Failed to get document status", error=str(e))
            raise HTTPException(status_code=500, detail=str(e))
    
    async def list_documents(self, namespace: str = "default", limit: int = 100) -> List[DocumentInfo]:
        """List all documents in a namespace."""
        try:
            documents = await self.document_repo.list_documents(namespace, limit)
            
            doc_infos = []
            for doc in documents:
                doc_info = DocumentInfo(
                    document_id=doc["document_id"],
                    filename=doc["filename"],
                    upload_time=doc["created_at"].isoformat(),
                    status=doc["status"],
                    namespace=doc.get("namespace"),
                    chunks=doc.get("total_chunks")
                )
                doc_infos.append(doc_info)
            
            return doc_infos
            
        except Exception as e:
            logger.error("Failed to list documents", error=str(e))
            raise HTTPException(status_code=500, detail=str(e))
    
    async def process_document(
        self, 
        process_request: DocumentProcessRequest,
        background_tasks: BackgroundTasks
    ) -> DocumentInfo:
        """Process an already uploaded document with specific settings."""
        try:
            doc_info = await redis_cache.get(f"document:{process_request.document_id}")
            
            if not doc_info:
                raise HTTPException(status_code=404, detail="Document not found")
            
            if "file_path" not in doc_info or not os.path.exists(doc_info["file_path"]):
                raise HTTPException(status_code=400, detail="Document file not found")
            
            # Update status
            doc_info["status"] = "reprocessing"
            await redis_cache.set(f"document:{process_request.document_id}", doc_info, ttl=86400)
            
            # Process in background
            background_tasks.add_task(
                self._process_document_background,
                process_request.document_id,
                doc_info["file_path"],
                doc_info.get("namespace", "default"),
                process_request.chunk_size,
                process_request.chunk_overlap
            )
            
            return DocumentInfo(**doc_info)
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error("Failed to process document", error=str(e))
            raise HTTPException(status_code=500, detail=str(e))
    
    async def search_documents(self, search_request: DocumentSearchRequest) -> DocumentSearchResponse:
        """Search within documents."""
        try:
            # Build filter for specific documents
            filter_dict = None
            if search_request.document_ids:
                filter_dict = {"document_id": {"$in": search_request.document_ids}}
            
            # Perform search based on type
            if search_request.search_type == "semantic":
                results = await self.vector_repo.similarity_search(
                    query=search_request.query,
                    k=search_request.limit,
                    filter=filter_dict
                )
            else:
                # For keyword search, we'd implement a different approach
                # For now, fallback to semantic search
                results = await self.vector_repo.similarity_search(
                    query=search_request.query,
                    k=search_request.limit,
                    filter=filter_dict
                )
            
            return DocumentSearchResponse(
                query=search_request.query,
                results=results,
                count=len(results),
                search_type=search_request.search_type
            )
            
        except Exception as e:
            logger.error("Failed to search documents", error=str(e))
            raise HTTPException(status_code=500, detail=str(e))
    
    async def delete_document(self, document_id: str) -> DocumentDeleteResponse:
        """Delete a document and its vectors."""
        try:
            doc_info = await redis_cache.get(f"document:{document_id}")
            
            if not doc_info:
                raise HTTPException(status_code=404, detail="Document not found")
            
            # Delete vectors if they exist
            if "vector_ids" in doc_info:
                namespace = doc_info.get("namespace", "default")
                await self.vector_repo.delete_by_ids(
                    doc_info["vector_ids"],
                    namespace=namespace
                )
            
            # Delete from cache
            await redis_cache.delete(f"document:{document_id}")
            
            # Delete from repository
            await self.document_repo.delete_document(document_id)
            
            # Delete file if it exists
            if "file_path" in doc_info and os.path.exists(doc_info["file_path"]):
                os.unlink(doc_info["file_path"])
            
            logger.info(f"Document deleted: {document_id}")
            
            return DocumentDeleteResponse(
                message=f"Document {document_id} deleted successfully",
                document_id=document_id
            )
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error("Failed to delete document", error=str(e))
            raise HTTPException(status_code=500, detail=str(e))
    
    async def extract_entities(self, document_id: str) -> EntityExtractionResponse:
        """Extract entities from a document."""
        try:
            doc_info = await redis_cache.get(f"document:{document_id}")
            
            if not doc_info:
                raise HTTPException(status_code=404, detail="Document not found")
            
            if "file_path" not in doc_info or not os.path.exists(doc_info["file_path"]):
                raise HTTPException(status_code=400, detail="Document file not found")
            
            # Use advanced processor for entity extraction
            entities = await advanced_document_processor.extract_entities(doc_info["file_path"])
            
            return EntityExtractionResponse(
                document_id=document_id,
                entities=entities
            )
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error("Failed to extract entities", error=str(e))
            raise HTTPException(status_code=500, detail=str(e))


def get_document_service(
    document_repo: DocumentRepository = Depends(get_document_repository),
    vector_repo: VectorRepository = Depends(get_vector_repository)
) -> DocumentService:
    """Dependency to get document service."""
    return DocumentService(document_repo, vector_repo)