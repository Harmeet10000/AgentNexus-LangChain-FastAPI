"""RAG service implementation."""

import tempfile
import os
from typing import Dict, Any
from fastapi import Depends, HTTPException, UploadFile

from features.rag.api.schemas import (
    RAGQuery, RAGResponse, DocumentUploadResponse, 
    VectorSearchResponse, VectorStatsResponse
)
from features.rag.repositories.vector_repository import VectorRepository, get_vector_repository
from features.rag.repositories.document_repository import DocumentRepository, get_document_repository
from core.config.logging_config import LoggerAdapter
from services.docling.processor import document_processor
from graphs.workflows.rag_workflow import rag_workflow

logger = LoggerAdapter(__name__)


class RAGService:
    """Service for RAG operations."""
    
    def __init__(
        self, 
        vector_repo: VectorRepository,
        document_repo: DocumentRepository
    ):
        """Initialize RAG service."""
        self.vector_repo = vector_repo
        self.document_repo = document_repo
    
    async def upload_document(
        self,
        file: UploadFile,
        namespace: str = "default",
        chunk_size: int = 1000,
        chunk_overlap: int = 200
    ) -> DocumentUploadResponse:
        """Upload and process document for RAG."""
        try:
            # Validate file type
            file_extension = file.filename.split(".")[-1].lower()
            if file_extension not in ["pdf", "txt", "docx", "md", "html", "csv", "xlsx"]:
                raise HTTPException(
                    status_code=400,
                    detail=f"Unsupported file type: {file_extension}"
                )
            
            # Save uploaded file
            with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_extension}") as tmp:
                content = await file.read()
                tmp.write(content)
                tmp_path = tmp.name
            
            try:
                # Process document
                chunks = await document_processor.process_document(
                    tmp_path,
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap
                )
                
                # Add document metadata
                for chunk in chunks:
                    chunk["metadata"]["source"] = file.filename
                    chunk["metadata"]["namespace"] = namespace
                
                # Save document metadata
                document_id = await self.document_repo.save_document({
                    "filename": file.filename,
                    "namespace": namespace,
                    "chunk_size": chunk_size,
                    "chunk_overlap": chunk_overlap,
                    "total_chunks": len(chunks),
                    "file_extension": file_extension
                })
                
                # Add to vector store
                vector_ids = await self.vector_repo.add_documents(
                    chunks,
                    namespace=namespace
                )
                
                logger.info(f"Uploaded document: {file.filename}, {len(chunks)} chunks")
                
                return DocumentUploadResponse(
                    message="Document processed and indexed successfully",
                    document_id=document_id,
                    chunks=len(chunks),
                    vector_ids=vector_ids,
                    namespace=namespace
                )
                
            finally:
                # Clean up temp file
                os.unlink(tmp_path)
                
        except Exception as e:
            logger.error("Failed to upload document", error=str(e))
            raise HTTPException(status_code=500, detail=str(e))
    
    async def query_rag(self, query: RAGQuery) -> RAGResponse:
        """Query RAG system."""
        try:
            if query.use_workflow:
                # Use LangGraph workflow
                result = await rag_workflow.run(
                    query=query.query,
                    metadata={
                        "namespace": query.namespace,
                        "top_k": query.top_k,
                        "score_threshold": query.score_threshold
                    }
                )
                
                return RAGResponse(
                    answer=result["answer"],
                    sources=result["sources"],
                    error=result.get("error")
                )
            else:
                # Direct vector search
                results = await self.vector_repo.similarity_search_with_relevance_scores(
                    query=query.query,
                    k=query.top_k,
                    namespace=query.namespace,
                    score_threshold=query.score_threshold
                )
                
                if not results:
                    return RAGResponse(
                        answer="No relevant documents found.",
                        sources=[],
                        error="No matches found"
                    )
                
                # Simple answer generation from top result
                answer = f"Based on the search, here's what I found:\n\n{results[0]['content'][:500]}..."
                
                return RAGResponse(
                    answer=answer,
                    sources=results,
                    confidence=results[0]["score"] if results else 0.0
                )
                
        except Exception as e:
            logger.error("Failed to query RAG", error=str(e))
            raise HTTPException(status_code=500, detail=str(e))
    
    async def vector_search(
        self, 
        query: str, 
        namespace: str = "default", 
        k: int = 5
    ) -> VectorSearchResponse:
        """Perform vector similarity search."""
        try:
            results = await self.vector_repo.similarity_search(
                query=query,
                k=k,
                namespace=namespace
            )
            
            return VectorSearchResponse(
                query=query,
                results=results,
                count=len(results)
            )
            
        except Exception as e:
            logger.error("Failed to perform vector search", error=str(e))
            raise HTTPException(status_code=500, detail=str(e))
    
    async def delete_vectors(self, ids: list, namespace: str = "default") -> Dict[str, Any]:
        """Delete vectors by IDs."""
        try:
            success = await self.vector_repo.delete_by_ids(ids, namespace)
            
            if success:
                return {"success": True, "deleted_count": len(ids)}
            else:
                raise HTTPException(status_code=500, detail="Failed to delete vectors")
                
        except Exception as e:
            logger.error("Failed to delete vectors", error=str(e))
            raise HTTPException(status_code=500, detail=str(e))
    
    async def get_vector_stats(self) -> VectorStatsResponse:
        """Get vector store statistics."""
        try:
            stats = await self.vector_repo.get_index_stats()
            
            return VectorStatsResponse(
                total_vectors=stats.get("total_vector_count", 0),
                namespaces=stats.get("namespaces", {}),
                dimension=stats.get("dimension", 0)
            )
            
        except Exception as e:
            logger.error("Failed to get vector stats", error=str(e))
            raise HTTPException(status_code=500, detail=str(e))


def get_rag_service(
    vector_repo: VectorRepository = Depends(get_vector_repository),
    document_repo: DocumentRepository = Depends(get_document_repository)
) -> RAGService:
    """Dependency to get RAG service."""
    return RAGService(vector_repo, document_repo)