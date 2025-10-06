"""RAG API routes."""

from typing import List
from fastapi import APIRouter, UploadFile, File, HTTPException, Depends, Form, Request

from features.rag.api.schemas import (
    RAGQuery,
    RAGResponse,
    DocumentUploadResponse,
    VectorSearchResponse,
    VectorStatsResponse,
    VectorDeleteRequest,
)
from features.rag.services.rag_service import RAGService, get_rag_service
from core.config.logging_config import LoggerAdapter
from shared.schemas.response import http_success, http_error

logger = LoggerAdapter(__name__)
router = APIRouter(prefix="/rag", tags=["RAG"])


@router.post("/upload", response_model=DocumentUploadResponse)
async def upload_document(
    request: Request,
    file: UploadFile = File(...),
    namespace: str = Form(default="default"),
    chunk_size: int = Form(default=1000),
    chunk_overlap: int = Form(default=200),
    service: RAGService = Depends(get_rag_service),
):
    """Upload and process document for RAG."""
    try:
        result = await service.upload_document(
            file=file,
            namespace=namespace,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

        return http_success(
            request, message="Document processed and indexed successfully", data=result
        )

    except Exception as e:
        logger.error("Failed to upload document", error=str(e))
        return http_error(request, e, 500)


@router.post("/query", response_model=RAGResponse)
async def query_rag(
    request: Request,
    rag_query: RAGQuery,
    service: RAGService = Depends(get_rag_service),
):
    """Query RAG system."""
    try:
        result = await service.query_rag(rag_query)

        return http_success(
            request, message="RAG query processed successfully", data=result
        )

    except Exception as e:
        logger.error("Failed to query RAG", error=str(e))
        return http_error(request, e, 500)


@router.get("/search", response_model=VectorSearchResponse)
async def vector_search(
    request: Request,
    query: str,
    namespace: str = "default",
    k: int = 5,
    service: RAGService = Depends(get_rag_service),
):
    """Perform vector similarity search."""
    try:
        result = await service.vector_search(query, namespace, k)

        return http_success(
            request, message="Vector search completed successfully", data=result
        )

    except Exception as e:
        logger.error("Failed to perform vector search", error=str(e))
        return http_error(request, e, 500)


@router.delete("/vectors")
async def delete_vectors(
    request: Request,
    delete_request: VectorDeleteRequest,
    service: RAGService = Depends(get_rag_service),
):
    """Delete vectors by IDs."""
    try:
        result = await service.delete_vectors(
            delete_request.ids, delete_request.namespace
        )

        return http_success(
            request,
            message=f"Deleted {len(delete_request.ids)} vectors from namespace {delete_request.namespace}",
            data=result,
        )

    except Exception as e:
        logger.error("Failed to delete vectors", error=str(e))
        return http_error(request, e, 500)


@router.get("/stats", response_model=VectorStatsResponse)
async def get_vector_stats(
    request: Request, service: RAGService = Depends(get_rag_service)
):
    """Get vector store statistics."""
    try:
        result = await service.get_vector_stats()

        return http_success(
            request,
            message="Vector store statistics retrieved successfully",
            data=result,
        )

    except Exception as e:
        logger.error("Failed to get vector stats", error=str(e))
        return http_error(request, e, 500)
