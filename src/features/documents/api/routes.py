"""Document management API routes."""

from typing import List
from fastapi import APIRouter, UploadFile, File, HTTPException, Depends, Form, BackgroundTasks, Request

from features.documents.api.schemas import (
    DocumentInfo, DocumentUploadRequest, DocumentProcessRequest, 
    DocumentSearchRequest, DocumentSearchResponse, DocumentDeleteResponse,
    EntityExtractionResponse
)
from features.documents.services.document_service import DocumentService, get_document_service
from core.config.logging_config import LoggerAdapter
from shared.schemas.response import http_success, http_error, http_created, http_not_found

logger = LoggerAdapter(__name__)
router = APIRouter(prefix="/documents", tags=["Documents"])


@router.post("/upload", response_model=DocumentInfo)
async def upload_document(
    request: Request,
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    namespace: str = Form(default="default"),
    process_immediately: bool = Form(default=True),
    chunk_size: int = Form(default=1000),
    chunk_overlap: int = Form(default=200),
    service: DocumentService = Depends(get_document_service)
):
    """Upload a document for processing."""
    try:
        upload_request = DocumentUploadRequest(
            namespace=namespace,
            process_immediately=process_immediately,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        
        result = await service.upload_document(
            file=file,
            upload_request=upload_request,
            background_tasks=background_tasks
        )
        
        return http_created(
            request,
            message="Document uploaded successfully",
            data=result
        )
        
    except Exception as e:
        logger.error("Failed to upload document", error=str(e))
        return http_error(request, e, 500)


@router.get("/status/{document_id}", response_model=DocumentInfo)
async def get_document_status(
    request: Request,
    document_id: str,
    service: DocumentService = Depends(get_document_service)
):
    """Get document processing status."""
    try:
        result = await service.get_document_status(document_id)
        
        if not result:
            return http_not_found(request, "Document not found")
        
        return http_success(
            request,
            message="Document status retrieved successfully",
            data=result
        )
        
    except Exception as e:
        logger.error("Failed to get document status", error=str(e))
        return http_error(request, e, 500)


@router.get("/list", response_model=List[DocumentInfo])
async def list_documents(
    request: Request,
    namespace: str = "default",
    limit: int = 100,
    service: DocumentService = Depends(get_document_service)
):
    """List all documents in a namespace."""
    try:
        result = await service.list_documents(namespace, limit)
        
        return http_success(
            request,
            message="Documents listed successfully",
            data=result
        )
        
    except Exception as e:
        logger.error("Failed to list documents", error=str(e))
        return http_error(request, e, 500)


@router.post("/process", response_model=DocumentInfo)
async def process_document(
    request: Request,
    background_tasks: BackgroundTasks,
    process_request: DocumentProcessRequest,
    service: DocumentService = Depends(get_document_service)
):
    """Process an already uploaded document with specific settings."""
    try:
        result = await service.process_document(process_request, background_tasks)
        
        return http_success(
            request,
            message="Document processing started",
            data=result
        )
        
    except Exception as e:
        logger.error("Failed to process document", error=str(e))
        return http_error(request, e, 500)


@router.post("/search", response_model=DocumentSearchResponse)
async def search_documents(
    request: Request,
    search_request: DocumentSearchRequest,
    service: DocumentService = Depends(get_document_service)
):
    """Search within documents."""
    try:
        result = await service.search_documents(search_request)
        
        return http_success(
            request,
            message="Document search completed successfully",
            data=result
        )
        
    except Exception as e:
        logger.error("Failed to search documents", error=str(e))
        return http_error(request, e, 500)


@router.delete("/{document_id}", response_model=DocumentDeleteResponse)
async def delete_document(
    request: Request,
    document_id: str,
    service: DocumentService = Depends(get_document_service)
):
    """Delete a document and its vectors."""
    try:
        result = await service.delete_document(document_id)
        
        return http_success(
            request,
            message=f"Document {document_id} deleted successfully",
            data=result
        )
        
    except Exception as e:
        logger.error("Failed to delete document", error=str(e))
        return http_error(request, e, 500)


@router.post("/extract-entities/{document_id}", response_model=EntityExtractionResponse)
async def extract_entities(
    request: Request,
    document_id: str,
    service: DocumentService = Depends(get_document_service)
):
    """Extract entities from a document."""
    try:
        result = await service.extract_entities(document_id)
        
        return http_success(
            request,
            message="Entities extracted successfully",
            data=result
        )
        
    except Exception as e:
        logger.error("Failed to extract entities", error=str(e))
        return http_error(request, e, 500)