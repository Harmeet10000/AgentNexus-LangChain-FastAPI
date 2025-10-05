"""Document management API schemas."""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from datetime import datetime


class DocumentInfo(BaseModel):
    """Document information model."""
    document_id: str
    filename: str
    upload_time: str
    status: str
    chunks: Optional[int] = None
    namespace: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class DocumentUploadRequest(BaseModel):
    """Document upload request model."""
    namespace: str = Field(default="default")
    process_immediately: bool = Field(default=True)
    chunk_size: int = Field(default=1000, ge=100, le=5000)
    chunk_overlap: int = Field(default=200, ge=0, le=500)


class DocumentProcessRequest(BaseModel):
    """Document processing request model."""
    document_id: str
    processing_type: str = Field(default="standard", description="Type of processing: standard, advanced, ocr")
    chunk_size: int = Field(default=1000, ge=100, le=5000)
    chunk_overlap: int = Field(default=200, ge=0, le=500)
    extract_tables: bool = Field(default=True)
    extract_images: bool = Field(default=False)


class DocumentSearchRequest(BaseModel):
    """Document search request model."""
    query: str
    document_ids: Optional[List[str]] = None
    limit: int = Field(default=10, ge=1, le=100)
    search_type: str = Field(default="semantic", description="Search type: semantic, keyword, hybrid")


class DocumentSearchResponse(BaseModel):
    """Document search response model."""
    query: str
    results: List[Dict[str, Any]]
    count: int
    search_type: str


class DocumentDeleteResponse(BaseModel):
    """Document delete response model."""
    message: str
    document_id: str


class EntityExtractionResponse(BaseModel):
    """Entity extraction response model."""
    document_id: str
    entities: Dict[str, Any]