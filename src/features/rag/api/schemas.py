"""RAG API schemas."""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class RAGQuery(BaseModel):
    """RAG query request model."""
    query: str = Field(..., description="Query to search for")
    namespace: Optional[str] = Field(default="default", description="Pinecone namespace")
    top_k: int = Field(default=5, ge=1, le=20)
    score_threshold: Optional[float] = Field(default=0.5, ge=0.0, le=1.0)
    use_workflow: bool = Field(default=True, description="Use LangGraph workflow")


class RAGResponse(BaseModel):
    """RAG response model."""
    answer: str
    sources: List[Dict[str, Any]]
    confidence: Optional[float] = None
    error: Optional[str] = None


class DocumentUploadRequest(BaseModel):
    """Document upload request model."""
    namespace: str = Field(default="default")
    chunk_size: int = Field(default=1000)
    chunk_overlap: int = Field(default=200)


class DocumentUploadResponse(BaseModel):
    """Document upload response model."""
    message: str
    document_id: str
    chunks: int
    vector_ids: List[str]
    namespace: str


class VectorSearchRequest(BaseModel):
    """Vector search request model."""
    query: str
    namespace: str = Field(default="default")
    k: int = Field(default=5, ge=1, le=20)


class VectorSearchResponse(BaseModel):
    """Vector search response model."""
    query: str
    results: List[Dict[str, Any]]
    count: int


class VectorDeleteRequest(BaseModel):
    """Vector delete request model."""
    ids: List[str]
    namespace: str = Field(default="default")


class VectorStatsResponse(BaseModel):
    """Vector store statistics response model."""
    total_vectors: int
    namespaces: Dict[str, Any]
    dimension: int