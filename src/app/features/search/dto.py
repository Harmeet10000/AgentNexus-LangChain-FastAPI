"""DTOs for the search feature."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from .constants import (
    DEFAULT_PAGE_SIZE,
    DEFAULT_RAG_TOKEN_BUDGET,
    HYBRID_CANDIDATE_LIMIT,
    MAX_PAGE_SIZE,
)

_STRICT_CONFIG = ConfigDict(extra="forbid")
_READ_MODEL_CONFIG = ConfigDict(extra="forbid", frozen=True, from_attributes=True)


class SearchMetadataFilter(BaseModel):
    """JSONB containment filter for search chunks."""

    model_config = _STRICT_CONFIG

    chunk_metadata: dict[str, object] = Field(default_factory=dict)


class SearchIngestRequest(BaseModel):
    """Request body for async search ingestion."""

    model_config = _STRICT_CONFIG

    title: str = Field(min_length=1, max_length=500)
    content: str = Field(min_length=1)
    source_uri: str | None = Field(default=None, max_length=2048)
    doc_metadata: dict[str, object] = Field(default_factory=dict)


class SearchIngestResponse(BaseModel):
    """Queued or duplicate-ingest response."""

    model_config = _READ_MODEL_CONFIG

    document_id: str
    task_id: str | None = None
    status: str
    duplicate: bool = False


class SearchTaskStatusResponse(BaseModel):
    """Normalized ingestion task status payload."""

    model_config = _READ_MODEL_CONFIG

    task_id: str
    status: str
    document_id: str | None = None
    result: dict[str, object] | None = None
    error: str | None = None


class HybridSearchRequest(BaseModel):
    """Hybrid retrieval request."""

    model_config = _STRICT_CONFIG

    query: str = Field(min_length=1)
    limit: int = Field(default=DEFAULT_PAGE_SIZE, ge=1, le=MAX_PAGE_SIZE)
    candidate_limit: int = Field(default=HYBRID_CANDIDATE_LIMIT, ge=1, le=200)
    metadata_filter: SearchMetadataFilter = Field(default_factory=SearchMetadataFilter)
    bypass_cache: bool = False


class SearchResultItem(BaseModel):
    """Ranked hybrid search hit."""

    model_config = _READ_MODEL_CONFIG

    chunk_id: str
    document_id: str
    title: str
    content: str
    chunk_index: int
    chunk_metadata: dict[str, object]
    score: float
    rank: int


class SearchResponse(BaseModel):
    """Hybrid search response."""

    model_config = _READ_MODEL_CONFIG

    items: list[SearchResultItem]
    cache_hit: bool = False


class RagContextSectionResponse(BaseModel):
    """Ordered context section for RAG consumers."""

    model_config = _READ_MODEL_CONFIG

    document_id: str
    title: str
    content: str
    chunk_indices: list[int]
    chunk_metadata: dict[str, object]


class RagSearchRequest(HybridSearchRequest):
    """Hybrid retrieval plus RAG context assembly request."""

    max_tokens: int = Field(default=DEFAULT_RAG_TOKEN_BUDGET, ge=1, le=20_000)


class RagSearchResponse(BaseModel):
    """Hybrid hits plus assembled RAG context."""

    model_config = _READ_MODEL_CONFIG

    items: list[SearchResultItem]
    context: list[RagContextSectionResponse]
    cache_hit: bool = False
