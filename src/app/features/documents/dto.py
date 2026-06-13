"""DTOs for unified document ingestion and retrieval."""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict, Field

from app.features.search.constants import DEFAULT_PAGE_SIZE, DEFAULT_RAG_TOKEN_BUDGET, MAX_PAGE_SIZE

if TYPE_CHECKING:
    from typing import Literal

_STRICT_CONFIG = ConfigDict(extra="forbid")
_READ_MODEL_CONFIG = ConfigDict(extra="forbid", frozen=True, from_attributes=True)


class QualityWarningDTO(BaseModel):
    model_config = _READ_MODEL_CONFIG

    stage: str
    code: str
    message: str
    severity: Literal["info", "warning", "error"]


class DocumentUploadResponse(BaseModel):
    model_config = _READ_MODEL_CONFIG

    doc_id: str
    status: str
    task_id: str | None = None
    duplicate: bool = False
    object_uri: str | None = None
    document_kind: str | None = None
    chunk_count: int | None = None
    warning_count: int = 0
    warnings: list[QualityWarningDTO] = Field(default_factory=list)


class DocumentStatusResponse(BaseModel):
    model_config = _READ_MODEL_CONFIG

    doc_id: str
    status: str
    object_uri: str
    title: str
    document_kind: str
    chunk_count: int
    verified_chunk_count: int
    warning_count: int
    warnings: list[QualityWarningDTO] = Field(default_factory=list)


class SearchMetadataFilter(BaseModel):
    model_config = _STRICT_CONFIG

    metadata_: dict[str, object] = Field(default_factory=dict)
    document_ids: list[str] = Field(default_factory=list)
    document_kind: str | None = None
    jurisdiction: str | None = None
    contract_type: str | None = None
    parties: list[str] = Field(default_factory=list)
    clause_type: str | None = None
    require_graphiti_verified: bool = False


class UnifiedSearchRequest(BaseModel):
    model_config = _STRICT_CONFIG

    query: str = Field(min_length=1)
    limit: int = Field(default=DEFAULT_PAGE_SIZE, ge=1, le=MAX_PAGE_SIZE)
    candidate_limit: int = Field(default=50, ge=1, le=200)
    metadata_filter: SearchMetadataFilter = Field(default_factory=SearchMetadataFilter)
    bypass_cache: bool = False


class DocumentSearchResultItem(BaseModel):
    model_config = _READ_MODEL_CONFIG

    chunk_id: str
    document_id: str
    title: str
    content: str
    chunk_index: int
    chunk_kind: str
    clause_type: str | None = None
    chunk_metadata: dict[str, object]
    quality_warnings: list[QualityWarningDTO] = Field(default_factory=list)
    graphiti_verified: bool = False
    score: float
    rank: int


class UnifiedSearchResponse(BaseModel):
    model_config = _READ_MODEL_CONFIG

    items: list[DocumentSearchResultItem]
    cache_hit: bool = False


class RagContextSectionResponse(BaseModel):
    model_config = _READ_MODEL_CONFIG

    document_id: str
    title: str
    content: str
    chunk_indices: list[int]
    chunk_metadata: dict[str, object]


class UnifiedRagRequest(UnifiedSearchRequest):
    max_tokens: int = Field(default=DEFAULT_RAG_TOKEN_BUDGET, ge=1, le=20_000)


class UnifiedRagResponse(BaseModel):
    model_config = _READ_MODEL_CONFIG

    items: list[DocumentSearchResultItem]
    context: list[RagContextSectionResponse]
    cache_hit: bool = False


class UnifiedAskRequest(BaseModel):
    model_config = _STRICT_CONFIG

    query: str = Field(min_length=1)
    doc_ids_filter: list[str] = Field(default_factory=list)
    jurisdiction: str | None = None
    contract_type: str | None = None
    clause_type: str | None = None
    bypass_cache: bool = False


class LegalCitationResponse(BaseModel):
    model_config = _READ_MODEL_CONFIG

    chunk_id: str
    clause_type: str
    claim: str


class UnifiedAskResponse(BaseModel):
    model_config = _READ_MODEL_CONFIG

    answer: str
    citations: list[LegalCitationResponse]
    confidence: Literal["high", "medium", "uncertain"]
    warnings: list[QualityWarningDTO] = Field(default_factory=list)
    cache_hit: bool = False
