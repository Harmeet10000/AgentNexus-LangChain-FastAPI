"""DTOs for unified document ingestion and retrieval."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from datetime import datetime
from typing import Literal

from graphiti_core.graphiti import Graphiti
from langchain_core.language_models import (
    BaseChatModel,
)
from pydantic import BaseModel, ConfigDict, Field

from app.extraction.langextract.service import AsyncExtractionService
from app.features.documents.adapters.persistence.postgres.document_repository import (
    DocumentRepository,
)
from app.features.documents.domain.constants import (
    DEFAULT_PAGE_SIZE,
    DEFAULT_RAG_TOKEN_BUDGET,
    MAX_PAGE_SIZE,
)
from app.features.documents.domain.fidelity import (
    GraphRelationshipsState,
    IngestionFidelityMetrics,
)
from app.utils.storage import (
    StorageService,
)

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
    document_version: int | None = None
    pipeline_version: str | None = None
    graph_indexed_at: datetime | None = None
    warnings: list[QualityWarningDTO] = Field(default_factory=list)
    # Typed ingestion truth.  ``None`` means the record predates the metric or
    # its persisted payload was invalid — an honest absence, not a default.
    graph_relationships: GraphRelationshipsState | None = None
    fidelity: IngestionFidelityMetrics | None = None


class SearchMetadataFilter(BaseModel):
    model_config = _STRICT_CONFIG

    metadata_: dict[str, object] = Field(default_factory=dict)
    document_ids: list[str] = Field(default_factory=list)
    document_kind: str | None = None
    jurisdiction: str | None = None
    contract_type: str | None = None
    contract_index: int | None = Field(default=None, ge=0)
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


# Evidence is deliberately a separate contract from the legacy search DTOs.
# ``UnifiedSearchResponse`` is used by the public search endpoint and contains
# hydrated chunk text for backwards compatibility.  Agent-facing evidence must
# instead expose a bounded preview plus a server-issued handle that is resolved
# through an authorization check at read time.
EvidenceStatus = Literal[
    "ok",
    "no_evidence",
    "index_not_ready",
    "ingestion_failed",
    "failed",
    "unauthorized",
    "invalid_handle",
    "stale_version",
    "expired",
    "transient_failure",
    "timeout",
    "budget_exhausted",
]
IngestionState = Literal[
    "ready",
    "completed_with_warnings",
    "index_not_ready",
    "failed",
    "unauthorized",
]
EvidencePolicy = Literal["mechanically_verified", "graph_indexed_only"]


class EvidenceRef(BaseModel):
    """A versioned, scoped reference to an exact persisted chunk."""

    model_config = _READ_MODEL_CONFIG

    evidence_id: str
    chunk_id: str
    document_id: str
    document_version: int
    # ``None`` means legacy or boundary unknown; it is not contract 0.
    contract_index: int | None = None
    pipeline_version: str | None = None
    page_no: int
    graph_indexed: bool
    graph_indexed_at: datetime | None = None
    retrieval_method: str
    score: float
    rank: int
    run_id: str | None = None
    thread_id: str | None = None
    checkpoint_id: str = ""


class EvidenceCoverage(BaseModel):
    """Coverage metadata that prevents an empty result being mistaken for proof."""

    model_config = _READ_MODEL_CONFIG

    documents_requested: int = 0
    documents_ready: int = 0
    documents_with_warnings: int = 0
    documents_failed: int = 0
    documents_not_ready: int = 0
    graph_indexed_documents: int = 0
    graph_unavailable_documents: int = 0


class EvidenceResultEnvelope(BaseModel):
    """Common status/provenance envelope for evidence operations."""

    model_config = _READ_MODEL_CONFIG

    status: EvidenceStatus
    evidence_policy: EvidencePolicy
    coverage: EvidenceCoverage = Field(default_factory=EvidenceCoverage)
    request_id: str | None = None
    message: str | None = None
    warnings: list[QualityWarningDTO] = Field(default_factory=list)


class EvidenceSearchRequest(BaseModel):
    model_config = _STRICT_CONFIG

    query: str = Field(min_length=1)
    limit: int = Field(default=5, ge=1, le=50)
    candidate_limit: int = Field(default=50, ge=1, le=200)
    jurisdiction: str | None = None
    contract_type: str | None = None
    contract_index: int | None = Field(default=None, ge=0)
    clause_type: str | None = None
    document_kind: str | None = None
    require_graph_indexed: bool = False


class EvidenceSearchItem(BaseModel):
    model_config = _READ_MODEL_CONFIG

    evidence: EvidenceRef
    document_title: str
    snippet: str
    chunk_index: int
    contract_index: int | None = None
    chunk_kind: str
    clause_type: str | None = None
    metadata: dict[str, object] = Field(default_factory=dict)


class EvidenceSearchResponse(EvidenceResultEnvelope):
    items: list[EvidenceSearchItem] = Field(default_factory=list)


class EvidenceReadItem(BaseModel):
    model_config = _READ_MODEL_CONFIG

    evidence: EvidenceRef
    document_title: str
    content: str
    preamble: str = ""
    contract_index: int | None = None
    metadata: dict[str, object] = Field(default_factory=dict)


class EvidenceReadResponse(EvidenceResultEnvelope):
    items: list[EvidenceReadItem] = Field(default_factory=list)


class IngestionDocumentState(BaseModel):
    model_config = _READ_MODEL_CONFIG

    document_id: str
    state: IngestionState
    status: str
    chunk_count: int = 0
    warning_count: int = 0
    graph_indexed_chunk_count: int = 0
    document_version: int | None = None
    pipeline_version: str | None = None
    graph_indexed_at: datetime | None = None


class IngestionStateResponse(EvidenceResultEnvelope):
    documents: list[IngestionDocumentState] = Field(default_factory=list)


class DocumentSearchResultItem(BaseModel):
    model_config = _READ_MODEL_CONFIG

    chunk_id: str
    document_id: str
    document_version: int = 1
    contract_index: int | None = None
    pipeline_version: str | None = None
    title: str
    content: str
    chunk_index: int
    chunk_kind: str
    clause_type: str | None = None
    chunk_metadata: dict[str, object]
    quality_warnings: list[QualityWarningDTO] = Field(default_factory=list)
    graphiti_verified: bool = False
    graph_indexed: bool = False
    graph_indexed_at: str | None = None
    score: float
    rank: int


class UnifiedSearchResponse(BaseModel):
    model_config = _READ_MODEL_CONFIG

    items: list[DocumentSearchResultItem]
    cache_hit: bool = False


class RagContextSectionResponse(BaseModel):
    model_config = _READ_MODEL_CONFIG

    document_id: str
    contract_index: int | None = None
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
    contract_index: int | None = Field(default=None, ge=0)
    clause_type: str | None = None
    bypass_cache: bool = False


class LegalCitationResponse(BaseModel):
    model_config = _READ_MODEL_CONFIG

    chunk_id: str
    clause_type: str
    claim: str
    document_id: str | None = None
    document_version: int | None = None
    contract_index: int | None = None
    pipeline_version: str | None = None
    page_no: int | None = None


class UnifiedAskResponse(BaseModel):
    model_config = _READ_MODEL_CONFIG

    answer: str
    citations: list[LegalCitationResponse]
    confidence: Literal["high", "medium", "uncertain"]
    evidence_policy: EvidencePolicy = "mechanically_verified"
    coverage: EvidenceCoverage = Field(default_factory=EvidenceCoverage)
    warnings: list[QualityWarningDTO] = Field(default_factory=list)
    cache_hit: bool = False


class IngestionJob(BaseModel):
    """Identity/data cluster for one document ingestion job (read model)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    document_id: str
    user_id: str
    tenant_id: str = ""
    document_version: int = Field(default=1, ge=1)
    attempt: int = Field(default=0, ge=0)
    task_id: str | None = None
    filename: str
    content_type: str
    object_uri: str


class IngestionRuntime(BaseModel):
    """Runtime services for document ingestion (explicit per-call dependencies)."""

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    object_store: StorageService
    repo: DocumentRepository
    graphiti: Graphiti | None = None
    llm: BaseChatModel
    extraction: AsyncExtractionService | None = None
    graph_writer: object | None = None
    idempotency: object | None = None
    transaction_checkpoint: Callable[[], Awaitable[None]] | None = None
