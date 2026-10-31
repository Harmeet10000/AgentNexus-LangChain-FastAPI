"""DTOs for the site-search feature (LegalAsk* deliberately not ported).

The legacy ``/search/ask`` endpoint is superseded by the documents feature's
grounded ask path; its DTOs are not carried over.

Numeric bounds are protocol constants (wire contract), pinned here; the
*tunables* (page-size defaults, candidate limit, token budget) default to
``None`` and are resolved by the service from ``config.settings`` SEARCH_*
fields, so configuration — not module constants — drives behavior.

Size bounds are DoS fences, not style: every accepted body is written to an
outbox row, carried on the broker message, and spent on provider embedding
calls per chunk. An unbounded body converts one anonymous POST into storage,
queue, and metered-API spend, so both the content and the metadata are capped
before validation succeeds.
"""

from __future__ import annotations

from typing import Any

import orjson
from pydantic import BaseModel, ConfigDict, Field, model_validator

_STRICT_CONFIG = ConfigDict(extra="forbid")
_READ_MODEL_CONFIG = ConfigDict(extra="forbid", frozen=True, from_attributes=True)

#: Maximum accepted ``content`` length (1 MB). The legacy feature accepted
#: unbounded bodies; the chunk pipeline spends O(content) provider calls.
MAX_INGEST_CONTENT_CHARS = 1_000_000

#: Maximum serialized ``doc_metadata`` size. The metadata rides the broker
#: message and lands in a JSONB column on every chunk-less write; 8 KiB covers
#: any realistic site-page taxonomy without letting a hostile body inflate the
#: outbox row.
MAX_INGEST_METADATA_BYTES = 8_192


class SearchMetadataFilter(BaseModel):
    """JSONB containment filter for search chunks."""

    model_config = _STRICT_CONFIG

    chunk_metadata: dict[str, object] = Field(default_factory=dict)


class SearchIngestRequest(BaseModel):
    """Request body for async search ingestion."""

    model_config = _STRICT_CONFIG

    title: str = Field(min_length=1, max_length=500)
    content: str = Field(min_length=1, max_length=MAX_INGEST_CONTENT_CHARS)
    source_uri: str | None = Field(default=None, max_length=2048)
    doc_metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _bound_doc_metadata(self) -> SearchIngestRequest:
        """Cap the serialized metadata size (deep values included).

        A pathological nesting depth makes even *measuring* the size explode
        (recursion), which is itself a rejection: any structure the serializer
        cannot walk within limits is not a site-page taxonomy.
        """
        try:
            serialized_size = len(orjson.dumps(self.doc_metadata))
        except (TypeError, RecursionError) as exc:
            unserializable = "doc_metadata is not serializable within limits"
            raise ValueError(unserializable) from exc
        if serialized_size > MAX_INGEST_METADATA_BYTES:
            oversized = f"doc_metadata exceeds {MAX_INGEST_METADATA_BYTES} serialized bytes"
            raise ValueError(oversized)
        return self


class SearchIngestResponse(BaseModel):
    """Queued or duplicate-ingest response."""

    model_config = _READ_MODEL_CONFIG

    document_id: str
    # The pollable task identity. Ingest stages the job through the
    # transactional job bus (atomic with the document row), so this is the
    # outbox event id; Celery assigns its own task id only at relay dispatch.
    task_id: str | None = None
    status: str
    duplicate: bool = False


class SearchTaskStatusResponse(BaseModel):
    """Normalized ingestion task status payload.

    ``status`` is the Celery result state, except that a staged-but-undispatched
    task (known via the enqueue-time ownership record) reports ``QUEUED`` —
    Celery's ``PENDING`` means "never seen this task", not "waiting".
    """

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
    limit: int | None = Field(default=None, ge=1, le=100)
    candidate_limit: int | None = Field(default=None, ge=1, le=200)
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

    max_tokens: int | None = Field(default=None, ge=1, le=20_000)


class RagSearchResponse(BaseModel):
    """Hybrid hits plus assembled RAG context."""

    model_config = _READ_MODEL_CONFIG

    items: list[SearchResultItem]
    context: list[RagContextSectionResponse]
    cache_hit: bool = False
