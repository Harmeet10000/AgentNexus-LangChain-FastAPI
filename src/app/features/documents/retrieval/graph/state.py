"""State and schemas for unified-chunk-backed legal retrieval."""

from __future__ import annotations

import operator
from dataclasses import dataclass, field
from typing import (
    Annotated,  # noqa: TC003 — LangGraph resolves state annotations via get_type_hints at build
    Any,  # noqa: TC003 — same: the context schema and state annotations resolve at runtime
    Literal,
)

from langgraph.graph import MessagesState
from pydantic import BaseModel, ConfigDict, Field

# Runtime import: LangGraph evaluates RetrievalState type hints via
# get_type_hints, so ContextSection must be present in this module's
# namespace — a TYPE_CHECKING-only import raises NameError at graph build.
from app.features.documents.retrieval.context_assembly import ContextSection  # noqa: TC001

# Same runtime-import discipline: RetrievalGraphContext is a LangGraph context
# schema, so its annotations resolve too and CountTokens must be present.
from app.retrieval.token_counter import CountTokens  # noqa: TC001


class QueryPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    rewritten_query: str
    sub_queries: list[str] = Field(default_factory=list)
    route: Literal["hybrid_postgres", "graph_neo4j", "both"] = "hybrid_postgres"
    query_type: Literal["exact", "conceptual", "mixed"] = "mixed"
    vector_weight: float = Field(default=0.4, ge=0.0, le=1.0)
    keyword_weight: float = Field(default=0.6, ge=0.0, le=1.0)
    jurisdiction: str | None = None
    contract_type: str | None = None
    bm25_threshold: float | None = None
    exact_phrase: str | None = None
    # Document-ID allowlist populated by the source-identifier node from cheap
    # metadata (jurisdiction, document kind, matter). Empty means unconstrained:
    # when the corpus cannot be narrowed, retrieval proceeds over everything
    # rather than returning nothing.
    allowlist: list[str] = Field(default_factory=list)


class SourceCriteria(BaseModel):
    """Cheap-metadata narrowing extracted from the query, before retrieval."""

    model_config = ConfigDict(extra="forbid")

    jurisdictions: list[str] = Field(default_factory=list)
    document_kinds: list[str] = Field(default_factory=list)
    matters: list[str] = Field(default_factory=list)


class RetrievedChunk(BaseModel):
    model_config = ConfigDict(extra="forbid")

    chunk_id: str
    chunk_text: str
    preamble: str
    clause_type: str
    parent_doc_id: str
    chunk_index: int = 0
    # NULL means legacy ingestion or an unknown contract boundary.
    contract_index: int | None = Field(default=None, ge=0)
    metadata_: dict[str, object]
    custom_metadata: dict[str, object]
    score: float


class Citation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    chunk_id: str
    clause_type: str
    claim: str
    contract_index: int | None = Field(default=None, ge=0)


class ContextGrade(BaseModel):
    model_config = ConfigDict(extra="forbid")

    sufficient: bool
    missing_aspects: list[str] = Field(default_factory=list)
    rewrite_suggestion: str | None = None


class GeneratedAnswer(BaseModel):
    model_config = ConfigDict(extra="forbid")

    answer: str
    citations: list[Citation] = Field(default_factory=list)
    confidence: Literal["high", "medium", "uncertain"] = "uncertain"


class StageUsage(BaseModel):
    """One stage's spend attribution entry (plan, retrieve/fusion, grade, ...).

    Surfaced in the result state so a caller (Agent Saul's budget ledger, G6
    tracing) can charge or trace per stage without importing this package's
    internals — and without this package importing `features/agent_saul`.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    stage: str
    # Number of billable operations the stage performed (LLM calls, retrieval
    # reads, rerank calls). A stage that executed but spent nothing reports 0.
    calls: int = 1


class StageWarning(BaseModel):
    """One quality warning in the response envelope's shape.

    Field-identical to the HTTP `QualityWarningDTO` (stage/code/message/
    severity) without importing the HTTP layer: the graph never renders HTTP,
    and the service boundary maps 1:1.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    stage: str
    code: str
    message: str
    severity: Literal["info", "warning", "error"]


class CoverageEnvelope(BaseModel):
    """Truthful ask coverage, field-identical to the HTTP `EvidenceCoverage`.

    Counts are derived from the hydrated rows and the scope-unavailable
    documents, never asserted.
    """

    model_config = ConfigDict(extra="forbid")

    documents_requested: int = 0
    documents_ready: int = 0
    graph_indexed_documents: int = 0
    graph_unavailable_documents: int = 0


class GroundedCitation(BaseModel):
    """A citation with immutable provenance resolved from a hydrated row.

    Document identity, version, and contract come from the hydrated chunk
    row — never from edge text or any model output (the provenance discipline
    the inline `_citation_response`/`_edge_citations` enforce).
    """

    model_config = ConfigDict(extra="forbid")

    chunk_id: str
    clause_type: str
    claim: str
    document_id: str | None = None
    document_version: int | None = None
    contract_index: int | None = None
    pipeline_version: str | None = None
    page_no: int = 0


@dataclass
class RetrievalGraphContext:
    """Per-invocation collaborators (the `compiled-graph-lifecycle` context).

    The compiled graph receives every request-scoped collaborator through
    `ainvoke(..., context=...)` rather than capturing any of them at compile
    time: the language model (nodes derive their structured runnables from it
    per invocation), the repository, the cache client, the relationship-graph
    client, the reranker, and the optional grader seam (an absent seam
    degrades to the legacy inline grading path verbatim). A field left
    ``None`` keeps that node's default (``get_configured_reranker``, the
    canonical token budget, no cache, no graph leg).
    """

    llm: Any = None
    repo: Any | None = None
    redis: Any | None = None
    graphiti: Any | None = None
    reranker: Any | None = None
    grader: Any | None = None
    post_process_max_tokens: int | None = None
    count_tokens: CountTokens | None = None
    # Reserved for collaborators that must not become compile-time captures.
    extras: dict[str, Any] = field(default_factory=dict)


class CachedAskEnvelope(BaseModel):
    """The full ask envelope the graph caches under the version-aware key.

    Field-identical to the HTTP `UnifiedAskResponse` minus ``cache_hit``
    (which the serving boundary sets true on a hit), without importing the
    HTTP layer. The envelope — not the bare answer — is what the cache
    stores, so a cache hit serves the same coverage, warnings, citation
    provenance, and evidence policy the miss that wrote it served
    (the inline contract: the inline path cached the full response envelope).
    """

    model_config = ConfigDict(extra="forbid")

    answer: str
    confidence: Literal["high", "medium", "uncertain"]
    citations: list[GroundedCitation] = Field(default_factory=list)
    evidence_policy: Literal["mechanically_verified", "graph_indexed_only"] = (
        "mechanically_verified"
    )
    coverage: CoverageEnvelope = Field(default_factory=CoverageEnvelope)
    warnings: list[StageWarning] = Field(default_factory=list)


class RetrievalState(MessagesState, total=False):
    user_id: str
    tenant_id: str
    allowed_doc_ids: list[str]
    document_versions: list[tuple[str, int, str | None]]
    document_contracts: list[tuple[str, int, tuple[int | None, ...], bool]]
    scope_unavailable_documents: list[str]
    scope_bound: bool
    query: str
    doc_ids_filter: list[str]
    contract_index: int | None
    # Request filter fields the version-aware cache key and the hybrid
    # retrieval filter need, matching the inline `_build_answer_cache_key`
    # and `_ask` filter derivations.
    jurisdiction: str | None
    contract_type: str | None
    clause_type: str | None
    # Request evidence policy (task 1.3): `graph_indexed_only` when true,
    # `mechanically_verified` otherwise. Threads into the hybrid node's
    # filter params and the fallback/finalised envelope.
    require_graphiti_verified: bool
    query_plan: QueryPlan
    graph_chunk_ids: list[str]
    retrieved_chunks: list[RetrievedChunk]
    # The hydrated rows behind `retrieved_chunks`: the warning envelope,
    # coverage counts, and edge-span resolution all read them.
    retrieved_rows: list[dict[str, Any]]
    reranked_chunks: list[RetrievedChunk]
    assembled_context: list[ContextSection]
    context_grade: ContextGrade
    iteration_count: int
    generated_answer: GeneratedAnswer
    cache_hit: bool
    # The full ask envelope a cache hit serves (task 3.1 / the cache-hit
    # contract): coverage, warnings, citations, and evidence policy travel
    # with the answer, exactly as the inline response cache stored them.
    cached_envelope: CachedAskEnvelope
    # Edge-aware grounding (task 1.1): the generator's grounding lines and
    # the server-derived edge citations, both empty unless an authorized
    # graph scope, a healthy graphiti, and a typed edge touching the
    # reranked chunks all exist.
    grounding_edge_texts: list[str]
    edge_citations: list[GroundedCitation]
    # Finalisation (task 1.4): the truthful coverage envelope, merged and
    # deduplicated warnings, resolved citation provenance, and the derived
    # evidence policy.
    coverage: CoverageEnvelope
    quality_warnings: list[StageWarning]
    resolved_citations: list[GroundedCitation]
    evidence_policy: Literal["mechanically_verified", "graph_indexed_only"]
    answer_cache_key: str
    # Per-stage spend attribution (task 1.6); appended across nodes.
    stage_usage: Annotated[list[StageUsage], operator.add]
