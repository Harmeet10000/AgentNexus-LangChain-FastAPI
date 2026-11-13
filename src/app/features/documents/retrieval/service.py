"""The one scoped retrieval service (harden 5.1): ``retrieve(ctx, query, filters, k, budget)``.

Design decision 3 of the change's design.md: one service composed of fused
Postgres retrieval, explicit source narrowing, optional Graphiti
narrowing/coverage, reranking with a deterministic fused-order fallback, and
token budgeting. The service does **not** implement a second fusion — the
fused machinery is `retrieve_fused` (the exact function the served retrieval
graph's hybrid+fusion nodes wrap), and the parity suite
(``test_hybrid_fused_parity``) keeps the graph path and this service on that
shared core.

Every dependency call carries an explicit timeout (harden 5.2): the Postgres
fusion runs under ``DOCUMENTS_RETRIEVAL_POSTGRES_TIMEOUT_SECONDS`` (the same
gather deadline bounds the optional Graphiti leg), and the reranker keeps its
existing ``RERANKER_TIMEOUT_SECONDS`` deadline. An outage is classified —
``GraphitiError``/branch-attributed failures degrade the graph coverage
signal while canonical chunks stand; deadlines surface as the typed
``timeout`` status — never an indistinguishable empty result.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from returns.result import Failure

from app.ai.embeddings import EmbeddingTaskType, embed_text
from app.config import get_settings
from app.features.documents.adapters.persistence.postgres.document_repository import (
    build_search_filter_params,
)
from app.features.documents.api.http.dto import EvidenceCoverage
from app.features.documents.retrieval import row_hydration
from app.features.documents.retrieval.graph.state import StageWarning
from app.observability.degradation import (
    GRAPHITI_UNAVAILABLE,
    NO_EVIDENCE,
    RERANK_FALLBACK,
    record_degradation,
)
from app.observability.logging import logger
from app.retrieval.token_counter import count_tokens as default_count_tokens

if TYPE_CHECKING:
    from collections.abc import Sequence
    from typing import Any

    from app.features.documents.api.http.dto import EvidenceStatus
    from app.features.documents.retrieval.graph.budgets import StageBudgetTracker
    from app.features.documents.retrieval.graph.state import RetrievedChunk
    from app.features.documents.retrieval.reranking.reranker import Reranker
    from app.retrieval.token_counter import CountTokens

# Returned chunk text is token-budgeted (harden 5.1 "token budgeting"): a
# caller that asks for `k` chunks still receives at most this many tokens of
# chunk text, so a pathological large-chunk corpus cannot blow the downstream
# context. Truncation never drops below one chunk.
_DEFAULT_RESULT_TOKEN_BUDGET = 4_000

# Branch attribution key written by `_run_branches` on any branch failure.
_GRAPHITI_BRANCH = "graphiti"


@dataclass(frozen=True)
class RetrievalRequestContext:
    """Server-derived authority and collaborators for one retrieval call.

    Identity, tenant, and the document allowlist are server-owned (never
    model-supplied); the repository and optional graphiti/reranker/budget
    collaborators ride with the context exactly as the compiled graph's
    `RetrievalGraphContext` carries them per invocation.
    """

    principal_id: str
    tenant_id: str
    allowed_doc_ids: tuple[str, ...] = ()
    scope_bound: bool = True
    document_versions: Sequence[tuple[str, int, str | None]] = ()
    document_contracts: Sequence[tuple[str, int, tuple[int | None, ...], bool]] = ()
    repo: Any = None
    redis: Any | None = None
    graphiti: Any | None = None
    reranker: Reranker | None = None
    budget: StageBudgetTracker | None = None


@dataclass(frozen=True)
class RetrievalFilters:
    """Query-scoped filters (the model may choose these; never authority)."""

    document_ids: tuple[str, ...] = ()
    jurisdiction: str | None = None
    contract_type: str | None = None
    contract_index: int | None = None
    clause_type: str | None = None
    document_kind: str | None = None
    require_graphiti_verified: bool = False
    chunk_ids: tuple[str, ...] = ()
    candidate_limit: int = 50
    max_result_tokens: int = _DEFAULT_RESULT_TOKEN_BUDGET


@dataclass(frozen=True)
class RetrievalResult:
    """The typed retrieval envelope (harden 5.6's retrieval-side statuses).

    ``status`` is the shared `EvidenceStatus` vocabulary; an empty result is
    always `no_evidence` with coverage, never a silent outage — outages,
    deadlines, and exhausted budgets arrive as their own statuses with
    ``retryable`` set truthfully.
    """

    status: EvidenceStatus
    rows: list[dict[str, Any]] = field(default_factory=list)
    chunks: list[RetrievedChunk] = field(default_factory=list)
    coverage: EvidenceCoverage = field(default_factory=EvidenceCoverage)
    retryable: bool = False
    graph_available: bool = False
    message: str | None = None
    warnings: list[StageWarning] = field(default_factory=list)
    rank_by_chunk_id: dict[str, int] = field(default_factory=dict)
    score_by_chunk_id: dict[str, float] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.status == "ok"


async def retrieve(  # noqa: PLR0912, PLR0914 — one readable typed-status pipeline
    ctx: RetrievalRequestContext,
    query: str,
    filters: RetrievalFilters,
    k: int,
    budget: StageBudgetTracker | None = None,
) -> RetrievalResult:
    """Run one scoped, fused, timeout-bounded retrieval (harden 5.1).

    Stages: explicit scope narrowing → query embedding → the shared fused
    branch registry (`retrieve_fused`, optionally with the Graphiti leg) →
    row hydration through the single `row_hydration` path → optional rerank
    with fused-order fallback → token budgeting. Every failure mode returns a
    typed status; nothing collapses an outage, an unauthorized scope, or an
    exhausted budget into an empty list.
    """
    tracker = budget or ctx.budget
    scope = tuple(sorted({doc_id for doc_id in ctx.allowed_doc_ids if doc_id}))
    if ctx.scope_bound and not scope:
        return RetrievalResult(
            status="unauthorized",
            message="No authorized documents were supplied for this retrieval.",
        )

    # Explicit source narrowing (design decision 3): the request's document
    # ids participate only inside the server-owned allowlist. A request
    # naming out-of-scope documents is unauthorized — never silently emptied.
    requested_ids = tuple(dict.fromkeys(doc_id for doc_id in filters.document_ids if doc_id))
    if ctx.scope_bound and requested_ids:
        if any(doc_id not in scope for doc_id in requested_ids):
            return RetrievalResult(
                status="unauthorized",
                message="A requested document is outside the authorized corpus.",
            )
        narrowed = requested_ids
    else:
        narrowed = scope

    if tracker is not None and not tracker.charge("retrieve_fusion"):
        return RetrievalResult(
            status="budget_exhausted",
            message="The retrieval budget for this run is exhausted.",
        )

    settings = get_settings()
    try:
        embedding = await embed_text(query, task_type=EmbeddingTaskType.QUERY, redis=ctx.redis)
    except TimeoutError:
        return RetrievalResult(
            status="timeout",
            retryable=True,
            message="Query embedding timed out before retrieval could run.",
        )
    except Exception as exc:  # noqa: BLE001 — classified below, never an empty result
        return _transient_result(exc, "query embedding failed")

    filter_params = build_search_filter_params(
        metadata_filter={
            "document_ids": list(narrowed),
            "document_kind": filters.document_kind,
            "jurisdiction": filters.jurisdiction,
            "contract_type": filters.contract_type,
            "contract_index": filters.contract_index,
            "clause_type": filters.clause_type,
            "require_graphiti_verified": filters.require_graphiti_verified,
        }
    )
    graphiti_branch = ctx.graphiti if settings.DOCUMENTS_GRAPHITI_FUSION_ENABLED else None
    fused_outcome = await _fused_search(
        ctx=ctx,
        query=query,
        embedding=embedding,
        filter_params=filter_params,
        candidate_limit=filters.candidate_limit,
        graphiti_branch=graphiti_branch,
    )
    if isinstance(fused_outcome, RetrievalResult):
        return fused_outcome
    fused, lookup, graph_available = fused_outcome
    if graphiti_branch is not None and not graph_available:
        # The Graphiti leg failed or missed its deadline: canonical chunks
        # stand, the coverage signal degrades (legal-corpus-retrieval spec).
        record_degradation(GRAPHITI_UNAVAILABLE, surface="retrieval_service")

    if not fused:
        record_degradation(NO_EVIDENCE, surface="retrieval_service")
        return RetrievalResult(
            status="no_evidence",
            message="No evidence was found in the authorized documents.",
            coverage=EvidenceCoverage(documents_requested=len(narrowed)),
        )

    from app.features.documents.application.queries.documents import (  # noqa: PLC0415 — queries.documents imports this package's graph at module load
        lookup_to_rows,
    )

    rows = lookup_to_rows(fused, lookup)[:k]
    chunks = [row_hydration.row_to_chunk(row) for row in rows]
    chunks = _apply_token_budget(
        chunks, filters.max_result_tokens, count_tokens=default_count_tokens
    )
    rows = rows[: len(chunks)]

    rank_by_chunk_id = {item.chunk_id: item.rank for item in fused}
    score_by_chunk_id = {item.chunk_id: item.score for item in fused}

    if ctx.reranker is not None and chunks:
        reranked = await _rerank_with_fallback(ctx.reranker, query, chunks, limit=k)
        if reranked is None:
            record_degradation(RERANK_FALLBACK, surface="retrieval_service")
        else:
            chunks = reranked
            rows = _reorder_rows(rows, chunks)
    graph_document_ids = {
        str(row["document_id"]) for row in rows if row_hydration.graph_indexed(row)
    }
    result_document_ids = {str(row["document_id"]) for row in rows}
    warnings = (
        []
        if graph_available
        else [
            StageWarning(
                stage="retrieve_fusion",
                code="GRAPHITI_UNAVAILABLE",
                message=(
                    "The relationship graph was unavailable; coverage is degraded and "
                    "canonical chunks were returned."
                ),
                severity="warning",
            )
        ]
    )
    return RetrievalResult(
        status="ok",
        rows=rows,
        chunks=chunks,
        coverage=EvidenceCoverage(
            documents_requested=len(narrowed),
            documents_ready=len(result_document_ids),
            graph_indexed_documents=len(graph_document_ids),
            graph_unavailable_documents=len(result_document_ids - graph_document_ids),
        ),
        graph_available=graph_available,
        rank_by_chunk_id=rank_by_chunk_id,
        score_by_chunk_id=score_by_chunk_id,
        warnings=warnings,
    )


async def _fused_search(
    *,
    ctx: RetrievalRequestContext,
    query: str,
    embedding: list[float],
    filter_params: dict[str, Any],
    candidate_limit: int,
    graphiti_branch: Any | None,
) -> RetrievalResult | tuple[list[Any], dict[str, dict[str, Any]], bool]:
    """One fused search under the explicit Postgres deadline (harden 5.2).

    Returns ``(fused, lookup, graph_available)`` on success, or a typed
    `RetrievalResult` degraded outcome. A Graphiti-attributed branch failure
    degrades to the canonical-only fusion (chunks preserved, coverage
    signal down); every other failure and the deadline itself surface as
    typed statuses.
    """
    from app.features.documents.application.queries.documents import (  # noqa: PLC0415 — see retrieve()
        retrieve_fused,
    )

    settings = get_settings()
    try:
        async with asyncio.timeout(settings.DOCUMENTS_RETRIEVAL_POSTGRES_TIMEOUT_SECONDS):
            fused_result = await retrieve_fused(
                repo=ctx.repo,
                user_id=ctx.principal_id,
                tenant_id=ctx.tenant_id,
                query_text=query,
                query_embedding=embedding,
                candidate_limit=candidate_limit,
                limit=candidate_limit,
                filter_params=filter_params,
                weights=None,
                graphiti=graphiti_branch,
                document_versions=ctx.document_versions,
                document_contracts=ctx.document_contracts,
            )
    except TimeoutError:
        return RetrievalResult(
            status="timeout",
            retryable=True,
            message="The fused retrieval deadline expired.",
        )
    if isinstance(fused_result, Failure):
        error = fused_result.failure()
        if _is_graphiti_branch_failure(error):
            degraded = await _fused_search(
                ctx=ctx,
                query=query,
                embedding=embedding,
                filter_params=filter_params,
                candidate_limit=candidate_limit,
                graphiti_branch=None,
            )
            if isinstance(degraded, RetrievalResult):
                return degraded
            return degraded[0], degraded[1], False
        logger.bind(operation="retrieval_service_fusion", error=error.message).warning(
            "retrieval_fusion_failed"
        )
        return RetrievalResult(
            status="transient_failure",
            retryable=bool(getattr(error, "retryable", True)),
            message=error.message,
        )
    fused, lookup = fused_result.unwrap()
    return fused, lookup, graphiti_branch is not None


def _is_graphiti_branch_failure(error: object) -> bool:
    details = getattr(error, "details", None)
    return isinstance(details, dict) and details.get("branch") == _GRAPHITI_BRANCH


def _transient_result(exc: Exception, operation: str) -> RetrievalResult:
    logger.bind(operation=operation, error=type(exc).__name__).warning(
        "retrieval_dependency_failed"
    )
    return RetrievalResult(
        status="transient_failure", retryable=True, message=f"{operation}: {exc}"
    )


async def _rerank_with_fallback(
    reranker: Reranker,
    query: str,
    chunks: list[RetrievedChunk],
    *,
    limit: int,
) -> list[RetrievedChunk] | None:
    """Rerank under the hosted deadline; None signals the fused-order fallback."""
    settings = get_settings()
    try:
        async with asyncio.timeout(settings.RERANKER_TIMEOUT_SECONDS):
            return await reranker.rerank(query, chunks, limit=limit)
    except Exception:  # noqa: BLE001 — fallback is the contract, never a failure surface
        logger.bind(operation="retrieval_service_rerank").warning("rerank_fallback_to_fused_order")
        return None


def _apply_token_budget(
    chunks: list[RetrievedChunk],
    max_tokens: int,
    *,
    count_tokens: CountTokens,
) -> list[RetrievedChunk]:
    """Keep at most `max_tokens` of chunk text, never fewer than one chunk."""
    if not chunks:
        return chunks
    kept: list[RetrievedChunk] = []
    spent = 0
    for chunk in chunks:
        cost = count_tokens(chunk.chunk_text)
        if kept and spent + cost > max_tokens:
            break
        kept.append(chunk)
        spent += cost
    return kept


def _reorder_rows(rows: list[dict[str, Any]], chunks: list[RetrievedChunk]) -> list[dict[str, Any]]:
    """Reorder hydrated rows to the reranked chunk order, dropping losses."""
    by_id = {str(row["chunk_id"]): row for row in rows}
    return [by_id[chunk.chunk_id] for chunk in chunks if chunk.chunk_id in by_id]
