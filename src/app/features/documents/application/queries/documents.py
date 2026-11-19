"""Unified document feature services."""

from __future__ import annotations

import asyncio
import hashlib
import inspect
from collections.abc import Mapping
from operator import itemgetter
from typing import TYPE_CHECKING, NamedTuple, Protocol, cast, runtime_checkable

from returns.result import Failure, Success

from app.ai.embeddings import EmbeddingTaskType, embed_text
from app.ai.generation import TextGenerationRequest
from app.features.auth.domain.tenancy import personal_tenant_id
from app.features.documents.adapters.persistence.postgres.document_repository import (
    DocumentRepository,
    build_search_filter_params,
)
from app.features.documents.api.http.dto import (
    DocumentSearchResultItem,
    EvidenceCoverage,
    EvidencePolicy,
    LegalCitationResponse,
    QualityWarningDTO,
    RagContextSectionResponse,
    UnifiedAskResponse,
    UnifiedRagResponse,
    UnifiedSearchRequest,
    UnifiedSearchResponse,
)
from app.features.documents.application.access_audit import (
    DocumentAccessAuditEvent,
    DocumentAccessAuditRecorder,
    InMemoryDocumentAccessAuditRecorder,
)
from app.features.documents.domain.constants import (
    CANONICAL_PIPELINE_VERSION,
    DEFAULT_SEARCH_CACHE_TTL_SECONDS,
    RRF_K,
)
from app.features.documents.domain.errors import DocumentDatabaseError
from app.features.documents.retrieval import row_hydration
from app.features.documents.retrieval.context_assembly import (
    SearchChunkRecord,
    assemble_rag_context,
)
from app.features.documents.retrieval.graph import (
    RetrievalEngine,
    RetrievalGraphContext,
)
from app.features.documents.retrieval.graph.budgets import build_stage_budget_tracker
from app.kernel.result import log_expected_failure
from app.observability.logging import logger, trace_layer
from app.retrieval.fusion import RankedChunk, RankedResultRow, reciprocal_rank_fusion
from app.retrieval.graphiti.retrieval_branch import fetch_ranked_chunk_ids
from app.retrieval.structural_navigator import navigate_tree
from app.retrieval.token_counter import count_tokens as default_count_tokens
from app.utils import InfrastructureException, ServiceUnavailableException, to_sorted_key_bytes

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Sequence
    from typing import Any

    from graphiti_core.graphiti import Graphiti
    from langchain_core.language_models import BaseChatModel
    from redis.asyncio import Redis

    from app.ai.generation import TextGenerationStructuredResult
    from app.features.documents.api.http import dto as documents_dto
    from app.features.documents.domain.errors import DocumentResult
    from app.features.documents.ingestion.chunking.classification import QualityWarning
    from app.features.documents.ingestion.enrichment.legal_metadata import (
        LegalMetadataExtraction,
    )
    from app.features.documents.retrieval.context_assembly import ContextSection
_FALLBACK_ANSWER = (
    "I do not have enough grounded document context to answer this reliably. "
    "Please narrow the question or ingest the relevant document sections."
)
_KNOWLEDGE_EXTRACTION_PROMPT = (
    "Extract grounded legal clauses and their attributes. Use extraction classes such as "
    "indemnity, termination, payment, confidentiality, governing_law, arbitration, "
    "limitation_of_liability, ip_ownership, or obliges. Preserve source character offsets."
)


# Retrieval branch registry. Each branch is a (name, run) record: `name` is the
# attribution label a failure carries, and `run` builds that branch's coroutine
# from one shared input. The pairing between names and `asyncio.gather`'s
# positional results used to live in the `_SEARCH_BRANCHES` tuple alone; a
# reorder there silently misattributed every failure, so the name now travels
# with the callable and the zip in `_run_branches` pairs records with
# results. Adding branch #4 is a registration below — the gather/zip/fusion
# does not change. Per-leg fusion weights travel on `retrieve_fused`'s `weights`
# argument (one weight per leg, defaulting to unweighted); the query plan's
# vector and keyword weights map onto the first two legs at the call sites.


class _BranchInput(NamedTuple):
    """Shared input every fused-retrieval branch runs from.

    The ``graphiti``/``document_versions``/``document_contracts`` tail exists
    for the optional relationship-graph branch (``retrieval-measurement-gate``):
    the three mandatory legs never read them, and the graph branch is registered
    only when the caller passes a client.
    """

    user_id: str
    tenant_id: str
    query_text: str
    query_embedding: list[float]
    candidate_limit: int
    filter_params: dict[str, Any]
    bm25_threshold: float | None = None
    exact_phrase: str | None = None
    graphiti: Graphiti | None = None
    document_versions: Sequence[tuple[str, int, str | None]] = ()
    document_contracts: Sequence[tuple[str, int, tuple[int | None, ...], bool]] = ()


class _LegalPreparation(NamedTuple):
    metadata: LegalMetadataExtraction | None
    warnings: list[QualityWarning]
    # True when the metadata extraction ran but failed validation: the record
    # must carry the explicit metadata_validation_failed state and must not
    # merge unvalidated heuristic metadata (validation-seam-adoption).
    validation_failed: bool = False


async def _run_bm25_branch(
    repo: DocumentRepository, args: _BranchInput
) -> DocumentResult[list[dict[str, Any]]]:
    """Keyword branch coroutine."""
    return await repo.bm25_search(
        user_id=args.user_id,
        tenant_id=args.tenant_id,
        query=args.query_text,
        candidate_limit=args.candidate_limit,
        filter_params=args.filter_params,
        bm25_threshold=args.bm25_threshold,
        exact_phrase=args.exact_phrase,
    )


async def _run_vector_branch(
    repo: DocumentRepository, args: _BranchInput
) -> DocumentResult[list[dict[str, Any]]]:
    """Dense-vector branch coroutine."""
    return await repo.vector_search(
        user_id=args.user_id,
        tenant_id=args.tenant_id,
        embedding=args.query_embedding,
        candidate_limit=args.candidate_limit,
        filter_params=args.filter_params,
    )


async def _run_graphiti_branch(
    repo: DocumentRepository, args: _BranchInput
) -> DocumentResult[list[dict[str, Any]]]:
    """Relationship-graph branch coroutine (optional fourth leg).

    The repository is unused: the read goes to the Graphiti knowledge graph,
    ranked, tenant-scoped, over the same chunk-identifier space the SQL legs
    return. A raised transport failure is converted to a typed Failure so
    `_run_branches` attributes it to the branch — the enabled branch fails the
    request naming the branch, it never silently shrinks the fusion. An empty
    ranking — no reachable relationships for the query — contributes nothing.
    """
    _ = repo
    try:
        chunk_ids = await fetch_ranked_chunk_ids(
            args.graphiti,
            args.query_text,
            allowed_doc_ids=tuple(args.filter_params.get("document_ids") or ()),
            tenant_id=args.tenant_id,
            contract_index=args.filter_params.get("contract_index"),
            document_versions=args.document_versions,
            document_contracts=args.document_contracts,
        )
    except Exception as exc:  # noqa: BLE001 — converted to a typed Failure below.
        return Failure(
            DocumentDatabaseError(
                message=f"graphiti retrieval branch read failed: {exc}",
                details={"branch": GRAPHITI_BRANCH_NAME, "error": type(exc).__name__},
            )
        )
    return Success(
        [
            {"chunk_id": chunk_id, "score": 1.0 / rank}
            for rank, chunk_id in enumerate(chunk_ids, start=1)
        ]
    )


class RetrievalBranchPolicy(NamedTuple):
    """One fused-retrieval branch: attribution name plus how to run it."""

    name: str
    run: Callable[
        [DocumentRepository, _BranchInput], Awaitable[DocumentResult[list[dict[str, Any]]]]
    ]


RETRIEVAL_BRANCHES: tuple[RetrievalBranchPolicy, ...] = (
    RetrievalBranchPolicy(name="bm25", run=_run_bm25_branch),
    RetrievalBranchPolicy(name="vector", run=_run_vector_branch),
)

# The optional fourth branch: registered here per the registry contract (name
# paired with its runnable), but appended to the fused set only when the caller
# passes a Graphiti client (the ask path does so only when
# DOCUMENTS_GRAPHITI_FUSION_ENABLED is set). Gather/zip/fusion code is shared
# with the mandatory legs unchanged; a failure attributes to "graphiti".
GRAPHITI_BRANCH_NAME = "graphiti"


def fused_branches(graphiti: Graphiti | None = None) -> tuple[RetrievalBranchPolicy, ...]:
    """Return the branches the fusion runs: the mandatory two, plus graph."""
    if graphiti is None:
        return RETRIEVAL_BRANCHES
    return (
        *RETRIEVAL_BRANCHES,
        RetrievalBranchPolicy(name=GRAPHITI_BRANCH_NAME, run=_run_graphiti_branch),
    )


def graphiti_branch_with_deadline(timeout_seconds: float) -> RetrievalBranchPolicy:
    """The graphiti branch policy bounded by its own deadline (harden 5.2).

    The deadline that wraps the whole fusion (``DOCUMENTS_RETRIEVAL_POSTGRES_TIMEOUT_SECONDS``)
    cannot also bound the graph leg: a Graphiti hang would consume the entire
    window and surface as a fused-search timeout, discarding healthy canonical
    results before the canonical-only fallback could run. With the leg under
    ``DOCUMENTS_RETRIEVAL_GRAPHITI_TIMEOUT_SECONDS``, a deadline miss is the
    same typed, branch-attributed Failure a raised transport failure produces —
    the caller's existing graphiti-failure routing handles it unchanged.
    """

    async def _run(
        repo: DocumentRepository, args: _BranchInput
    ) -> DocumentResult[list[dict[str, Any]]]:
        try:
            async with asyncio.timeout(timeout_seconds):
                return await _run_graphiti_branch(repo, args)
        except TimeoutError:
            return Failure(
                DocumentDatabaseError(
                    message=f"graphiti retrieval branch deadline ({timeout_seconds}s) expired",
                    details={"branch": GRAPHITI_BRANCH_NAME, "error": "timeout"},
                )
            )

    return RetrievalBranchPolicy(name=GRAPHITI_BRANCH_NAME, run=_run)


# Derived, not authored: the i-th label names the i-th gathered result, and the
# pairing tests pin this derivation rather than a hand-kept tuple.
_SEARCH_BRANCHES = tuple(branch.name for branch in RETRIEVAL_BRANCHES)


async def _run_branches(
    *,
    repo: DocumentRepository,
    branch_input: _BranchInput,
    branches: Sequence[RetrievalBranchPolicy] = RETRIEVAL_BRANCHES,
    branch_diagnostics: dict[str, Any] | None = None,
) -> DocumentResult[list[list[RankedResultRow]]]:
    """Run every retrieval branch and return ranked row sets, or fail with attribution.

    A branch that *raised* keeps its own error kind via `model_copy` — re-wrapping
    would flatten the taxonomy and turn a 422 from one branch into a 503 for the
    whole request — with only the branch name added. When ``branch_diagnostics``
    is provided, each executed branch's contribution (name, identifiers
    contributed) is recorded into it for the response path's diagnostics.
    """
    results = await asyncio.gather(*(branch.run(repo, branch_input) for branch in branches))
    row_sets: list[list[RankedResultRow]] = []
    for branch, branch_result in zip(branches, results, strict=True):
        if isinstance(branch_result, Failure):
            error = branch_result.failure()
            return Failure(
                error.model_copy(
                    update={
                        "message": f"{branch.name} retrieval branch failed: {error.message}",
                        "details": {**(error.details or {}), "branch": branch.name},
                    }
                )
            )
        # `else`, not `elif isinstance(..., Success)`. The old form had no final branch, so a
        # value that was neither would have been dropped from `row_sets` entirely — shrinking
        # the fusion input with no log line and no failure.
        row_set = _to_ranked_rows(branch_result.unwrap())
        row_sets.append(row_set)
        if branch_diagnostics is not None:
            branch_diagnostics.setdefault("branches_executed", []).append(
                {
                    "branch": branch.name,
                    "executed": True,
                    "identifiers_contributed": len(row_set),
                }
            )
    return Success(row_sets)


async def retrieve_fused(
    *,
    repo: DocumentRepository,
    user_id: str,
    query_text: str,
    query_embedding: list[float],
    candidate_limit: int,
    limit: int,
    filter_params: dict[str, Any],
    weights: Sequence[float] | None = None,
    bm25_threshold: float | None = None,
    exact_phrase: str | None = None,
    tenant_id: str | None = None,
    graphiti: Graphiti | None = None,
    document_versions: Sequence[tuple[str, int, str | None]] = (),
    document_contracts: Sequence[tuple[str, int, tuple[int | None, ...], bool]] = (),
    branch_diagnostics: dict[str, Any] | None = None,
    graphiti_timeout_seconds: float | None = None,
) -> DocumentResult[tuple[list[RankedChunk], dict[str, dict[str, Any]]]]:
    """Run the branch legs over base chunks and fuse them in Python.

    The single fused path shared by the search endpoint, `ask`, and the
    retrieval-graph hybrid node: every caller issues the same branch SQL with the
    same inputs and fuses with the same weights, so one query yields identical
    chunk-id order from every door. A branch that *raised* keeps its attribution
    (branch name in details); a branch that legitimately matches nothing
    contributes an empty rank list and fusion proceeds. The optional
    relationship-graph leg (``graphiti``) joins the fusion through the branch
    registry — no second fusion implementation — and only when the caller
    passes a client. When ``graphiti_timeout_seconds`` is set, the graph leg
    runs under its own deadline instead of the caller's fused deadline, so a
    slow graph degrades to the canonical-only fusion rather than expiring the
    whole fused read.
    """
    branch_input = _BranchInput(
        user_id=user_id,
        tenant_id=_resolve_tenant_id(tenant_id, user_id),
        query_text=query_text,
        query_embedding=query_embedding,
        candidate_limit=candidate_limit,
        filter_params=filter_params,
        bm25_threshold=bm25_threshold,
        exact_phrase=exact_phrase,
        graphiti=graphiti,
        document_versions=document_versions,
        document_contracts=document_contracts,
    )
    branches = fused_branches(graphiti)
    if graphiti is not None and graphiti_timeout_seconds is not None:
        branches = (*RETRIEVAL_BRANCHES, graphiti_branch_with_deadline(graphiti_timeout_seconds))
    results = await _run_branches(
        repo=repo,
        branch_input=branch_input,
        branches=branches,
        branch_diagnostics=branch_diagnostics,
    )
    if isinstance(results, Failure):
        return Failure(results.failure())
    row_sets = results.unwrap()
    # Phrase matches may rank below many vector-only candidates. Preserve
    # every candidate already bounded by the branch limits until the phrase
    # constraint has been applied; a second pre-filter cutoff can lose valid hits.
    fusion_limit = sum(len(row_set) for row_set in row_sets) if exact_phrase else limit
    fused = reciprocal_rank_fusion(*row_sets, k=RRF_K, limit=fusion_limit, weights=weights)
    if not fused:
        return Success(([], {}))
    chunk_ids = [item.chunk_id for item in fused]
    fetch_kwargs: dict[str, Any] = {}
    # Keep narrow repository test doubles and downstream adapters source
    # compatible while using the tenant-aware repository in production.
    fetch_parameters = inspect.signature(repo.fetch_chunks_by_ids).parameters
    if "exact_phrase" in fetch_parameters:
        fetch_kwargs["exact_phrase"] = exact_phrase
    if "tenant_id" in fetch_parameters:
        fetch_kwargs["tenant_id"] = _resolve_tenant_id(tenant_id, user_id)
    if "allowed_doc_ids" in fetch_parameters:
        fetch_kwargs["allowed_doc_ids"] = filter_params.get("document_ids") or None
    chunk_lookup_result = await repo.fetch_chunks_by_ids(chunk_ids, **fetch_kwargs)
    if isinstance(chunk_lookup_result, Failure):
        return Failure(chunk_lookup_result.failure())
    lookup: dict[str, dict[str, Any]] = chunk_lookup_result.unwrap()
    if exact_phrase:
        # Hydration applies the same escaped PostgreSQL ILIKE predicate used by
        # the BM25 leg, so vector candidates cannot diverge on Unicode
        # or wildcard semantics.
        fused = [item for item in fused if item.chunk_id in lookup][:limit]
        lookup = {item.chunk_id: lookup[item.chunk_id] for item in fused if item.chunk_id in lookup}
    return Success((fused, lookup))


def lookup_to_rows(
    fused: Sequence[RankedChunk],
    lookup: dict[str, dict[str, Any]],
) -> list[dict[str, object]]:
    """Adapt fused ranks plus the chunk lookup to `_row_to_chunk` row shape."""
    rows: list[dict[str, object]] = []
    for item in fused:
        record = lookup.get(item.chunk_id)
        if record is None:
            continue
        rows.append(
            {
                "chunk_id": item.chunk_id,
                "chunk_text": record.get("content"),
                "preamble": record.get("preamble") or "",
                "clause_type": record.get("clause_type"),
                # `document_id` backs citation provenance and coverage counts;
                # `parent_doc_id` is the historical `_row_to_chunk` key. Both
                # name the same hydrated identity.
                "document_id": record.get("document_id"),
                "parent_doc_id": record.get("document_id"),
                "metadata_": record.get("chunk_metadata") or {},
                "custom_metadata": {},
                "quality_warnings": record.get("quality_warnings", []),
                "graphiti_verified": bool(record.get("graphiti_verified", False)),
                "graph_indexed": bool(record.get("graphiti_verified", False)),
                "graph_indexed_at": record.get("graph_indexed_at"),
                "document_version": record.get("document_version", 1),
                "contract_index": record.get("contract_index"),
                # Harden 5.3: hydrated through the ONE provenance reader — a
                # legacy row without the column still carries the version its
                # chunk/document metadata recorded.
                "pipeline_version": row_hydration.pipeline_version(record),
                "rrf_score": item.score,
                "title": record.get("title"),
                "chunk_index": record.get("chunk_index"),
                "chunk_kind": record.get("chunk_kind"),
                # Harden 5.3: the row hydration DTOs carry the page number so
                # citation provenance (`_citation_response`) and the evidence
                # refs read it from the same hydrated row — it used to be
                # dropped here, pinning ask-citation pages at 0.
                "page_no": record.get("page_no"),
            }
        )
    return rows


@runtime_checkable
class ContextGrader(Protocol):
    """The narrow slice of the text-generation seam the context grader consumes.

    Structural, so the lifespan-owned ``LangChainTextGenerator`` and any test
    double with the same single method both satisfy it; documents never import
    the concrete adapter.
    """

    def generate_structured(
        self, request: TextGenerationRequest
    ) -> Awaitable[TextGenerationStructuredResult]: ...


class DocumentQueryService:
    """Unified retrieval and grounded QA service."""

    def __init__(
        self,
        repo: DocumentRepository,
        llm_factory: Callable[[], BaseChatModel],
        redis: Redis | None,
        graphiti: Graphiti | None,
        *,
        grader: ContextGrader | None = None,
        audit_recorder: DocumentAccessAuditRecorder | None = None,
        retrieval_engine: RetrievalEngine | None = None,
    ):
        self.repo: DocumentRepository = repo
        # The model client is constructed on first use, not at dependency
        # resolution: building it eagerly lets an environment failure (missing
        # provider package, bad key) answer an unauthenticated request with a
        # 500 that masks the 401 auth already earned.
        self._llm: BaseChatModel | None = None
        self._llm_factory: Callable[[], BaseChatModel] = llm_factory
        self.redis: Redis | None = redis
        self.graphiti: Graphiti | None = graphiti
        # Optional structured-grading seam (item 274 reopen pilot): the lifespan
        # owns `app.state.text_generator`, which text generation disables when
        # Redis or the provider key is missing. A disabled seam must not take
        # `ask` down, so ``None`` keeps grading on the legacy inline path below;
        # every other consumer of `self.llm` is unchanged either way.
        self.grader: ContextGrader | None = grader
        self.audit_recorder = audit_recorder or InMemoryDocumentAccessAuditRecorder()
        # The compile-once retrieval engine (mount-retrieval-graph 4.x): the
        # graph-served ask path invokes `retrieval_engine.graph` with this
        # request's collaborators as the invocation context. ``None`` means
        # this process has no provisioned engine and `_ask` fails closed with
        # a typed unavailability — never a fallback answering path.
        self.retrieval_engine = retrieval_engine

    async def _record_access(
        self,
        result: DocumentResult[Any],
        *,
        operation: str,
        user_id: str,
        tenant_id: str,
        requested_document_ids: Sequence[str],
        document_ids: Sequence[str] = (),
        reason: str | None = None,
        outcome: str | None = None,
    ) -> DocumentResult[Any]:
        outcome = outcome or ("ok" if isinstance(result, Success) else "denied")
        if reason is None and isinstance(result, Failure):
            reason = result.failure().message
        try:
            await self.audit_recorder.append(
                DocumentAccessAuditEvent(
                    tenant_id=tenant_id,
                    principal_id=user_id,
                    operation=operation,
                    requested_document_ids=tuple(requested_document_ids),
                    document_ids=tuple(document_ids),
                    policy="mechanically_verified",
                    outcome=outcome,
                    reason=reason,
                )
            )
        except Exception as exc:  # noqa: BLE001 — audit is a fail-closed boundary
            exc.add_note(f"operation={operation}, tenant_id={tenant_id}")
            logger.bind(operation=operation, tenant_id=tenant_id).exception(
                "document_access_audit_failed"
            )
        return result

    @property
    def llm(self) -> BaseChatModel:
        if self._llm is None:
            self._llm = self._llm_factory()
        return self._llm

    @trace_layer("service")
    async def navigate_structure(
        self,
        *,
        user_id: str,
        tenant_id: str | None = None,
        query: str,
        document_ids: list[str] | None = None,
        limit: int = 5,
    ) -> DocumentResult[list[tuple[str, ...]]]:
        """Navigate stored document trees through the repository boundary."""
        trees_result = await self.repo.fetch_structural_trees(
            user_id=user_id,
            tenant_id=tenant_id,
            document_ids=document_ids or [],
        )
        if isinstance(trees_result, Failure):
            return Failure(trees_result.failure())
        paths: list[tuple[str, ...]] = []
        for row in trees_result.unwrap():
            tree = row.get("structural_tree")
            if not isinstance(tree, dict):
                continue
            document_id = str(row.get("document_id") or "")
            paths.extend((document_id, *path) for path in navigate_tree(tree, query, limit=limit))
            if len(paths) >= limit:
                break
        return Success(paths[:limit])

    @trace_layer("service")
    async def search(
        self,
        *,
        user_id: str,
        tenant_id: str | None = None,
        allowed_doc_ids: tuple[str, ...] | None = None,
        payload: UnifiedSearchRequest,
        weights: Sequence[float] | None = None,
    ) -> DocumentResult[UnifiedSearchResponse]:
        resolved_tenant = _resolve_tenant_id(tenant_id, user_id)
        result = await self._search(
            user_id=user_id,
            tenant_id=resolved_tenant,
            allowed_doc_ids=allowed_doc_ids,
            payload=payload,
            weights=weights,
        )
        scope = tuple(allowed_doc_ids or ())
        document_ids = (
            tuple(item.document_id for item in result.unwrap().items)
            if isinstance(result, Success)
            else ()
        )
        return cast(
            "DocumentResult[UnifiedSearchResponse]",
            await self._record_access(
                result,
                operation="document.search",
                user_id=user_id,
                tenant_id=resolved_tenant,
                requested_document_ids=scope,
                document_ids=document_ids,
                reason=("requested document scope is empty" if allowed_doc_ids == () else None),
                outcome="denied" if allowed_doc_ids == () else None,
            ),
        )

    async def _search(  # noqa: PLR0914
        self,
        *,
        user_id: str,
        tenant_id: str | None = None,
        allowed_doc_ids: tuple[str, ...] | None = None,
        payload: UnifiedSearchRequest,
        weights: Sequence[float] | None = None,
    ) -> DocumentResult[UnifiedSearchResponse]:
        tenant_id = _resolve_tenant_id(tenant_id, user_id)
        explicit_scope = allowed_doc_ids is not None
        scope_ids = tuple(dict.fromkeys(allowed_doc_ids or ()))
        if explicit_scope and not scope_ids:
            return Success(UnifiedSearchResponse(items=[], cache_hit=False))
        scope_versions = await _resolve_scope_versions(
            self.repo, tenant_id=tenant_id, allowed_doc_ids=scope_ids
        )
        cache_key = _build_cache_key(
            "documents:search",
            payload,
            user_id=user_id,
            tenant_id=tenant_id,
            allowed_doc_ids=scope_ids,
            document_versions=scope_versions,
            evidence_policy="mechanically_verified",
        )
        lock_key = f"{cache_key}:lock"
        lock_acquired = False
        if not payload.bypass_cache and self.redis is not None:
            cached = await self.redis.get(cache_key)
            if cached is not None:
                response: UnifiedSearchResponse = UnifiedSearchResponse.model_validate_json(cached)
                return Success(response.model_copy(update={"cache_hit": True}))

            # ponytail: setnx lock prevents concurrent duplicate compute for same query
            lock_acquired = await self.redis.setnx(lock_key, "1")
            if not lock_acquired:
                for _ in range(30):
                    await asyncio.sleep(0.05)
                    cached = await self.redis.get(cache_key)
                    if cached is not None:
                        return Success(
                            UnifiedSearchResponse.model_validate_json(cached).model_copy(
                                update={"cache_hit": True}
                            )
                        )
            else:
                await self.redis.expire(lock_key, 15)

        query_embedding = await embed_text(
            payload.query,
            task_type=EmbeddingTaskType.QUERY,
            redis=None if payload.bypass_cache else self.redis,
        )
        filter_params = build_search_filter_params(
            metadata_filter={
                **payload.metadata_filter.model_dump(),
                "document_ids": list(scope_ids)
                if explicit_scope
                else payload.metadata_filter.document_ids,
                "scope_bound": explicit_scope,
            }
        )
        fused_result = await retrieve_fused(
            repo=self.repo,
            user_id=user_id,
            tenant_id=tenant_id,
            query_text=payload.query,
            query_embedding=query_embedding,
            candidate_limit=payload.candidate_limit,
            limit=payload.limit,
            filter_params=filter_params,
            weights=weights,
        )
        if isinstance(fused_result, Failure):
            error = fused_result.failure()
            log_expected_failure(error, operation="hybrid_search")
            # The setnx lock above is not released on this path. It carries a 15s expiry for
            # exactly this reason, and every other raise in this method already relied on it.
            return Failure(error)
        fused_results, chunk_lookup = fused_result.unwrap()
        items: list[DocumentSearchResultItem] = _build_search_items(
            fused_results=fused_results, chunk_lookup=chunk_lookup
        )
        response = UnifiedSearchResponse(items=items, cache_hit=False)
        if not payload.bypass_cache and self.redis is not None:
            await self.redis.setex(
                name=cache_key,
                time=DEFAULT_SEARCH_CACHE_TTL_SECONDS,
                value=response.model_dump_json(),
            )
            if lock_acquired:
                await self.redis.delete(lock_key)
        return Success(response)

    async def _fuse_search_branches(
        self,
        *,
        user_id: str,
        tenant_id: str | None = None,
        payload: UnifiedSearchRequest,
        query_embedding: list[float],
        filter_params: dict[str, Any],
        weights: Sequence[float] | None = None,
    ) -> DocumentResult[list[RankedChunk]]:
        """Run the three retrieval modes and fuse them, or fail naming the branch that broke.

        The distinction this method exists to draw: **an empty result from a healthy branch is
        not a failure.** A keyword branch that legitimately matches nothing contributes an empty
        rank list and the fusion proceeds over two modes; a keyword branch that *raised* used to
        contribute an identical empty rank list, so a partially-broken index answered `200` with
        results silently fused from fewer modes than the caller asked for. The two cases were
        indistinguishable in the response, which is what made the old degrade path a correctness
        problem rather than a resilience feature.

        Returned as a `Result` rather than raised because the branch identity is the payload: the
        caller is the ownership boundary, and a test can assert the branch name without having to
        catch an exception and re-parse its message.
        """
        branch_input = _BranchInput(
            user_id=user_id,
            tenant_id=_resolve_tenant_id(tenant_id, user_id),
            query_text=payload.query,
            query_embedding=query_embedding,
            candidate_limit=payload.candidate_limit,
            filter_params=filter_params,
        )
        row_sets_result = await _run_branches(repo=self.repo, branch_input=branch_input)
        if isinstance(row_sets_result, Failure):
            return Failure(row_sets_result.failure())
        return Success(
            reciprocal_rank_fusion(
                *row_sets_result.unwrap(),
                k=RRF_K,
                limit=payload.limit,
                weights=weights,
            )
        )

    @trace_layer("service")
    async def rag(
        self,
        *,
        user_id: str,
        tenant_id: str | None = None,
        allowed_doc_ids: tuple[str, ...] | None = None,
        payload: documents_dto.UnifiedRagRequest,
    ) -> DocumentResult[UnifiedRagResponse]:
        tenant_id = _resolve_tenant_id(tenant_id, user_id)
        # ``UnifiedSearchRequest`` is a strict (extra="forbid") schema while ``UnifiedRagRequest``
        # adds ``max_tokens``. Dumping the RAG payload verbatim and re-validating it against the
        # strict search schema always failed on the inherited ``max_tokens`` default (F1) — every
        # /search/rag request 500'd. Drop the RAG-only field before the strict re-validation.
        search_payload = UnifiedSearchRequest.model_validate(
            payload.model_dump(exclude={"max_tokens"})
        )
        search_result = await self.search(
            user_id=user_id,
            tenant_id=tenant_id,
            allowed_doc_ids=allowed_doc_ids,
            payload=search_payload,
        )
        if isinstance(search_result, Failure):
            return Failure(search_result.failure())
        response = search_result.unwrap()
        chunk_lookup: dict[str, SearchChunkRecord] = {
            item.chunk_id: SearchChunkRecord(
                document_id=item.document_id,
                title=item.title,
                content=item.content,
                chunk_index=item.chunk_index,
                chunk_metadata=item.chunk_metadata,
            )
            for item in response.items
        }
        ranked_chunks: list[RankedChunk] = [
            RankedChunk(chunk_id=item.chunk_id, score=item.score, rank=item.rank)
            for item in response.items
        ]
        context_sections: list[ContextSection] = assemble_rag_context(
            ranked_chunks,
            chunk_lookup,
            max_tokens=payload.max_tokens,
            count_tokens=default_count_tokens,
        )
        return Success(
            UnifiedRagResponse(
                items=response.items,
                context=[
                    RagContextSectionResponse(
                        document_id=section.document_id,
                        contract_index=_context_contract_index(section.chunk_metadata),
                        title=section.title,
                        content=section.content,
                        chunk_indices=section.chunk_indices,
                        chunk_metadata=section.chunk_metadata,
                    )
                    for section in context_sections
                ],
                cache_hit=response.cache_hit,
            )
        )

    @trace_layer("service")
    async def ask(
        self,
        *,
        user_id: str,
        tenant_id: str | None = None,
        allowed_doc_ids: tuple[str, ...] | None = None,
        payload: documents_dto.UnifiedAskRequest,
        require_graphiti_verified: bool,
    ) -> DocumentResult[UnifiedAskResponse]:
        resolved_tenant = _resolve_tenant_id(tenant_id, user_id)
        result = await self._ask(
            user_id=user_id,
            tenant_id=resolved_tenant,
            allowed_doc_ids=allowed_doc_ids,
            payload=payload,
            require_graphiti_verified=require_graphiti_verified,
        )
        scope = tuple(allowed_doc_ids or ())
        document_ids = scope if isinstance(result, Success) else ()
        return cast(
            "DocumentResult[UnifiedAskResponse]",
            await self._record_access(
                result,
                operation="document.read",
                user_id=user_id,
                tenant_id=resolved_tenant,
                requested_document_ids=scope,
                document_ids=document_ids,
                reason=("requested document scope is empty" if allowed_doc_ids == () else None),
                outcome="denied" if allowed_doc_ids == () else None,
            ),
        )

    async def _ask(
        self,
        *,
        user_id: str,
        tenant_id: str | None = None,
        allowed_doc_ids: tuple[str, ...] | None = None,
        payload: documents_dto.UnifiedAskRequest,
        require_graphiti_verified: bool,
    ) -> DocumentResult[UnifiedAskResponse]:
        """Answer through the compile-once retrieval graph (mount-retrieval-graph 4.x).

        The inline plan→retrieve→grade→generate loop this method used to
        re-implement was deleted once the parity suite (28 tests in
        ``tests/unit/features/documents/retrieval/``) pinned the graph's
        behaviour byte-identically against the recorded inline contract. The
        graph is the single implementation; this method resolves the
        server-owned scope, drives one graph invocation, and maps the
        finalised result state onto the response envelope. The audit wrapper
        (``ask``) is unchanged.
        """
        tenant_id = _resolve_tenant_id(tenant_id, user_id)
        explicit_scope = allowed_doc_ids is not None
        scope_ids = tuple(dict.fromkeys(allowed_doc_ids or ()))
        if explicit_scope and not scope_ids:
            # An explicitly empty authorized scope is a deny, not an answering
            # path: the grounded fallback carries the request's evidence policy
            # and no engine is needed (the inline contract, byte-identical).
            return Success(
                UnifiedAskResponse(
                    answer=_FALLBACK_ANSWER,
                    citations=[],
                    confidence="uncertain",
                    evidence_policy=(
                        "graph_indexed_only"
                        if require_graphiti_verified
                        else "mechanically_verified"
                    ),
                )
            )
        engine = self.retrieval_engine
        if engine is None or engine.graph is None:
            # Fail closed (task 4.2): a typed unavailability, never a
            # substitution of the single-shot or any other answering path.
            message = "Retrieval engine is not provisioned in this process"
            raise ServiceUnavailableException(
                message,
                data={"capability": "retrieval_engine"},
            )
        scope_versions = await _resolve_scope_versions(
            self.repo, tenant_id=tenant_id, allowed_doc_ids=scope_ids
        )
        scope_contracts = await _resolve_scope_contracts(
            self.repo, tenant_id=tenant_id, allowed_doc_ids=scope_ids
        )
        try:
            result = await engine.graph.ainvoke(
                {
                    "user_id": user_id,
                    "tenant_id": tenant_id,
                    "allowed_doc_ids": list(scope_ids),
                    "document_versions": list(scope_versions),
                    "document_contracts": list(scope_contracts),
                    "scope_unavailable_documents": resolve_unavailable_scope_documents(
                        scope_ids=scope_ids,
                        scope_versions=scope_versions,
                        scope_contracts=scope_contracts,
                        contract_index=payload.contract_index,
                    ),
                    "scope_bound": explicit_scope,
                    "query": payload.query,
                    "doc_ids_filter": payload.doc_ids_filter,
                    "contract_index": payload.contract_index,
                    "jurisdiction": payload.jurisdiction,
                    "contract_type": payload.contract_type,
                    "clause_type": payload.clause_type,
                    "require_graphiti_verified": require_graphiti_verified,
                    "messages": [],
                    "iteration_count": 0,
                },
                context=RetrievalGraphContext(
                    llm=self.llm,
                    repo=self.repo,
                    redis=None if payload.bypass_cache else self.redis,
                    graphiti=self.graphiti,
                    grader=self.grader,
                    # Per-ask stage budgets (G6 task 4.1); None when disabled.
                    budgets=build_stage_budget_tracker(),
                ),
            )
        except InfrastructureException as exc:
            # The graph's retrieval stage raises a typed infrastructure error
            # where the inline loop returned a typed Failure. The served
            # contract is the Failure (and the failure-path audit the wrapper
            # records), so the boundary converts back — the branch-attributed
            # message and details travel unchanged. `APIException.detail` is
            # the rich envelope; the original message rides inside it.
            rich_detail = exc.detail
            message = (
                str(rich_detail.get("message", str(exc)))
                if isinstance(rich_detail, Mapping)
                else str(rich_detail)
            )
            details = exc.data if isinstance(exc.data, dict) else {}
            return Failure(DocumentDatabaseError(message=message, details=details))
        return Success(_graph_ask_response(result))


def _graph_ask_response(result: Mapping[str, Any]) -> UnifiedAskResponse:
    """Map the graph's finalised result state onto the ask response envelope.

    The single crossing point from retrieval state to the HTTP contract: the
    answer and confidence from the generated (or cache-hit) answer, citations
    with the hydrated-row provenance the graph resolved, the truthful
    coverage envelope, the merged warnings, and the derived evidence policy.
    Field-for-field identical to the inline wrapper the parity suite pins.
    """
    answer: Any = result["generated_answer"]
    coverage: Any = result.get("coverage")
    return UnifiedAskResponse(
        answer=answer.answer,
        citations=[
            LegalCitationResponse(
                chunk_id=citation.chunk_id,
                clause_type=citation.clause_type,
                claim=citation.claim,
                document_id=citation.document_id,
                document_version=citation.document_version,
                contract_index=citation.contract_index,
                pipeline_version=citation.pipeline_version,
                page_no=citation.page_no,
            )
            for citation in result["resolved_citations"]
        ],
        confidence=answer.confidence,
        evidence_policy=result["evidence_policy"],
        coverage=EvidenceCoverage(
            documents_requested=coverage.documents_requested,
            documents_ready=coverage.documents_ready,
            graph_indexed_documents=coverage.graph_indexed_documents,
            graph_unavailable_documents=coverage.graph_unavailable_documents,
        )
        if coverage is not None
        else EvidenceCoverage(),
        warnings=[
            QualityWarningDTO(
                stage=warning.stage,
                code=warning.code,
                message=warning.message,
                severity=warning.severity,
            )
            for warning in result.get("quality_warnings", [])
        ],
        cache_hit=bool(result.get("cache_hit")),
    )


def resolve_unavailable_scope_documents(
    *,
    scope_ids: Sequence[str],
    scope_versions: Sequence[tuple[str, int, str | None]],
    scope_contracts: Sequence[tuple[str, int, tuple[int | None, ...], bool]],
    contract_index: int | None = None,
) -> list[str]:
    """Documents whose authoritative version/contract scope is unavailable.

    Shared by the HTTP ask path and the qna engine's scope mapping: a scoped
    document with no resolved version, a version/contract mismatch, or a
    request contract index the document does not partition into carries the
    ``GRAPH_SCOPE_UNAVAILABLE`` warning and no graph read — never a silent
    fallback to a legacy identity.
    """
    version_by_document = {document_id: version for document_id, version, _ in scope_versions}
    contracts_by_document = {
        document_id: (version, indexes, legacy_null)
        for document_id, version, indexes, legacy_null in scope_contracts
    }
    unavailable: list[str] = []
    for document_id in scope_ids:
        version = version_by_document.get(document_id)
        contract_scope = contracts_by_document.get(document_id)
        if version is None or contract_scope is None or contract_scope[0] != version:
            unavailable.append(document_id)
            continue
        _scope_version, indexes, legacy_null = contract_scope
        if contract_index is not None:
            if contract_index not in indexes:
                unavailable.append(document_id)
        elif not indexes and not legacy_null:
            unavailable.append(document_id)
    return unavailable


def _to_ranked_rows(rows: list[dict[str, object]]) -> list[RankedResultRow]:
    return [
        RankedResultRow(
            chunk_id=str(row["chunk_id"]),
            score=float(cast("float | int | str", row["score"])),
            rank=index,
        )
        for index, row in enumerate(rows, start=1)
    ]


def _build_search_items(
    *,
    fused_results: Sequence[RankedChunk],
    chunk_lookup: dict[str, dict[str, object]],
) -> list[DocumentSearchResultItem]:
    items: list[DocumentSearchResultItem] = []
    for ranked_chunk in fused_results:
        row = chunk_lookup.get(ranked_chunk.chunk_id)
        if row is None:
            continue
        items.append(
            DocumentSearchResultItem(
                chunk_id=ranked_chunk.chunk_id,
                document_id=str(row["document_id"]),
                document_version=int(cast("int | str", row.get("document_version", 1))),
                contract_index=(
                    int(cast("int | str", row["contract_index"]))
                    if isinstance(row.get("contract_index"), (int, str))
                    and not isinstance(row.get("contract_index"), bool)
                    else None
                ),
                pipeline_version=(
                    str(row["pipeline_version"])
                    if row.get("pipeline_version") is not None
                    else None
                ),
                title=str(row["title"]),
                content=str(row["content"]),
                chunk_index=int(cast("int | str", row["chunk_index"])),
                chunk_kind=str(row["chunk_kind"]),
                clause_type=str(row["clause_type"]) if row["clause_type"] is not None else None,
                chunk_metadata=cast("dict[str, object]", row["chunk_metadata"] or {}),
                quality_warnings=_flatten_warnings(row.get("quality_warnings", [])),  # ty: ignore[invalid-argument-type]
                graphiti_verified=bool(row.get("graphiti_verified", False)),
                graph_indexed=bool(row.get("graphiti_verified", False)),
                graph_indexed_at=(
                    str(row["graph_indexed_at"])
                    if row.get("graph_indexed_at") is not None
                    else None
                ),
                score=ranked_chunk.score,
                rank=ranked_chunk.rank,
            )
        )
    return items


async def _resolve_scope_versions(
    repo: DocumentRepository,
    *,
    tenant_id: str,
    allowed_doc_ids: Sequence[str],
) -> tuple[tuple[str, int, str | None], ...]:
    """Read current document identities before consulting a response cache.

    Narrow repository doubles used by offline evaluation predate this method;
    in that case the empty identity is safe (it disables version invalidation
    without weakening the tenant/allowlist fence).  A repository failure also
    returns an empty identity and retrieval remains the source of truth.
    """
    resolver = getattr(repo, "resolve_scope_versions", None)
    if resolver is None or not allowed_doc_ids:
        return ()
    try:
        result = await resolver(tenant_id=tenant_id, document_ids=tuple(allowed_doc_ids))
    except (AttributeError, TypeError):
        return ()
    if isinstance(result, Failure):
        return ()
    try:
        raw = result.unwrap() if isinstance(result, Success) else result
        return tuple(
            (str(doc_id), int(version), str(pipeline) if pipeline is not None else None)
            for doc_id, version, pipeline in raw
        )
    except (AttributeError, TypeError, ValueError):
        return ()


async def _resolve_scope_contracts(
    repo: DocumentRepository,
    *,
    tenant_id: str,
    allowed_doc_ids: Sequence[str],
) -> tuple[tuple[str, int, tuple[int | None, ...], bool], ...]:
    """Resolve exact contract partitions; an unavailable scope fails closed."""
    resolver = getattr(repo, "resolve_scope_contracts", None)
    if resolver is None or not allowed_doc_ids:
        return ()
    try:
        result = await resolver(tenant_id=tenant_id, document_ids=tuple(allowed_doc_ids))
    except (AttributeError, TypeError):
        return ()
    if isinstance(result, Failure):
        return ()
    try:
        raw = result.unwrap() if isinstance(result, Success) else result
        return tuple(
            (
                str(doc_id),
                int(version),
                tuple(index for index in indexes if index is None or int(index) >= 0),
                bool(legacy_null),
            )
            for doc_id, version, indexes, legacy_null in raw
        )
    except (AttributeError, TypeError, ValueError):
        return ()


def _resolve_tenant_id(tenant_id: str | None, user_id: str) -> str:
    """Migration compatibility boundary for pre-membership callers."""
    return tenant_id if tenant_id is not None else personal_tenant_id(user_id)


def _build_cache_key(
    kind: str,
    payload: UnifiedSearchRequest,
    *,
    user_id: str,
    tenant_id: str | None = None,
    allowed_doc_ids: Sequence[str] = (),
    document_versions: Sequence[tuple[str, int, str | None]] = (),
    evidence_policy: EvidencePolicy = "mechanically_verified",
) -> str:
    tenant_id = tenant_id or user_id
    normalized_query = " ".join(payload.query.lower().split())
    filter_json = to_sorted_key_bytes(payload.metadata_filter.model_dump())
    raw = b"|".join(
        [
            kind.encode("utf-8"),
            user_id.encode("utf-8"),
            tenant_id.encode("utf-8"),
            normalized_query.encode("utf-8"),
            filter_json,
            to_sorted_key_bytes({"allowed_doc_ids": sorted(allowed_doc_ids)}),
            to_sorted_key_bytes(
                {
                    "document_versions": sorted(
                        (
                            (str(doc_id), int(version), pipeline)
                            for doc_id, version, pipeline in document_versions
                        ),
                        key=itemgetter(0),
                    )
                }
            ),
            evidence_policy.encode("utf-8"),
            str(payload.limit).encode("utf-8"),
            str(payload.candidate_limit).encode("utf-8"),
            CANONICAL_PIPELINE_VERSION.encode("utf-8"),
        ]
    )
    return "documents:" + hashlib.sha256(raw).hexdigest()


def _build_answer_cache_key(
    *,
    user_id: str,
    tenant_id: str | None = None,
    query: str,
    doc_ids_filter: list[str],
    jurisdiction: str | None,
    contract_type: str | None,
    clause_type: str | None,
    require_graphiti_verified: bool,
    contract_index: int | None = None,
    allowed_doc_ids: Sequence[str] = (),
    evidence_policy: EvidencePolicy | None = None,
    document_versions: Sequence[tuple[str, int, str | None]] = (),
) -> str:
    tenant_id = tenant_id or user_id
    raw = to_sorted_key_bytes(
        {
            "user_id": user_id,
            "tenant_id": tenant_id,
            "query": " ".join(query.lower().split()),
            "doc_ids_filter": sorted(doc_ids_filter),
            "allowed_doc_ids": sorted(allowed_doc_ids),
            "document_versions": sorted(
                (
                    (str(doc_id), int(version), pipeline)
                    for doc_id, version, pipeline in document_versions
                ),
                key=itemgetter(0),
            ),
            "jurisdiction": jurisdiction,
            "contract_type": contract_type,
            "contract_index": contract_index,
            "clause_type": clause_type,
            "require_graphiti_verified": require_graphiti_verified,
            "evidence_policy": evidence_policy
            or ("graph_indexed_only" if require_graphiti_verified else "mechanically_verified"),
            "pipeline_version": CANONICAL_PIPELINE_VERSION,
        },
    )
    return "documents:answer:" + hashlib.sha256(raw).hexdigest()


def _batched[T](values: Sequence[T], batch_size: int) -> list[Sequence[T]]:
    return [values[index : index + batch_size] for index in range(0, len(values), batch_size)]


def _flatten_warnings(
    raw_groups: list[Any],
) -> list[QualityWarningDTO]:  # ponytail: SQL results are untyped dicts
    warnings: list[QualityWarningDTO] = []
    for group in raw_groups:
        if isinstance(group, list):
            warnings.extend(
                QualityWarningDTO.model_validate(warning)
                for warning in group
                if isinstance(warning, dict)
            )
        elif isinstance(group, dict):
            warnings.append(QualityWarningDTO.model_validate(group))
    return warnings


def _context_contract_index(metadata: dict[str, object]) -> int | None:
    """Return a contract index only when a context section is contract-local."""
    value = metadata.get("contract_index")
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value if value >= 0 else None
    if isinstance(value, str):
        try:
            parsed = int(value)
        except ValueError:
            return None
        return parsed if parsed >= 0 else None
    return None
