"""Nodes for the unified-chunk-backed legal retrieval graph."""

from __future__ import annotations

import asyncio
import hashlib
import re
import time
from contextlib import asynccontextmanager
from operator import itemgetter
from typing import TYPE_CHECKING, Protocol, cast, runtime_checkable

import opentelemetry.trace as otel_trace
from graphiti_core.errors import GraphitiError
from langchain_core.exceptions import LangChainException
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from pydantic import ValidationError
from returns.result import Failure

from app.ai.embeddings import EmbeddingTaskType, embed_text
from app.ai.generation import TextGenerationRequest
from app.ai.graph.retry import TransientExternalError, retry_immediate
from app.config import get_settings
from app.features.documents.retrieval import row_hydration
from app.features.documents.retrieval.reranking.reranker import get_configured_reranker
from app.features.documents.retrieval.service import _is_graphiti_branch_failure
from app.kernel.result import log_expected_failure
from app.observability.degradation import GRADER_FALLBACK, GRAPHITI_UNAVAILABLE, record_degradation
from app.observability.logging import logger
from app.retrieval.edge_expansion import EdgeSeedChunk, expand_along_edges
from app.retrieval.graphiti.client import fetch_relationship_edges
from app.retrieval.graphiti.grouping import GraphitiDocumentScope, fanout_graphiti_group_ids
from app.retrieval.token_counter import count_tokens as default_count_tokens
from app.shared.langchain_layer import render_prompt_sections, serialize_to_toon
from app.utils import InfrastructureException, to_sorted_key_bytes

from .budgets import (
    STAGE_EDGE_GROUNDING,
    STAGE_GENERATE,
    STAGE_GRADE,
    STAGE_PLAN,
    STAGE_RERANK,
    STAGE_RETRIEVE_FUSION,
    STAGE_SOURCE_IDENTIFIER,
)
from .state import (  # noqa: TC001 — RetrievalState resolves at runtime for branch schemas
    CachedAskEnvelope,
    Citation,
    ContextGrade,
    CoverageEnvelope,
    GeneratedAnswer,
    GroundedCitation,
    QueryPlan,
    RetrievalGraphContext,
    RetrievalState,
    RetrievedChunk,
    SourceCriteria,
    StageUsage,
    StageWarning,
)

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Awaitable, Callable, Mapping, Sequence
    from typing import Any

    from app.ai.generation import TextGenerationStructuredResult
    from app.features.documents.adapters.persistence.postgres.document_repository import (
        DocumentRepository,
    )
    from app.retrieval.edge_expansion import EdgeContextBlock, EdgeExpansion
    from app.retrieval.token_counter import CountTokens

    from .budgets import StageBudgetTracker, StageBudgetViolation

# The rerank stage's top-k: how many fused candidates the cross-encoder sees
# reordered, and how many survive it. The flag-off pass-through truncates to
# the same limit so downstream shape is identical with and without the stage.
RERANK_LIMIT = 5

_QUERY_ANALYZER_SYSTEM_PROMPT = render_prompt_sections(
    ("IDENTITY", "You are a legal retrieval query planning engine."),
    (
        "OBJECTIVE",
        "Analyze the legal retrieval query and produce a QueryPlan that maximizes grounded retrieval.",
    ),
    (
        "CONTEXT POLICY",
        "Use recent conversation messages only to resolve references and clarify retrieval intent.",
    ),
    (
        "EXECUTION POLICY",
        "Rewrite coreferences using the conversation, decompose multi-part questions, choose route "
        "hybrid_postgres, graph_neo4j, or both, and choose vector_weight and keyword_weight. "
        "Exact clause-reference queries should favor BM25; conceptual obligation or risk questions should favor vector search.",
    ),
    ("CONSTRAINTS", "Return only QueryPlan."),
)

# The grade seam's system prompt, verbatim from the inline `_grade_context`
# contract the parity suite pins: the injected-grader request and the
# seam-absent degrade both use this exact string.
_GRADE_SEAM_SYSTEM_PROMPT = "You are a retrieval sufficiency grader. Return only ContextGrade."

_GENERATOR_SYSTEM_PROMPT = render_prompt_sections(
    ("IDENTITY", "You are a grounded legal answer generator."),
    (
        "OBJECTIVE",
        "Answer the user's question using only the retrieved chunks.",
    ),
    (
        "CONTEXT POLICY",
        "Treat retrieved chunks as the only admissible basis for factual claims.",
    ),
    (
        "CONSTRAINTS",
        "Every factual claim must cite exact chunk_id and clause_type in the citations list. "
        "Return only GeneratedAnswer.",
    ),
    (
        "UNCERTAINTY POLICY",
        "If support is weak, set confidence to uncertain and avoid stronger claims than the evidence supports.",
    ),
)

# Verbatim from the served inline `ask` fallback contract (`_FALLBACK_ANSWER`
# in `application/queries/documents.py`): the parity suite pins byte-identical
# fallback answers, so the two constants must carry the same text.
FALLBACK_ANSWER = (
    "I do not have enough grounded document context to answer this reliably. "
    "Please narrow the question or ingest the relevant document sections."
)

# Edge-aware context expansion bounds, verbatim from the inline ask contract:
# additive and small by construction — at most a few server-authored grounding
# lines per reranked chunk, never a second retrieval pass, and only for
# queries that actually touch edges.
_EDGE_EXPANSION_MAX_TOKENS: int = 256
_EDGE_EXPANSION_MAX_BLOCKS_PER_SEED: int = 4
_EDGE_FETCH_LIMIT: int = 32

# The grade loop's two-iteration bound (`max_attempts` includes the first
# attempt, exactly as the inline pipeline's `_ASK_MAX_ATTEMPTS` does).
_GRADE_LOOP_MAX_ATTEMPTS: int = 2


@asynccontextmanager
async def _stage_span(
    stage: str,
    tracker: StageBudgetTracker | None = None,
    document_class: str = "unknown",
) -> AsyncIterator[list[StageBudgetViolation]]:
    """Emit one span attributed to a retrieval-graph stage, with its duration.

    The OpenTelemetry span carries the stage attribution; a structured
    ``retrieval_stage_span`` log event mirrors it so the slice is observable
    (and testable) without a configured OTel exporter. First slice of G6.

    When a stage-budget tracker is supplied (G6 task 4.1), the exit duration is
    checked against the stage's latency budget; a breach is recorded by the
    tracker (degradation event + sustained-alert window) and yielded back to
    the node as a violation so the ask envelope names the stage that ran over.
    """
    started = time.perf_counter()
    tracer = otel_trace.get_tracer(__name__)
    violations: list[StageBudgetViolation] = []
    with tracer.start_as_current_span(f"retrieval_graph.{stage}", attributes={"stage": stage}):
        try:
            yield violations
        finally:
            duration_ms = round((time.perf_counter() - started) * 1000, 2)
            logger.bind(stage=stage, duration_ms=duration_ms).info("retrieval_stage_span")
            if tracker is not None:
                violation = tracker.check_latency(stage, duration_ms, document_class=document_class)
                if violation is not None:
                    violations.append(violation)


def _stage_usage(stage: str, calls: int) -> dict[str, object]:
    """One stage-usage delta; nodes append, the state channel concatenates."""
    return {"stage_usage": [StageUsage(stage=stage, calls=calls)]}


async def _with_redis_deadline(operation: str, call: Callable[[], Awaitable[Any]]) -> Any | None:
    """Run one Redis cache call under the explicit Redis deadline (harden 5.2).

    The ask cache is an optimization: a deadline miss or transport failure
    degrades to a cache miss — never to an empty retrieval or a failed ask.
    The deadline is DOCUMENTS_RETRIEVAL_REDIS_TIMEOUT_SECONDS.
    """
    settings = get_settings()
    try:
        async with asyncio.timeout(settings.DOCUMENTS_RETRIEVAL_REDIS_TIMEOUT_SECONDS):
            return await call()
    except Exception as exc:  # noqa: BLE001 — cache degradation by contract
        logger.bind(operation=operation, error=type(exc).__name__).warning(
            "retrieval_cache_degraded_to_miss"
        )
        return None


_BUDGET_WARNING_CODE = "STAGE_BUDGET_EXHAUSTED"


def _violation_warning(violation: StageBudgetViolation) -> StageWarning:
    """The typed degraded-outcome record: the envelope names stage and budget."""
    return StageWarning(
        stage=violation.stage,
        code=_BUDGET_WARNING_CODE,
        message=(
            f"Stage '{violation.stage}' exhausted its {violation.budget_kind} budget "
            f"({violation.observed:g} > {violation.limit:g}); the ask degraded instead of "
            "continuing unbounded."
        ),
        severity="warning",
    )


def _violation_state(violations: Sequence[StageBudgetViolation]) -> dict[str, object]:
    """State delta carrying the typed budget warnings, when any fired."""
    if not violations:
        return {}
    return {"stage_warnings": [_violation_warning(violation) for violation in violations]}


def _charge_stage(
    context: RetrievalGraphContext, stage: str, document_class: str = "unknown"
) -> list[StageBudgetViolation]:
    """Pre-charge the stage's call budget; a refusal stops the unbounded run."""
    tracker = context.budgets
    if tracker is None:
        return []
    violation = tracker.try_charge(stage, document_class=document_class)
    return [violation] if violation is not None else []


def _state_document_class(state: RetrievalState) -> str:
    """Best-effort document-class attribution from the hydrated rows."""
    for row in state.get("retrieved_rows", []):
        kind = row.get("chunk_kind")
        if kind:
            return str(kind)
    return "unknown"


def _invocation_context(runtime: Any) -> RetrievalGraphContext:
    """Read the per-invocation collaborators from the LangGraph runtime.

    Every request-scoped collaborator arrives here — the `compiled-graph-lifecycle`
    contract — so no node factory captures a repository, session, client, or
    model at compile time. A missing context (a direct node call in tests)
    degrades to an empty context whose ``None`` collaborators trigger each
    node's fail-closed default.
    """
    context = getattr(runtime, "context", None) if runtime is not None else None
    if isinstance(context, RetrievalGraphContext):
        return context
    return RetrievalGraphContext()


def _structured(llm: Any, schema: type[Any]) -> Any:
    """Derive a structured-output runnable from the invocation-context model."""
    if hasattr(llm, "with_structured_output"):
        return llm.with_structured_output(schema)
    return llm


def make_query_analyzer_node() -> Callable[[RetrievalState, Any], Awaitable[dict[str, object]]]:
    async def query_analyzer_node(state: RetrievalState, runtime: Any = None) -> dict[str, object]:
        context = _invocation_context(runtime)
        document_class = _state_document_class(state)
        # Pre-call charge (G6 task 4.1): a refusal degrades to the default plan
        # instead of spending another unbounded planning call.
        refusals = _charge_stage(context, STAGE_PLAN, document_class)
        if refusals:
            query = state["query"]
            return {
                "query_plan": QueryPlan(rewritten_query=query, sub_queries=[query]),
                "cache_hit": False,
                **_stage_usage(STAGE_PLAN, 0),
                **_violation_state(refusals),
            }
        async with _stage_span(STAGE_PLAN, context.budgets, document_class) as violations:
            result = await _plan_and_cache(state, context)
        return {**result, **_stage_usage(STAGE_PLAN, 1), **_violation_state(violations)}

    return query_analyzer_node


async def _plan_and_cache(
    state: RetrievalState, context: RetrievalGraphContext
) -> dict[str, object]:
    query_llm = _structured(context.llm, QueryPlan)
    query = state["query"]
    plan_input = serialize_to_toon(
        {
            "query": query,
            "messages": [str(message.content) for message in state.get("messages", [])[-8:]],
            "doc_ids_filter": state.get("doc_ids_filter", []),
            "iteration_count": state.get("iteration_count", 0),
            "rewrite_suggestion": getattr(state.get("context_grade"), "rewrite_suggestion", None),
        }
    )
    messages: list[SystemMessage | HumanMessage] = [
        SystemMessage(content=_QUERY_ANALYZER_SYSTEM_PROMPT),
        HumanMessage(content=plan_input),
    ]
    try:
        raw_plan = await retry_immediate(
            lambda: query_llm.ainvoke(cast("list[Any]", messages)),
            label="gemini_query_analyzer",
        )
        plan = _normalize_plan(QueryPlan.model_validate(raw_plan))
    # Both routes into this branch (the C6 contract): a deterministic
    # framework failure arrives unwrapped; a transient one arrives retried
    # and, once the budget is spent, as the boundary's transient type with
    # the original reachable through its cause. Catching only the first
    # left this degradation branch dead in production.
    except (LangChainException, TransientExternalError) as exc:
        exc.add_note(f"query={query[:80]}, operation=query_analyzer")
        logger.bind(query=query[:80], operation="query_analyzer", error=str(exc)).warning(
            "query_analyzer_failed_using_default"
        )
        plan = QueryPlan(rewritten_query=query, sub_queries=[query])

    answer_cache_key = _answer_cache_key(state)
    redis = context.redis
    if redis is not None:
        cached = await _with_redis_deadline(
            "ask_plan_cache_read", lambda: redis.get(answer_cache_key)
        )
        if cached:
            try:
                envelope = CachedAskEnvelope.model_validate_json(str(cached))
            except ValidationError:
                # A pre-envelope cache entry (or a foreign payload) is a miss,
                # never a crash: the derivation recomputes and rewrites.
                envelope = None
            if envelope is not None:
                return {
                    "query_plan": plan,
                    "cache_hit": True,
                    "cached_envelope": envelope,
                    "answer_cache_key": answer_cache_key,
                }

    return {"query_plan": plan, "cache_hit": False, "answer_cache_key": answer_cache_key}


_SOURCE_IDENTIFIER_SYSTEM_PROMPT = render_prompt_sections(
    ("IDENTITY", "You are a corpus-narrowing source identifier."),
    (
        "OBJECTIVE",
        "Extract the cheap-metadata narrowing the query names — jurisdiction, document kind, "
        "matter — so retrieval searches a smaller corpus. Name only what the query states; "
        "when it names nothing, return empty lists.",
    ),
    (
        "FEW-SHOT EXAMPLES",
        "Query: 'What termination rights survive under our India services agreements?' "
        "-> {jurisdictions: ['India'], document_kinds: ['contracts'], matters: []}. "
        "Query: 'What does section 73 provide about compensation for breach?' "
        "-> {jurisdictions: [], document_kinds: ['statutes'], matters: []}. "
        "Query: 'Summarise the key risks in this filing.' "
        "-> {jurisdictions: [], document_kinds: [], matters: []}.",
    ),
    ("CONSTRAINTS", "Return only SourceCriteria."),
)

#: Upper bound on a resolved allowlist. More identifiers than this narrow
#: nothing and only bloat the filter parameters, so an over-wide resolution is
#: treated as "cannot narrow" and retrieval proceeds unconstrained.
SOURCE_ALLOWLIST_MAX_IDS = 100


def _intersect_doc_ids(*, request_ids: list[str], allowlist: list[str]) -> list[str]:
    """Combine the request document filter with the identifier allowlist."""
    if not allowlist:
        return []
    if not request_ids:
        return allowlist
    return [doc_id for doc_id in allowlist if doc_id in request_ids]


def make_source_identifier_node() -> Callable[[RetrievalState, Any], Awaitable[dict[str, object]]]:
    """Narrow the corpus to a document allowlist before retrieval runs.

    On a retry iteration the previous narrowing is dropped instead: a wrong
    allowlist that excluded the answer is unrecoverable any other way, and the
    iteration cap (not the allowlist) is what bounds the loop.
    """

    async def source_identifier_node(
        state: RetrievalState, runtime: Any = None
    ) -> dict[str, object]:
        context = _invocation_context(runtime)
        document_class = _state_document_class(state)
        refusals = _charge_stage(context, STAGE_SOURCE_IDENTIFIER, document_class)
        if refusals:
            return {
                "query_plan": state["query_plan"],
                **_stage_usage(STAGE_SOURCE_IDENTIFIER, 0),
                **_violation_state(refusals),
            }
        async with _stage_span(
            STAGE_SOURCE_IDENTIFIER, context.budgets, document_class
        ) as violations:
            result = await _identify_sources(
                state, _structured(context.llm, SourceCriteria), context.repo
            )
        return {**result, **_violation_state(violations)}

    return source_identifier_node


async def _identify_sources(
    state: RetrievalState, identifier_llm: Any, repo: DocumentRepository
) -> dict[str, object]:
    plan: QueryPlan = state["query_plan"]
    server_scope = list(state.get("allowed_doc_ids") or [])
    if state.get("scope_bound") and not server_scope:
        # An explicit empty capability is not equivalent to "all docs".
        # Keep the graph alive for its envelope, but make every retrieval
        # leg return no evidence.
        return {
            "query_plan": plan.model_copy(update={"allowlist": []}),
            **_stage_usage("source_identifier", 0),
        }
    if state.get("iteration_count", 0) > 0:
        if not plan.allowlist:
            return {"query_plan": plan, **_stage_usage("source_identifier", 0)}
        logger.bind(operation="source_identifier", iteration=state.get("iteration_count")).info(
            "source_allowlist_widened_on_retry"
        )
        # Widen only the model-inferred narrowing.  The server-owned
        # allowlist remains authoritative and is re-applied by the
        # hybrid node; never widen it on a retry.
        return {
            "query_plan": plan.model_copy(update={"allowlist": []}),
            **_stage_usage("source_identifier", 0),
        }

    query = plan.rewritten_query
    messages: list[SystemMessage | HumanMessage] = [
        SystemMessage(content=_SOURCE_IDENTIFIER_SYSTEM_PROMPT),
        HumanMessage(content=query),
    ]
    try:
        raw_criteria = await retry_immediate(
            lambda: identifier_llm.ainvoke(cast("list[Any]", messages)),
            label="gemini_source_identifier",
        )
        criteria = SourceCriteria.model_validate(raw_criteria)
    except Exception as exc:  # noqa: BLE001 — cannot narrow means unconstrained, not failed
        exc.add_note(f"query={query[:80]}, operation=source_identifier")
        logger.bind(query=query[:80], operation="source_identifier", error=str(exc)).warning(
            "source_identifier_failed_proceeding_unconstrained"
        )
        return {"query_plan": plan, **_stage_usage("source_identifier", 1)}

    if not (criteria.jurisdictions or criteria.document_kinds or criteria.matters):
        return {"query_plan": plan, **_stage_usage("source_identifier", 1)}
    resolved = await repo.find_document_ids_by_metadata(
        user_id=state["user_id"],
        tenant_id=state.get("tenant_id"),
        jurisdictions=criteria.jurisdictions,
        document_kinds=criteria.document_kinds,
        matters=criteria.matters,
        limit=SOURCE_ALLOWLIST_MAX_IDS + 1,
    )
    if isinstance(resolved, Failure):
        error = resolved.failure()
        log_expected_failure(error, operation="source_allowlist_resolve")
        return {"query_plan": plan, **_stage_usage("source_identifier", 1)}
    allowlist = resolved.unwrap()
    if len(allowlist) > SOURCE_ALLOWLIST_MAX_IDS:
        return {"query_plan": plan, **_stage_usage("source_identifier", 1)}
    return {
        "query_plan": plan.model_copy(update={"allowlist": allowlist}),
        **_stage_usage("source_identifier", 1),
    }


def make_graph_retrieval_node() -> Callable[[RetrievalState, Any], Awaitable[dict[str, object]]]:
    async def graph_retrieval_node(  # noqa: PLR0914 — scope fan-out keeps identity together
        state: RetrievalState, runtime: Any = None
    ) -> dict[str, object]:
        graphiti = _invocation_context(runtime).graphiti
        if graphiti is None:
            return {"graph_chunk_ids": []}
        # Graphiti's historical user group is not an authorization boundary:
        # it can contain documents from several tenant scopes.  The SQL leg
        # must not be the first place that discards those results.  Only the
        # server-issued document capability may reach this optional graph leg;
        # an unscoped request therefore degrades to no graph narrowing.
        server_doc_ids = list(state.get("allowed_doc_ids") or [])
        if state.get("scope_bound") and not server_doc_ids:
            return {"graph_chunk_ids": []}
        if not server_doc_ids:
            return {"graph_chunk_ids": []}
        request_doc_ids = list(state.get("doc_ids_filter") or [])
        document_ids = _intersect_doc_ids(
            request_ids=request_doc_ids,
            allowlist=server_doc_ids,
        )
        if not document_ids or not state.get("tenant_id"):
            return {"graph_chunk_ids": []}
        versions = {
            document_id: version for document_id, version, _ in state.get("document_versions", [])
        }
        scoped_contracts = {
            document_id: (indexes, legacy_null)
            for document_id, _version, indexes, legacy_null in state.get("document_contracts", [])
        }
        scopes = []
        for document_id in document_ids:
            version = versions.get(document_id)
            if version is None or document_id in state.get("scope_unavailable_documents", []):
                # A missing version/contract grant is an unavailable graph
                # scope, never an invitation to probe the legacy v1 group.
                continue
            explicit_contract = state.get("contract_index")
            scoped = scoped_contracts.get(document_id)
            if scoped is None:
                continue
            indexes, legacy_null = scoped
            if explicit_contract is not None:
                if explicit_contract not in indexes:
                    continue
                indexes, legacy_null = (explicit_contract,), False
            scopes.append(
                GraphitiDocumentScope(
                    tenant_id=state["tenant_id"],
                    document_id=document_id,
                    document_version=version,
                    contract_indexes=indexes,
                    legacy_null_contract=legacy_null,
                )
            )
        group_ids = fanout_graphiti_group_ids(scopes)
        if not group_ids:
            return {"graph_chunk_ids": []}
        plan = state["query_plan"]
        try:
            raw_results = await retry_immediate(
                lambda: graphiti.search(
                    query=plan.rewritten_query,
                    group_ids=group_ids,
                    num_results=20,
                ),
                label="graphiti_retrieval_search",
            )
        # Same C6 contract as above: a retry-exhausted transient failure
        # arrives as the boundary's type, not as `GraphitiError`.
        except (GraphitiError, TransientExternalError) as exc:
            exc.add_note(f"query={plan.rewritten_query[:80]}, operation=graph_retrieval")
            logger.bind(
                query=plan.rewritten_query[:80], operation="graph_retrieval", error=str(exc)
            ).warning("graph_retrieval_failed")
            return {"graph_chunk_ids": []}

        chunk_ids: list[str] = []
        for result in raw_results or []:
            source_description = getattr(result, "source_description", "") or ""
            content = getattr(result, "content", "") or getattr(result, "episode_body", "") or ""
            chunk_ids.extend(_extract_postgres_chunk_ids(source_description))
            chunk_ids.extend(_extract_postgres_chunk_ids(content))
        return {"graph_chunk_ids": sorted(set(chunk_ids))}

    return graph_retrieval_node


def _document_version_for(
    document_versions: list[tuple[str, int, str | None]], document_id: str
) -> int:
    """Resolve the server-issued version for a scoped document.

    Version one is the compatibility identity for legacy states that predate
    version grants; it remains encoded in the canonical group ID and never
    falls back to a bare document group.
    """
    return next(
        (version for candidate, version, _ in document_versions if candidate == document_id),
        1,
    )


def make_hybrid_retrieval_node() -> Callable[[RetrievalState, Any], Awaitable[dict[str, object]]]:
    async def hybrid_retrieval_node(
        state: RetrievalState, runtime: Any = None
    ) -> dict[str, object]:
        context = _invocation_context(runtime)
        # An explicitly empty authorized scope is the fallback envelope, not a
        # wide search (task 1.3): no retrieval leg executes at all, matching
        # the served inline `ask`'s immediate empty-scope return.
        server_doc_ids_early = list(state.get("allowed_doc_ids") or [])
        if state.get("scope_bound") and not server_doc_ids_early:
            return {
                "retrieved_chunks": [],
                "retrieved_rows": [],
                **_stage_usage("retrieve_fusion", 0),
            }
        document_class = _state_document_class(state)
        refusals = _charge_stage(context, STAGE_RETRIEVE_FUSION, document_class)
        if refusals:
            return {
                "retrieved_chunks": [],
                "retrieved_rows": [],
                **_stage_usage(STAGE_RETRIEVE_FUSION, 0),
                **_violation_state(refusals),
            }
        async with _stage_span(
            STAGE_RETRIEVE_FUSION, context.budgets, document_class
        ) as violations:
            result = await _retrieve_fused_state(state, context)
        return {**result, **_violation_state(violations)}

    return hybrid_retrieval_node


async def _retrieve_fused_state(  # noqa: PLR0914 — scope, fusion, and outage attribution stay together
    state: RetrievalState,
    context: RetrievalGraphContext,
) -> dict[str, object]:
    plan = state["query_plan"]
    redis = context.redis
    # The invocation context is provisioned per ask (the engine opens the
    # session); a repository-less context is a provisioning bug, not a
    # retrievable state.
    repo = cast("DocumentRepository", context.repo)
    graphiti = context.graphiti
    # `QUERY`, not `DOCUMENT`. This is the side of the asymmetry that was never declared:
    # the prior helper passed no task type at all, so a query vector was drawn from the
    # document projection and compared against stored vectors drawn from the same one.
    # Mutually consistent, and both wrong — which is why nothing ever errored.
    embedding = await retry_immediate(
        lambda: embed_text(
            plan.rewritten_query,
            task_type=EmbeddingTaskType.QUERY,
            redis=redis,
        ),
        label="gemini_query_embedding",
    )
    chunk_ids = state.get("graph_chunk_ids") or None
    # Local imports (noqa: PLC0415): `documents.service` imports this package at
    # module load, so top-level imports would close a cycle. The fused
    # retrieval helper lives with the other retrieval-region policy in the
    # service module, and this node reaches the same shared path as `ask`.
    from app.features.documents.adapters.persistence.postgres.document_repository import (  # noqa: PLC0415
        build_search_filter_params,
    )
    from app.features.documents.application.queries.documents import (  # noqa: PLC0415
        lookup_to_rows,
        retrieve_fused,
    )
    from app.features.documents.domain.constants import (  # noqa: PLC0415
        HYBRID_CANDIDATE_LIMIT,
        RRF_WEIGHT_GRAPHITI,
    )

    # `user_id` is not a translation of an old argument — the reader this replaces had no
    # tenant predicate at all, so every fused search read across all owners and was held
    # back only by the caller never passing another user's chunk ids. The unified query
    # scopes on the parent document's owner, which is why the state field is required here
    # rather than optional.
    request_doc_ids = list(state.get("doc_ids_filter") or [])
    server_doc_ids = list(state.get("allowed_doc_ids") or [])
    scoped_doc_ids = (
        _intersect_doc_ids(request_ids=request_doc_ids, allowlist=server_doc_ids)
        if state.get("scope_bound")
        else request_doc_ids
    )
    # Optional relationship-graph fusion branch (retrieval-measurement-gate):
    # the same registration contract the inline ask path uses — the branch is
    # joined through the shared `retrieve_fused` registry only when the flag is
    # set AND a Graphiti client is wired, and no graph read runs while disabled.
    graphiti_branch = graphiti if get_settings().DOCUMENTS_GRAPHITI_FUSION_ENABLED else None
    filter_params = build_search_filter_params(
        metadata_filter={
            # The request filter and the identifier allowlist both constrain:
            # intersect when both name documents, otherwise take whichever
            # is present. A conflicting inference loses to the explicit
            # request filter rather than silently emptying retrieval.
            "document_ids": (
                _intersect_doc_ids(
                    request_ids=scoped_doc_ids,
                    allowlist=list(plan.allowlist),
                )
                if plan.allowlist
                else scoped_doc_ids
            ),
            "scope_bound": bool(state.get("scope_bound")),
            "chunk_ids": list(chunk_ids or []),
            "jurisdiction": state.get("jurisdiction") or plan.jurisdiction,
            "contract_type": state.get("contract_type") or plan.contract_type,
            "contract_index": state.get("contract_index"),
            "clause_type": state.get("clause_type"),
            # Evidence policy (task 1.3): `/legal/ask`-shaped requests filter
            # on graph-indexed evidence here, exactly as the inline loop does.
            "require_graphiti_verified": bool(state.get("require_graphiti_verified", False)),
        }
    )
    rows_result = await retry_immediate(
        lambda: retrieve_fused(
            repo=repo,
            user_id=state["user_id"],
            tenant_id=state.get("tenant_id"),
            query_text=plan.rewritten_query,
            query_embedding=embedding,
            candidate_limit=HYBRID_CANDIDATE_LIMIT,
            limit=20,
            filter_params=filter_params,
            weights=[plan.keyword_weight, plan.vector_weight]
            + ([RRF_WEIGHT_GRAPHITI] if graphiti_branch is not None else []),
            bm25_threshold=plan.bm25_threshold,
            exact_phrase=plan.exact_phrase,
            graphiti=graphiti_branch,
            document_versions=state.get("document_versions", []),
            document_contracts=state.get("document_contracts", []),
        ),
        label="postgres_fused_search",
    )
    graphiti_degraded = False
    if isinstance(rows_result, Failure):
        error = rows_result.failure()
        if _is_graphiti_branch_failure(error):
            # Harden 5.2 / legal-corpus-retrieval: a Graphiti outage is a
            # coverage degradation, not an ask failure. The canonical-only
            # fusion reruns (same branch registry, same weights minus the
            # graph leg) so canonical chunks stand and the coverage signal
            # degrades honestly — Graphiti is never legal truth.
            record_degradation(GRAPHITI_UNAVAILABLE, surface="ask_hybrid_node")
            graphiti_degraded = True
            rows_result = await retrieve_fused(
                repo=repo,
                user_id=state["user_id"],
                tenant_id=state.get("tenant_id"),
                query_text=plan.rewritten_query,
                query_embedding=embedding,
                candidate_limit=HYBRID_CANDIDATE_LIMIT,
                limit=20,
                filter_params=filter_params,
                weights=[plan.keyword_weight, plan.vector_weight],
                bm25_threshold=plan.bm25_threshold,
                exact_phrase=plan.exact_phrase,
                graphiti=None,
                document_versions=state.get("document_versions", []),
                document_contracts=state.get("document_contracts", []),
            )
        if isinstance(rows_result, Failure):
            error = rows_result.failure()
            log_expected_failure(error, operation="postgres_fused_search")
            raise InfrastructureException(
                detail=error.message,
                error_code=error.code,
                retryable=error.retryable,
                data=error.details,
            )
    fused, lookup = rows_result.unwrap()
    rows = lookup_to_rows(fused, lookup)
    return {
        "retrieved_chunks": [_row_to_chunk(row) for row in rows],
        # The hydrated rows feed the warning envelope, coverage counts, and
        # edge-span resolution (task 1.4 / task 1.1).
        "retrieved_rows": rows,
        **_stage_usage("retrieve_fusion", 1),
        **(
            {
                "stage_warnings": [
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
            }
            if graphiti_degraded
            else {}
        ),
    }


def make_reranker_node() -> Callable[[RetrievalState, Any], Awaitable[dict[str, object]]]:
    async def reranker_node(state: RetrievalState, runtime: Any = None) -> dict[str, object]:
        # A context-supplied reranker wins; otherwise the process-configured
        # one resolves here, per invocation (never captured at compile).
        context = _invocation_context(runtime)
        resolved = context.reranker or get_configured_reranker()
        plan: QueryPlan = state["query_plan"]
        chunks: list[RetrievedChunk] = state.get("retrieved_chunks", [])[:20]
        if resolved is None:
            # Flag-off (or nothing configured): fused order passes through
            # truncated to the rerank limit — the same shape a provider
            # degradation yields — without spending the stage or the call.
            return {
                "reranked_chunks": chunks[:RERANK_LIMIT],
                **_stage_usage(STAGE_RERANK, 0),
            }
        document_class = _state_document_class(state)
        refusals = _charge_stage(context, STAGE_RERANK, document_class)
        if refusals:
            return {
                "reranked_chunks": [],
                **_stage_usage(STAGE_RERANK, 0),
                **_violation_state(refusals),
            }
        async with _stage_span(STAGE_RERANK, context.budgets, document_class) as violations:
            reranked: list[RetrievedChunk] = await resolved.rerank(
                plan.rewritten_query, chunks, limit=RERANK_LIMIT
            )
        return {
            "reranked_chunks": reranked,
            **_stage_usage(STAGE_RERANK, 1),
            **_violation_state(violations),
        }

    return reranker_node


def make_post_process_node() -> Callable[[RetrievalState, Any], Awaitable[dict[str, object]]]:
    """Deduplicate reranked chunks and assemble document-ordered context.

    A chunk returned by two branches reaches generation once, and chunks of
    one document read in ascending in-document order rather than relevance
    order. Assembly reuses `assemble_rag_context` — no second implementation.
    Both the reordered chunk list (what the grader and generator read) and the
    assembled sections are returned so the two stay consistent.
    """

    async def post_process_node(state: RetrievalState, runtime: Any = None) -> dict[str, object]:
        context = _invocation_context(runtime)
        # Local imports (noqa: PLC0415): `documents` modules import this
        # package at module load, so top-level imports would close a cycle.
        from app.features.documents.domain.constants import (  # noqa: PLC0415
            DEFAULT_RAG_TOKEN_BUDGET,
        )
        from app.features.documents.retrieval.context_assembly import (  # noqa: PLC0415
            SearchChunkRecord,
            assemble_rag_context,
        )
        from app.retrieval.fusion import RankedChunk  # noqa: PLC0415

        max_tokens = context.post_process_max_tokens or DEFAULT_RAG_TOKEN_BUDGET
        count_tokens = context.count_tokens or default_count_tokens

        chunks: list[RetrievedChunk] = state.get("reranked_chunks", [])
        unique: dict[str, RetrievedChunk] = {}
        for chunk in chunks:
            unique.setdefault(chunk.chunk_id, chunk)
        grouped: dict[str, list[RetrievedChunk]] = {}
        for chunk in unique.values():
            grouped.setdefault(chunk.parent_doc_id, []).append(chunk)
        ordered: list[RetrievedChunk] = [
            chunk
            for document_chunks in grouped.values()
            for chunk in sorted(document_chunks, key=lambda item: item.chunk_index)
        ]
        ranked = [
            RankedChunk(chunk_id=chunk.chunk_id, score=chunk.score, rank=index)
            for index, chunk in enumerate(ordered, start=1)
        ]
        lookup = {
            chunk.chunk_id: SearchChunkRecord(
                document_id=chunk.parent_doc_id,
                title="",
                content=chunk.chunk_text,
                chunk_index=chunk.chunk_index,
                chunk_metadata=dict(chunk.metadata_),
            )
            for chunk in ordered
        }
        sections = assemble_rag_context(
            ranked, lookup, max_tokens=max_tokens, count_tokens=count_tokens
        )
        included_chunk_ids = {chunk_id for section in sections for chunk_id in section.chunk_ids}
        budgeted_chunks = [chunk for chunk in ordered if chunk.chunk_id in included_chunk_ids]
        return {"reranked_chunks": budgeted_chunks, "assembled_context": sections}

    return post_process_node


def make_edge_grounding_node() -> Callable[[RetrievalState, Any], Awaitable[dict[str, object]]]:
    """Expand the reranked context along persisted typed relationship edges.

    Ported from the served inline ask path (`_edge_grounding_for_reranked` /
    `_resolved_edge_blocks` / `_edge_citations`) so the graph, not the service
    method, owns the behaviour. Fail-closed on every axis: without a live
    graphiti, a resolved tenant, or an explicit non-empty authorized document
    scope there is no edge read at all, and the expansion result is additive —
    an edge-free query keeps the exact edge-free contract.
    """

    async def edge_grounding_node(state: RetrievalState, runtime: Any = None) -> dict[str, object]:
        context = _invocation_context(runtime)
        graphiti = context.graphiti
        plan: QueryPlan = state["query_plan"]
        reranked: list[RetrievedChunk] = state.get("reranked_chunks", [])
        tenant_id = str(state.get("tenant_id") or "")
        scope_ids = tuple(state.get("allowed_doc_ids") or []) if state.get("scope_bound") else ()
        if graphiti is None or not tenant_id or not scope_ids or not reranked:
            # Fail-closed additive boundary: no expansion, no citation, and no
            # error surfaced for the absent grounding.
            return {
                "grounding_edge_texts": [],
                "edge_citations": [],
                **_stage_usage("edge_grounding", 0),
            }
        document_class = _state_document_class(state)
        refusals = _charge_stage(context, STAGE_EDGE_GROUNDING, document_class)
        if refusals:
            return {
                "grounding_edge_texts": [],
                "edge_citations": [],
                **_stage_usage(STAGE_EDGE_GROUNDING, 0),
                **_violation_state(refusals),
            }
        async with _stage_span(STAGE_EDGE_GROUNDING, context.budgets, document_class) as violations:
            expansions = await _edge_grounding_for_reranked(
                graphiti=graphiti,
                query=plan.rewritten_query,
                tenant_id=tenant_id,
                scope_ids=scope_ids,
                scope_versions=state.get("document_versions", []),
                scope_contracts=state.get("document_contracts", []),
                contract_index=state.get("contract_index"),
                reranked=reranked,
                count_tokens=context.count_tokens,
            )
        rows_by_chunk_id = {str(row["chunk_id"]): row for row in state.get("retrieved_rows", [])}
        resolved_blocks = _resolved_edge_blocks(expansions, rows_by_chunk_id)
        return {
            "grounding_edge_texts": [block.text for block, _ in resolved_blocks],
            "edge_citations": _edge_citations(resolved_blocks),
            **_stage_usage(STAGE_EDGE_GROUNDING, 1),
            **_violation_state(violations),
        }

    return edge_grounding_node


async def _edge_grounding_for_reranked(
    *,
    graphiti: Any,
    query: str,
    tenant_id: str,
    scope_ids: Sequence[str],
    scope_versions: Sequence[tuple[str, int, str | None]],
    scope_contracts: Sequence[tuple[str, int, tuple[int | None, ...], bool]],
    contract_index: int | None,
    reranked: Sequence[RetrievedChunk],
    count_tokens: CountTokens | None,
) -> list[EdgeExpansion]:
    """Expand reranked chunk context along persisted typed relationship edges.

    Verbatim port of the inline helper: a transport failure inside the reader
    degrades to no expansion, so an edge-free query keeps the exact pre-edge
    ask contract.
    """
    if graphiti is None or not tenant_id or not scope_ids:
        return []
    edges = await fetch_relationship_edges(
        graphiti,
        query,
        allowed_doc_ids=tuple(dict.fromkeys(scope_ids)),
        tenant_id=tenant_id,
        contract_index=contract_index,
        document_versions=scope_versions,
        document_contracts=scope_contracts,
        num_results=_EDGE_FETCH_LIMIT,
    )
    if not edges or not reranked:
        return []
    seeds = [
        EdgeSeedChunk(
            chunk_id=chunk.chunk_id,
            document_id=chunk.parent_doc_id,
            content=chunk.chunk_text,
            clause_ids=_chunk_clause_ids(chunk),
            contract_index=_chunk_contract_index(chunk),
            document_version=_chunk_document_version(chunk),
        )
        for chunk in reranked
    ]
    return expand_along_edges(
        seeds=seeds,
        edges=edges,
        max_expansion_tokens=_EDGE_EXPANSION_MAX_TOKENS,
        count_tokens=count_tokens or default_count_tokens,
        max_blocks_per_seed=_EDGE_EXPANSION_MAX_BLOCKS_PER_SEED,
    )


def _chunk_document_version(chunk: RetrievedChunk) -> int | None:
    """Read the chunk's document version from server metadata, if persisted."""
    raw = (chunk.metadata_ or {}).get("document_version")
    return int(raw) if isinstance(raw, int) and raw >= 1 else None


def _chunk_contract_index(chunk: RetrievedChunk) -> int | None:
    """Read the chunk's contract index from server metadata, if persisted.

    The value rides in metadata (never in the field the LLM could see), so
    reading the field would leave every seed "unknown" and disable the
    cross-contract matching guard.
    """
    return _context_contract_index(chunk.metadata_ or {})


def _chunk_clause_ids(chunk: RetrievedChunk) -> tuple[str, ...]:
    """Read server-persisted clause identities from chunk metadata, if any."""
    metadata = chunk.metadata_ or {}
    raw = metadata.get("clause_ids", metadata.get("clause_id"))
    if isinstance(raw, str) and raw:
        return (raw,)
    if isinstance(raw, (list, tuple, set, frozenset)):
        return tuple(str(item) for item in raw if item)
    return ()


def _resolved_edge_blocks(
    expansions: Sequence[EdgeExpansion],
    rows_by_chunk_id: Mapping[str, dict[str, object]],
) -> list[tuple[EdgeContextBlock, dict[str, object]]]:
    """Keep only expansion blocks whose grounded span resolves to a hydrated
    passage — the same resolution the citations use, so the generator never
    sees a relationship it could not cite against a retrieved chunk."""
    rows_by_document: dict[str, list[dict[str, object]]] = {}
    for row in rows_by_chunk_id.values():
        rows_by_document.setdefault(str(row.get("document_id")), []).append(row)
    resolved: list[tuple[EdgeContextBlock, dict[str, object]]] = []
    for expansion in expansions:
        for block in expansion.blocks:
            if block.char_start is None or block.char_end is None:
                continue
            supporting = _row_containing_span(
                rows_by_document.get(expansion.document_id, ()),
                char_start=block.char_start,
                char_end=block.char_end,
                document_version=block.document_version,
                contract_index=block.contract_index,
            )
            if supporting is not None:
                resolved.append((block, supporting))
    return resolved


def _edge_citations(
    resolved: Sequence[tuple[EdgeContextBlock, dict[str, object]]],
) -> list[GroundedCitation]:
    """Server-derived citations for edge-grounded expansion blocks.

    Provenance discipline matches `_citation_response`: document identity,
    version, and contract come from the hydrated chunk row — never from the
    edge text or any model output — and the claim is authored here,
    deterministically, from the edge's grounded span.

    The citation points at the passage the edge's grounded span actually
    falls in, resolved among the hydrated rows — never at the seed chunk
    that merely triggered the expansion. An edge whose span resolves to no
    hydrated passage — or that carries no span at all — earns no citation.
    """
    citations: list[GroundedCitation] = []
    seen: set[tuple[str, str]] = set()
    for block, supporting in resolved:
        chunk_id = str(supporting.get("chunk_id"))
        span = f"chars {block.char_start}-{block.char_end}"
        claim = (
            f"Grounded by {block.relationship.upper()} edge "
            f"'{block.from_entity}' -> '{block.to_entity}' at {span} "
            f"(clause {block.clause_id}, edge {block.edge_id})"
        )
        key = (chunk_id, claim)
        if key in seen:
            continue
        seen.add(key)
        citations.append(
            _citation_response(
                Citation(
                    chunk_id=chunk_id,
                    clause_type=str(supporting.get("clause_type") or "other"),
                    claim=claim,
                ),
                supporting,
            )
        )
    return citations


def _row_containing_span(
    rows: Sequence[dict[str, object]],
    *,
    char_start: int,
    char_end: int,
    document_version: int | None = None,
    contract_index: int | None = None,
) -> dict[str, object] | None:
    """Find the hydrated row whose persisted source span covers the edge span.

    When the edge carries version/contract provenance, the supporting passage
    must match it: two stored versions of one document can both cover a char
    range, and citing the wrong one dresses an unsupported answer as grounded.
    """
    for row in rows:
        metadata = row.get("metadata_") or {}
        if not isinstance(metadata, dict):
            continue
        start = metadata.get("source_start_char")
        end = metadata.get("source_end_char")
        if not isinstance(start, int) or not isinstance(end, int):
            continue
        if not (start <= char_start and char_end <= end):
            continue
        if document_version is not None:
            row_version = metadata.get("document_version")
            if isinstance(row_version, int) and row_version != document_version:
                continue
        if contract_index is not None:
            row_contract = metadata.get("contract_index")
            if isinstance(row_contract, int) and row_contract != contract_index:
                continue
        return row
    return None


@runtime_checkable
class ContextGrader(Protocol):
    """The narrow slice of the text-generation seam the context grader consumes.

    Structural, so the lifespan-owned text generator and any test double with
    the same single method satisfy it; this package never imports the concrete
    adapter. Verbatim shape of the inline ask path's grader seam.
    """

    def generate_structured(
        self, request: TextGenerationRequest
    ) -> Awaitable[TextGenerationStructuredResult]: ...


def _grader_from_runtime(runtime: Any) -> ContextGrader | None:
    """Read the optional grader seam from the invocation context, if supplied."""
    return _invocation_context(runtime).grader


def make_context_grader_node() -> Callable[[RetrievalState, Any], Awaitable[dict[str, object]]]:
    """Grade retrieval sufficiency through the injected grader seam.

    Ported from the inline ask loop's grade stage (the validation-pipeline
    adoption's `_ContextGradeEvaluator` over `_grade_context`): the optional
    grader arrives through the invocation context (`RetrievalGraphContext
    .grader`), the request runs at temperature 0.1, and the seam-absent
    degrade replays the legacy inline path verbatim. The bounded rewrite
    retry (`max_attempts=2`) is the graph's own conditional edge — one full
    plan→retrieve→rerank iteration per attempt, identical to the pipeline's
    feedback-carrying loop.
    """

    async def context_grader_node(state: RetrievalState, runtime: Any = None) -> dict[str, object]:
        context = _invocation_context(runtime)
        plan: QueryPlan = state["query_plan"]
        chunks: list[RetrievedChunk] = state.get("reranked_chunks", [])
        document_class = _state_document_class(state)
        refusals = _charge_stage(context, STAGE_GRADE, document_class)
        if refusals:
            # The same degrade the seam-failure path chose — proceed to
            # generation — but honest: the envelope names the stage and budget.
            grade = ContextGrade(sufficient=True, missing_aspects=[])
            return {
                "context_grade": grade,
                "iteration_count": state.get("iteration_count", 0) + 1,
                **_stage_usage(STAGE_GRADE, 0),
                **_violation_state(refusals),
            }
        async with _stage_span(STAGE_GRADE, context.budgets, document_class) as violations:
            grade = await _grade_context(
                grader_llm=_structured(context.llm, ContextGrade),
                grader=context.grader,
                query=plan.rewritten_query,
                reranked=chunks,
            )
        iteration_count = state.get("iteration_count", 0) + 1
        # Structured decision-log event (task 1.7): outcome, iteration, and
        # the cap-exhaustion escape flag, so the escape rate is derivable
        # without re-parsing free text.
        escaped_to_fallback = not grade.sufficient and iteration_count >= _GRADE_LOOP_MAX_ATTEMPTS
        logger.bind(
            outcome="sufficient" if grade.sufficient else "insufficient",
            iteration=iteration_count,
            escaped_to_fallback=escaped_to_fallback,
        ).info("grade_decision")
        return {
            "context_grade": grade,
            "iteration_count": iteration_count,
            **_stage_usage(STAGE_GRADE, 1),
            **_violation_state(violations),
        }

    return context_grader_node


async def _grade_context(
    *,
    grader_llm: Any,
    grader: ContextGrader | None,
    query: str,
    reranked: Sequence[RetrievedChunk],
) -> ContextGrade:
    """Grade retrieval sufficiency through the structured seam, or its legacy twin.

    Verbatim port of the inline dual-path grade: the injected grader rides the
    lifespan-owned text-generator seam (retries, breaker, schema-validated
    output) at temperature 0.1, while ``grader=None`` — the seam disabled at
    lifespan — replays the prior inline path verbatim so a disabled optional
    capability cannot change ask() results.
    """
    if not reranked:
        return ContextGrade(
            sufficient=False, missing_aspects=["No retrieved chunks"], rewrite_suggestion=query
        )
    payload = serialize_to_toon(
        {"query": query, "chunks": [chunk.model_dump() for chunk in reranked]}
    )
    if grader is None:
        return await _grade_context_via_llm(grader_llm=grader_llm, payload=payload)

    request = TextGenerationRequest(
        prompt=payload,
        system=_GRADE_SEAM_SYSTEM_PROMPT,
        response_schema=ContextGrade,
        # Grading is the same near-deterministic call the legacy inline path
        # made (dependencies.py builds the document LLM at temperature=0.1);
        # without this override the seam would inherit GEMINI_TEMPERATURE (0.7).
        temperature=0.1,
    )
    result = await grader.generate_structured(request)
    if isinstance(result, Failure):
        error = result.failure()
        log_expected_failure(error, operation="documents_context_grader")
        # Same degrade the legacy parse-failure path chose: an unavailable grade
        # proceeds to generation rather than failing the whole ask() loop.
        record_degradation(GRADER_FALLBACK, surface="ask_context_grader", reason="seam_failure")
        return ContextGrade(sufficient=True, missing_aspects=[])
    return ContextGrade.model_validate(result.unwrap().output)


async def _grade_context_via_llm(*, grader_llm: Any, payload: str) -> ContextGrade:
    """Legacy inline grading path, kept verbatim for the seam-absent degrade."""
    messages = [SystemMessage(content=_GRADE_SEAM_SYSTEM_PROMPT), HumanMessage(content=payload)]
    try:
        raw = await retry_immediate(
            lambda: grader_llm.ainvoke(cast("list[Any]", messages)),
            label="documents_context_grader",
        )
        return ContextGrade.model_validate(raw)
    except (ValueError, TypeError):
        record_degradation(GRADER_FALLBACK, surface="ask_context_grader", reason="parse_failure")
        return ContextGrade(sufficient=True, missing_aspects=[])


async def _fallback_outcome(
    state: RetrievalState,
    redis: Any | None,
    *,
    extra_warnings: list[StageWarning] | None = None,
) -> dict[str, object]:
    """The typed fallback outcome: fallback answer, finalised envelope, no spend.

    Shared by the grade-cap escape and the generation budget refusal; the
    caller adds its own stage-usage delta and, for a budget refusal, the
    typed violation warnings the envelope must name.
    """
    answer = GeneratedAnswer(answer=FALLBACK_ANSWER, citations=[], confidence="uncertain")
    finalised = _finalise_ask_state(state, answer)
    if extra_warnings:
        # The refusal warnings must reach the envelope this run finalises
        # (the state channel lands only after the node returns).
        finalised["quality_warnings"] = _merge_warning_lists(
            [
                cast("list[StageWarning]", finalised["quality_warnings"]),
                extra_warnings,
            ]
        )
    await _cache_ask_envelope(state, redis, answer, finalised)
    return {
        **finalised,
        "messages": [AIMessage(content=answer.answer)],
    }


def make_generator_node() -> Callable[[RetrievalState, Any], Awaitable[dict[str, object]]]:
    async def generator_node(state: RetrievalState, runtime: Any = None) -> dict[str, object]:
        context = _invocation_context(runtime)
        redis = context.redis
        envelope: CachedAskEnvelope | None = state.get("cached_envelope")
        if state.get("cache_hit") and envelope is not None:
            # The cached envelope serves the full contract the miss that wrote
            # it served — answer, citation provenance, coverage, warnings, and
            # evidence policy — with no new spend (the inline cache-hit
            # contract: the boundary's response is byte-identical, cache_hit
            # excepted).
            answer = GeneratedAnswer(
                answer=envelope.answer,
                citations=[
                    Citation(
                        chunk_id=citation.chunk_id,
                        clause_type=citation.clause_type,
                        claim=citation.claim,
                        contract_index=citation.contract_index,
                    )
                    for citation in envelope.citations
                ],
                confidence=envelope.confidence,
            )
            return {
                "generated_answer": answer,
                "resolved_citations": list(envelope.citations),
                "coverage": envelope.coverage,
                "quality_warnings": list(envelope.warnings),
                "evidence_policy": envelope.evidence_policy,
                "messages": [AIMessage(content=answer.answer)],
                **_stage_usage("generate", 0),
            }

        grade: ContextGrade | None = state.get("context_grade")
        cap_exhausted = (
            grade is not None
            and not grade.sufficient
            and state.get("iteration_count", 0) >= _GRADE_LOOP_MAX_ATTEMPTS
        )
        if cap_exhausted:
            return {
                **(await _fallback_outcome(state, redis)),
                **_stage_usage(STAGE_GENERATE, 0),
            }

        plan: QueryPlan = state["query_plan"]
        chunks: list[RetrievedChunk] = state.get("reranked_chunks", [])
        # Pre-call charge (G6 task 4.1): a refusal returns the typed fallback
        # outcome instead of spending an unbounded generation call.
        refusals = _charge_stage(context, STAGE_GENERATE, _state_document_class(state))
        if refusals:
            refusal_state = _violation_state(refusals)
            refusal_warnings = cast("list[StageWarning]", refusal_state.get("stage_warnings", []))
            return {
                **(await _fallback_outcome(state, redis, extra_warnings=refusal_warnings)),
                **_stage_usage(STAGE_GENERATE, 0),
                **refusal_state,
            }
        answer, violations = await _generate_answer(context, state, plan, chunks)

        if answer.confidence == "uncertain" and FALLBACK_ANSWER not in answer.answer:
            answer = answer.model_copy(update={"answer": f"{answer.answer}\n\n{FALLBACK_ANSWER}"})

        finalised = _finalise_ask_state(state, answer)
        await _cache_ask_envelope(state, redis, answer, finalised)

        return {
            **finalised,
            "messages": [AIMessage(content=answer.answer)],
            **_stage_usage(STAGE_GENERATE, 1),
            **_violation_state(violations),
        }

    return generator_node


async def _generate_answer(
    context: RetrievalGraphContext,
    state: RetrievalState,
    plan: QueryPlan,
    chunks: list[RetrievedChunk],
) -> tuple[GeneratedAnswer, list[StageBudgetViolation]]:
    """One grounded generation attempt over the assembled payload (G6/G5 seam).

    The payload gains ``grounding_edges`` only when edge expansion produced
    blocks (a query that touches no edges renders the exact pre-edge payload)
    and ``section_context`` only when hydration is enabled (G5, default off);
    citations always resolve from the chunk rows, never from either addition.
    """
    payload_data: dict[str, Any] = {
        "query": plan.rewritten_query,
        "chunks": [chunk.model_dump() for chunk in chunks],
    }
    grounding_edges = state.get("grounding_edge_texts") or []
    if grounding_edges:
        payload_data["grounding_edges"] = list(grounding_edges)
    # Section hydration (G5 task 4.3): DEFAULT OFF, adopted only on the
    # baseline's context-starvation attribution. The section text widens what
    # the generator sees; citations still resolve to the matched clauses from
    # the chunk rows, never from the hydrated text.
    if get_settings().DOCUMENTS_SECTION_HYDRATION_ENABLED:
        section_context = await _hydrate_sections(context, state, chunks)
        if section_context:
            payload_data["section_context"] = section_context
    payload = serialize_to_toon(payload_data)
    messages: list[SystemMessage | HumanMessage] = [
        SystemMessage(content=_GENERATOR_SYSTEM_PROMPT),
        HumanMessage(content=payload),
    ]
    generator_llm = _structured(context.llm, GeneratedAnswer)
    async with _stage_span(
        STAGE_GENERATE, context.budgets, _state_document_class(state)
    ) as violations:
        try:
            raw_answer = await retry_immediate(
                lambda: generator_llm.ainvoke(cast("list[Any]", messages)),
                label="gemini_grounded_generator",
            )
            return GeneratedAnswer.model_validate(raw_answer), violations
        except Exception as exc:  # noqa: BLE001 — generator failure must return hard fallback
            exc.add_note("operation=generator")
            logger.bind(operation="generator", error=str(exc)).warning(
                "generator_failed_using_fallback"
            )
            return (
                GeneratedAnswer(answer=FALLBACK_ANSWER, citations=[], confidence="uncertain"),
                violations,
            )


async def _hydrate_sections(
    context: RetrievalGraphContext,
    state: RetrievalState,
    chunks: list[RetrievedChunk],
) -> dict[str, str]:
    """Per-chunk parent-section text from the stored structural trees.

    Fail-closed and additive: no repository, no resolved tenant, a tree-read
    failure, or a chunk whose heading path matches nothing all hydrate
    nothing — the generator payload is then byte-identical to the
    non-hydrated contract.
    """
    from app.features.documents.retrieval.graph.hydration import (  # noqa: PLC0415
        hydrate_sections_for_chunks,
    )

    repo = context.repo
    user_id = str(state.get("user_id") or "")
    document_ids = sorted({chunk.parent_doc_id for chunk in chunks})
    if repo is None or not user_id or not document_ids:
        return {}
    try:
        trees_result = await repo.fetch_structural_trees(
            user_id=user_id,
            tenant_id=state.get("tenant_id"),
            document_ids=document_ids,
        )
    except Exception:  # noqa: BLE001 — hydration degrades, never fails the ask
        logger.bind(operation="section_hydration").warning("section_hydration_failed")
        return {}
    if trees_result is None or isinstance(trees_result, Failure):
        return {}
    trees = {
        str(row["document_id"]): row.get("structural_tree") or {}
        for row in trees_result.unwrap()
        if isinstance(row.get("structural_tree"), dict)
    }
    return hydrate_sections_for_chunks(
        trees=trees,
        chunks=chunks,
        count_tokens=context.count_tokens or default_count_tokens,
    )


async def _cache_ask_envelope(
    state: RetrievalState,
    redis: Any | None,
    answer: GeneratedAnswer,
    finalised: dict[str, object],
) -> None:
    """Write the full finalised ask envelope under the version-aware key.

    The cache stores the envelope — not the bare answer — so a cache hit
    serves the same coverage, warnings, citations, and evidence policy the
    miss served (the inline contract; drift-register item 4).
    """
    if redis is None:
        return
    envelope = CachedAskEnvelope(
        answer=answer.answer,
        confidence=answer.confidence,
        citations=list(finalised["resolved_citations"]),  # ty: ignore[invalid-argument-type]
        evidence_policy=finalised["evidence_policy"],  # ty: ignore[invalid-argument-type]
        coverage=finalised["coverage"],  # ty: ignore[invalid-argument-type]
        warnings=list(finalised["quality_warnings"]),  # ty: ignore[invalid-argument-type]
    )
    cache_key = state.get("answer_cache_key") or _answer_cache_key(state)
    from app.features.documents.domain.constants import (  # noqa: PLC0415
        DEFAULT_SEARCH_CACHE_TTL_SECONDS,
    )

    await _with_redis_deadline(
        "ask_envelope_cache_write",
        lambda: redis.setex(
            cache_key, DEFAULT_SEARCH_CACHE_TTL_SECONDS, envelope.model_dump_json()
        ),
    )


def _finalise_ask_state(state: RetrievalState, answer: GeneratedAnswer) -> dict[str, object]:
    """Finalisation (task 1.4): the truthful coverage and warning envelope.

    Coverage is derived from the hydrated rows and the scope-unavailable
    documents, never asserted; warnings are merged and deduplicated exactly
    as the inline `_warnings_from_rows`/`_merge_warning_lists` pair does; the
    scope-unavailable warning matches the inline contract verbatim; citation
    provenance is resolved from the hydrated rows (`_citation_response`
    discipline) and the server-derived edge citations are appended.
    """
    rows: list[dict[str, Any]] = state.get("retrieved_rows", [])
    rows_by_chunk_id = {str(row["chunk_id"]): row for row in rows}
    require_graphiti_verified = bool(state.get("require_graphiti_verified", False))
    scope_bound = bool(state.get("scope_bound"))
    result_document_ids = {str(row["document_id"]) for row in rows}
    graph_document_ids = {
        str(row["document_id"]) for row in rows if bool(row.get("graphiti_verified", False))
    }
    warnings = _merge_warning_lists(
        [_warnings_from_rows(rows), list(state.get("stage_warnings", []))]
    )
    unavailable_scope_documents = list(state.get("scope_unavailable_documents", []))
    if unavailable_scope_documents:
        warnings.append(
            StageWarning(
                stage="graph_scope",
                code="GRAPH_SCOPE_UNAVAILABLE",
                message=(
                    "Authoritative document version/contract scope was unavailable "
                    f"for {len(unavailable_scope_documents)} authorized document(s)."
                ),
                severity="warning",
            )
        )
    citations = [
        _citation_response(citation, rows_by_chunk_id.get(citation.chunk_id))
        for citation in answer.citations
    ]
    citations.extend(state.get("edge_citations", []))
    return {
        "generated_answer": answer,
        "coverage": CoverageEnvelope(
            documents_requested=(
                len(state.get("allowed_doc_ids", []))
                if scope_bound
                else len(state.get("doc_ids_filter", []))
            ),
            documents_ready=len(result_document_ids),
            graph_indexed_documents=len(graph_document_ids),
            graph_unavailable_documents=len(result_document_ids - graph_document_ids),
        ),
        "quality_warnings": warnings,
        "resolved_citations": citations,
        "evidence_policy": (
            "graph_indexed_only" if require_graphiti_verified else "mechanically_verified"
        ),
    }


def should_run_graph(state: RetrievalState) -> str:
    if state.get("cache_hit"):
        return "generate"
    route = state["query_plan"].route
    return "graph" if route in {"graph_neo4j", "both"} else "hybrid"


def should_retry_or_generate(state: RetrievalState) -> str:
    grade = state.get("context_grade")
    if (
        grade is not None
        and not grade.sufficient
        and state.get("iteration_count", 0) < _GRADE_LOOP_MAX_ATTEMPTS
    ):
        return "query_analyzer"
    return "generate"


def _normalize_plan(plan: QueryPlan) -> QueryPlan:
    vector_weight = plan.vector_weight
    keyword_weight = plan.keyword_weight
    total = vector_weight + keyword_weight
    if total <= 0:
        vector_weight = 0.4
        keyword_weight = 0.6
    else:
        # Total-normalization ONLY — identical to the inline plan contract. The
        # former query_type forcing (0.75/0.25 exact, 0.65/0.35 conceptual) made
        # the two fused doors compute different rankings for the same plan
        # (parity: test_hybrid_fused_parity); the trigram constant had masked it.
        # Deliberate plan-weight steering belongs in the plan itself, not in the
        # fusion door. (remove-trigram-fusion-branch, drift item 9.)
        vector_weight /= total
        keyword_weight /= total
    return plan.model_copy(
        update={"vector_weight": vector_weight, "keyword_weight": keyword_weight}
    )


def _row_to_chunk(row: dict[str, Any]) -> RetrievedChunk:
    # Harden 5.3: hydration is centralized in `retrieval.row_hydration` — this
    # module-level alias keeps the parity suite's import surface stable while
    # nodes and the evidence service share one implementation.
    return row_hydration.row_to_chunk(row)


def _answer_cache_key(state: Mapping[str, Any]) -> str:
    """Version-aware answer cache key, matching the inline `_build_answer_cache_key`.

    The derivation covers the tenant, the authorized scope, the resolved
    document version of every scoped document, the canonical pipeline
    version, and the request's evidence policy — so a cached answer can never
    be served after a scoped document's version or the pipeline version has
    changed. The query is the *original* request query (normalized), not the
    plan's rewrite, matching the inline derivation.
    """
    from app.features.documents.domain.constants import (  # noqa: PLC0415
        CANONICAL_PIPELINE_VERSION,
    )

    # `user_id` is mandatory on the served path (the hybrid node requires it),
    # but the derivation tolerates its absence so a malformed state fails at
    # retrieval, not in the cache read.
    user_id = str(state.get("user_id") or "")
    tenant_id = str(state.get("tenant_id") or user_id)
    require_graphiti_verified = bool(state.get("require_graphiti_verified", False))
    raw = to_sorted_key_bytes(
        {
            "user_id": user_id,
            "tenant_id": tenant_id,
            "query": " ".join(str(state["query"]).lower().split()),
            "doc_ids_filter": sorted(str(doc_id) for doc_id in state.get("doc_ids_filter", [])),
            "allowed_doc_ids": sorted(str(doc_id) for doc_id in state.get("allowed_doc_ids", [])),
            "document_versions": sorted(
                (
                    (str(doc_id), int(version), pipeline)
                    for doc_id, version, pipeline in state.get("document_versions", [])
                ),
                key=itemgetter(0),
            ),
            "jurisdiction": state.get("jurisdiction"),
            "contract_type": state.get("contract_type"),
            "contract_index": state.get("contract_index"),
            "clause_type": state.get("clause_type"),
            "require_graphiti_verified": require_graphiti_verified,
            "evidence_policy": (
                "graph_indexed_only" if require_graphiti_verified else "mechanically_verified"
            ),
            "pipeline_version": CANONICAL_PIPELINE_VERSION,
        },
    )
    return "documents:answer:" + hashlib.sha256(raw).hexdigest()


def _flatten_warnings(raw_groups: list[Any]) -> list[StageWarning]:
    warnings: list[StageWarning] = []
    for group in raw_groups:
        if isinstance(group, list):
            warnings.extend(
                StageWarning.model_validate(warning)
                for warning in group
                if isinstance(warning, dict)
            )
        elif isinstance(group, dict):
            warnings.append(StageWarning.model_validate(group))
    return warnings


def _merge_warning_lists(groups: list[list[StageWarning]]) -> list[StageWarning]:
    """Merge warning groups, deduplicating on (stage, code, message, severity).

    Verbatim port of the inline `_merge_warning_lists`: first occurrence
    wins, insertion order preserved.
    """
    merged: dict[tuple[str, str, str, str], StageWarning] = {}
    for group in groups:
        for warning in group:
            merged[warning.stage, warning.code, warning.message, warning.severity] = warning
    return list(merged.values())


def _warnings_from_rows(rows: list[dict[str, Any]]) -> list[StageWarning]:
    return _flatten_warnings([row.get("quality_warnings", []) for row in rows])


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


def _citation_response(citation: Citation, row: dict[str, object] | None) -> GroundedCitation:
    """Attach immutable source provenance to a citation when a row is available.

    Verbatim port of the inline `_citation_response` discipline: identity,
    version, contract, pipeline version, and page come from the hydrated row,
    never from the citation text or any model output.
    """
    grounded = GroundedCitation(
        chunk_id=citation.chunk_id,
        clause_type=citation.clause_type,
        claim=citation.claim,
    )
    if row is None:
        return grounded
    return grounded.model_copy(
        update={
            "document_id": str(row["document_id"]) if row.get("document_id") else None,
            "document_version": int(cast("int | str", row.get("document_version", 1))),
            "contract_index": (
                int(cast("int | str", row["contract_index"]))
                if isinstance(row.get("contract_index"), (int, str))
                and not isinstance(row.get("contract_index"), bool)
                else None
            ),
            "pipeline_version": (
                str(row["pipeline_version"]) if row.get("pipeline_version") is not None else None
            ),
            # Harden 5.3: page numbers come from the centralized hydration
            # path (None/absent → 0), never a bare int() over the raw row.
            "page_no": row_hydration.page_no(row),
        }
    )


def _extract_postgres_chunk_ids(value: str) -> list[str]:
    return re.findall(
        r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}",
        value,
    )
