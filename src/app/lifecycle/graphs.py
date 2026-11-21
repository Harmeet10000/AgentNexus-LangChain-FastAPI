"""Process-scoped graph providers shared by API and worker startup paths.

Importing this module performs no provisioning and compiles no graph.  Callers
own the process lifecycle and pass already-created infrastructure handles.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from sqlalchemy import (
    Integer,
    String,
    cast,
    func,
    select,
)

from app.ai.llm import build_chat_model
from app.ai.memory import AgentMemoryService
from app.connections.checkpointer import (
    setup_langgraph_checkpointer,
)
from app.features.agent_saul.graph import build_saul_graph
from app.features.documents.application.evidence.service import build_evidence_service
from app.features.documents.application.tools.bundle import build_tool_bundle
from app.features.documents.application.tools.portfolio import (
    RelationshipEdgeVerdict,
    relationship_edges_available,
)
from app.features.documents.domain.chunk import UnifiedChunk
from app.observability.logging import logger
from app.resilience.idempotency import (
    IdempotencyGuard,
)

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable
    from typing import Any

    from graphiti_core import Graphiti
    from langchain_core.language_models import BaseChatModel
    from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
    from langgraph.graph.state import CompiledStateGraph
    from redis.asyncio import Redis
    from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker

    from app.config.settings import Settings
    from app.extraction.langextract.service import AsyncExtractionService
    from app.features.documents.application.evidence.service import EvidenceService
    from app.features.documents.application.retrieval.graph.engine import RetrievalEngine
    from app.features.documents.application.tools.portfolio import RelationshipEdgeProbe
    from app.retrieval.graphiti.client import BoundGraphitiService as GraphitiService
    from app.utils.storage import StorageService

    type IngestDocumentFn = Callable[..., Awaitable[Any]]


async def provide_langgraph_checkpointer(database_url: str) -> AsyncPostgresSaver:
    """Provision the checkpointer for the process that owns its pool."""

    return await setup_langgraph_checkpointer(database_url)


def provide_retrieval_engine(
    *,
    session_factory: async_sessionmaker[AsyncSession],
    redis: Redis | None,
    graphiti: Graphiti | None,
    grader: Any | None = None,
) -> RetrievalEngine:
    """Compile the retrieval graph once and bind its per-invocation collaborators.

    The compiled graph captures nothing request-scoped (mount-retrieval-graph
    task 3.1): the repository arrives per invocation over a short-lived
    session, and the language model is constructed on first use — an eager
    construction would let a provider misconfiguration break boot for a
    capability that can degrade. Provisioning failure degrades via the
    startup policy registry; the ask endpoints and the qna node then fail
    closed with a typed unavailability.
    """
    from app.features.documents.application.retrieval.graph.engine import (  # noqa: PLC0415
        RetrievalEngine,
    )

    return RetrievalEngine(
        session_factory=session_factory,
        redis=redis,
        graphiti=graphiti,
        # Same construction contract as the documents HTTP dependency: built
        # lazily, at the serving temperature the ask contract pins (0.1).
        llm_factory=lambda: build_chat_model(
            model_name=None,
            temperature=0.1,
            implementation="generic",
        ),
        grader=grader,
    )


def provide_agent_memory_service(settings: Settings) -> AgentMemoryService:
    """Construct Agent Saul's cheap, process-scoped memory collaborator."""

    return AgentMemoryService(partition_prefix=settings.COGNEE_DATASET_PREFIX)


def provide_document_ingestion_graph(
    *,
    settings: Settings,
    object_store: StorageService,
    graphiti: Graphiti | None,
    ingest_document_fn: IngestDocumentFn,
    extraction: AsyncExtractionService | None = None,
    graph_writer: object | None = None,
    idempotency: object | None = None,
) -> CompiledStateGraph[Any]:
    """Compile ingestion once; the repository arrives with each invocation.

    The graph resolves embeddings through the shared embedding layer.  Do not
    add an ``embedding_fn`` argument from older revisions.
    """
    from app.features.documents.application.ingestion.workflow import (  # noqa: PLC0415
        build_document_ingestion_graph,
    )

    llm: BaseChatModel = build_chat_model(
        model_name=settings.GEMINI_FLASH_MODEL,
        temperature=0.1,
        implementation="generic",
    )
    return build_document_ingestion_graph(
        object_store=object_store,
        graphiti=graphiti,
        ingest_document_fn=ingest_document_fn,
        llm=llm,
        extraction=extraction,
        graph_writer=graph_writer,
        idempotency=idempotency,
    )


async def provide_saul_graph(
    *,
    settings: Settings,
    checkpointer: AsyncPostgresSaver,
    redis: Redis,
    db_engine: AsyncEngine,
    session_factory: async_sessionmaker[AsyncSession],
    graphiti: GraphitiService,
    evidence_service: EvidenceService | None = None,
    retrieval_engine: RetrievalEngine | None = None,
) -> CompiledStateGraph[Any]:
    """Compile Agent Saul once from already-provisioned process resources.

    The portfolio traversal binding (openspec ``agent-portfolio-traversal-tools``)
    runs the binding-time gate probe here, against the SAME provisioned
    Graphiti service the other graph consumers use — no new connection. A
    probe that cannot complete (backend error, timeout, empty pre-ingestion
    corpus) degrades to skipped bindings with a structured log, never a
    startup failure; only a completed probe is passed on, and a completed
    ``no_edges`` verdict still fails the graph build loudly (design D2). The
    probe runs strictly, so a swallowed backend error cannot masquerade as a
    completed ``no_edges`` verdict — an unevaluable read skips the bindings
    instead of failing the boot.
    """

    memory_service = provide_agent_memory_service(settings)
    idempotency = IdempotencyGuard(redis=redis, db_engine=db_engine)
    if evidence_service is None:
        evidence_service = build_evidence_service(
            session_factory=session_factory,
            redis=redis,
            handle_ttl_seconds=settings.EVIDENCE_HANDLE_TTL_SECONDS,
            graphiti=graphiti,
        )
    tool_bundle = build_tool_bundle(
        graphiti_service=graphiti,
        db_engine=db_engine,
        idempotency=idempotency,
        evidence_service=evidence_service,
    )
    pro_llm = build_chat_model(model_name=settings.GEMINI_PRO_MODEL)
    flash_llm = build_chat_model(model_name=settings.GEMINI_FLASH_MODEL)
    try:
        probe = await traversal_gate_probe(graphiti, session_factory)
    except Exception as exc:  # noqa: BLE001 — degrade-gracefully: never block boot
        logger.bind(
            component="saul_graph",
            operation="traversal_gate_probe",
            error=str(exc),
        ).opt(exception=True).warning(
            "Traversal gate probe raised; portfolio traversal bindings skipped for this boot"
        )
        probe = None
    return build_saul_graph(
        checkpointer=checkpointer,
        pro_llm=pro_llm,
        flash_llm=flash_llm,
        memory_service=memory_service,
        tool_registry=tool_bundle,
        traversal_graphiti_service=graphiti if probe is not None else None,
        traversal_gate_probe=probe,
        retrieval_engine=retrieval_engine,
    )


# ---------------------------------------------------------------------------
# Portfolio traversal binding-time gate (openspec agent-portfolio-traversal-tools)
# ---------------------------------------------------------------------------

#: The gate is existential per deployment (does the relationship producer
#: persist typed edges at all), so a bounded sample of the corpus answers it
#: without scanning every partition.
_TRAVERSAL_PROBE_TENANT_LIMIT = 8
_TRAVERSAL_PROBE_DOCS_PER_TENANT = 16
_TRAVERSAL_PROBE_ROW_LIMIT = _TRAVERSAL_PROBE_TENANT_LIMIT * _TRAVERSAL_PROBE_DOCS_PER_TENANT


async def _corpus_graph_scopes(
    session_factory: async_sessionmaker[AsyncSession],
) -> dict[
    str,
    tuple[
        tuple[str, ...],
        tuple[tuple[str, int, str | None], ...],
        tuple[tuple[str, int, tuple[int | None, ...], bool], ...],
    ],
]:
    """Sample the ingested corpus as per-tenant Graphiti read scopes.

    One bounded aggregate over the chunk rows (the same truth source the
    repository's ``resolve_scope_contracts`` aggregates), grouped by
    tenant/document/version with each version's contract partitions. Bounded
    because the gate is existential: a fixed-size sample answers "does this
    deployment's graph carry typed edges" without enumerating every tenant.
    """
    statement = (
        select(
            UnifiedChunk.tenant_id.label("tenant_id"),
            cast(UnifiedChunk.document_id, String).label("document_id"),
            cast(UnifiedChunk.document_version, Integer).label("document_version"),
            func.array_agg(func.distinct(UnifiedChunk.contract_index)).label("contract_indexes"),
            func.bool_and(UnifiedChunk.contract_index.is_(None)).label("all_legacy_null"),
        )
        .group_by(
            UnifiedChunk.tenant_id,
            UnifiedChunk.document_id,
            UnifiedChunk.document_version,
        )
        .order_by(UnifiedChunk.tenant_id.asc(), UnifiedChunk.document_id.asc())
        .limit(_TRAVERSAL_PROBE_ROW_LIMIT)
    )
    async with session_factory() as session:
        result = await session.execute(statement)
        rows = result.mappings().all()
    corpus: dict[
        str,
        tuple[
            list[str],
            list[tuple[str, int, str | None]],
            list[tuple[str, int, tuple[int | None, ...], bool]],
        ],
    ] = {}
    for row in rows:
        tenant_id = str(row["tenant_id"])
        document_id = str(row["document_id"])
        version = int(row["document_version"])
        scopes = corpus.setdefault(tenant_id, ([], [], []))
        doc_ids, versions, contracts = scopes
        if len(doc_ids) >= _TRAVERSAL_PROBE_DOCS_PER_TENANT:
            continue
        doc_ids.append(document_id)
        versions.append((document_id, version, None))
        raw_indexes = row["contract_indexes"] or []
        indexes = tuple(int(index) for index in raw_indexes if index is not None and index >= 0)
        legacy_null = bool(row["all_legacy_null"]) and not indexes
        contracts.append((document_id, version, indexes, legacy_null))
    return {tenant: (tuple(a), tuple(b), tuple(c)) for tenant, (a, b, c) in corpus.items()}


async def traversal_gate_probe(
    graphiti: GraphitiService,
    session_factory: async_sessionmaker[AsyncSession],
) -> RelationshipEdgeProbe | None:
    """Probe the corpus for typed relationship edges at graph-build time.

    Returns a completed probe, or ``None`` when the gate is not evaluable:
    any error along the way (Neo4j unreachable, database failure, graph
    backend timeout) and an empty pre-ingestion corpus degrade to ``None``
    with a structured log — bindings stay skipped and the application boots.
    The probe's edge read runs strictly (``relationship_edges_available(...,
    strict=True)``), so a backend failure is re-raised here and becomes
    ``None`` — a swallowed transport error can never masquerade as a
    completed ``no_edges`` verdict and fail the boot. A completed
    ``no_edges`` verdict from a read that genuinely ran is returned as-is so
    the graph factory can fail the build loudly (design D2); the
    per-request runtime probe in the traversal tools keeps per-user honesty
    either way.
    """
    try:
        corpus = await _corpus_graph_scopes(session_factory)
    except Exception as exc:  # noqa: BLE001 — degrade-gracefully: never block boot
        logger.bind(component="saul_graph", operation="traversal_gate_probe", error=str(exc)).opt(
            exception=True
        ).warning(
            "Traversal gate probe could not read the corpus scope; "
            "portfolio traversal bindings skipped for this boot"
        )
        return None
    if not corpus:
        logger.bind(component="saul_graph", operation="traversal_gate_probe").info(
            "Corpus is empty; traversal gate not evaluable and portfolio "
            "traversal bindings skipped until documents are ingested"
        )
        return None
    try:
        probe: RelationshipEdgeProbe | None = None
        for tenant_id in sorted(corpus):
            doc_ids, versions, contracts = corpus[tenant_id]
            probe = await relationship_edges_available(
                graphiti,
                allowed_doc_ids=doc_ids,
                tenant_id=tenant_id,
                document_versions=versions,
                document_contracts=contracts,
                strict=True,
            )
            if probe.verdict is RelationshipEdgeVerdict.EDGES_PRESENT:
                return probe
    except Exception as exc:  # noqa: BLE001 — degrade-gracefully: never block boot
        # Includes the strict probe's re-raised transport errors/timeouts:
        # an unevaluable read skips the bindings, it never fails the boot.
        logger.bind(component="saul_graph", operation="traversal_gate_probe", error=str(exc)).opt(
            exception=True
        ).warning(
            "Traversal gate probe failed against the graph backend; "
            "portfolio traversal bindings skipped for this boot"
        )
        return None
    # A completed no_edges verdict across the whole sampled corpus is
    # returned as-is: the graph factory fails the build loudly on it (D2).
    return probe


def graphiti_service(graphiti: Graphiti) -> GraphitiService:
    """Bind a raw Graphiti client to the application's canonical operations."""
    from app.retrieval.graphiti.client import BoundGraphitiService  # noqa: PLC0415

    return BoundGraphitiService(graphiti)
