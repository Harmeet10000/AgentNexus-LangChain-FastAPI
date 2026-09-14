"""Process-scoped graph providers shared by API and worker startup paths.

Importing this module performs no provisioning and compiles no graph.  Callers
own the process lifecycle and pass already-created infrastructure handles.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable
    from typing import Any

    from graphiti_core import Graphiti
    from langchain_core.language_models import BaseChatModel
    from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
    from langgraph.graph.state import CompiledStateGraph
    from redis.asyncio import Redis
    from sqlalchemy.ext.asyncio import AsyncEngine

    from app.config.settings import Settings
    from app.shared.langchain_layer.agents.memory import AgentMemoryService
    from app.shared.rag.graphiti.client import GraphitiService
    from app.shared.rag.langextract.service import AsyncExtractionService
    from app.shared.services.storage import StorageService

    type IngestDocumentFn = Callable[..., Awaitable[Any]]


async def provide_langgraph_checkpointer(database_url: str) -> AsyncPostgresSaver:
    """Provision the checkpointer for the process that owns its pool."""
    from app.shared.langgraph_layer.checkpointer import (  # noqa: PLC0415
        setup_langgraph_checkpointer,
    )

    return await setup_langgraph_checkpointer(database_url)


def provide_agent_memory_service(settings: Settings) -> AgentMemoryService:
    """Construct Agent Saul's cheap, process-scoped memory collaborator."""
    from app.shared.langchain_layer.agents.memory import AgentMemoryService  # noqa: PLC0415

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
    from app.features.documents.ingestion_graph import (  # noqa: PLC0415
        build_document_ingestion_graph,
    )
    from app.shared.langchain_layer.models import build_chat_model  # noqa: PLC0415

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


def provide_saul_graph(
    *,
    settings: Settings,
    checkpointer: AsyncPostgresSaver,
    redis: Redis,
    db_engine: AsyncEngine,
    graphiti: GraphitiService,
) -> CompiledStateGraph[Any]:
    """Compile Agent Saul once from already-provisioned process resources."""
    from app.shared.langchain_layer.agents.tools.idempotency import (  # noqa: PLC0415
        IdempotencyGuard,
    )
    from app.shared.langchain_layer.models import build_chat_model  # noqa: PLC0415
    from app.shared.langgraph_layer.agent_saul import build_saul_graph  # noqa: PLC0415
    from app.shared.rag.graphiti.registry import build_tool_bundle  # noqa: PLC0415

    memory_service = provide_agent_memory_service(settings)
    idempotency = IdempotencyGuard(redis=redis, db_engine=db_engine)
    tool_bundle = build_tool_bundle(
        graphiti_service=graphiti,
        db_engine=db_engine,
        idempotency=idempotency,
    )
    pro_llm = build_chat_model(model_name=settings.GEMINI_PRO_MODEL)
    flash_llm = build_chat_model(model_name=settings.GEMINI_FLASH_MODEL)
    return build_saul_graph(
        checkpointer=checkpointer,
        pro_llm=pro_llm,
        flash_llm=flash_llm,
        memory_service=memory_service,
        tool_registry=tool_bundle,
    )


def graphiti_service(graphiti: Graphiti) -> GraphitiService:
    """Bind a raw Graphiti client to the application's canonical operations."""
    from app.shared.rag.graphiti.client import BoundGraphitiService  # noqa: PLC0415

    return cast("GraphitiService", BoundGraphitiService(graphiti))
