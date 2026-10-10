"""Celery child-process lifecycle for document ingestion resources."""

from __future__ import annotations

import asyncio
import os
from typing import TYPE_CHECKING, Any  # noqa: TC003 — Any resolved at runtime by Pydantic

from celery.signals import worker_process_init, worker_process_shutdown, worker_shutdown
from graphiti_core import Graphiti  # noqa: TC002 — resolved at runtime by Pydantic
from langgraph.graph.state import CompiledStateGraph
from pydantic import BaseModel, ConfigDict
from redis.asyncio import Redis  # noqa: TC002 — resolved at runtime by Pydantic
from sqlalchemy.ext.asyncio import (  # noqa: TC002 — resolved at runtime by Pydantic
    AsyncEngine,
    async_sessionmaker,
)

from app.config import get_settings
from app.connections import init_db
from app.connections.graphiti import (
    close_graphiti,
    setup_graphiti,
    setup_graphiti_indices,
)
from app.connections.redis import create_redis_client
from app.features.documents.service import process_document_ingestion
from app.lifecycle.graphs import graphiti_service, provide_document_ingestion_graph
from app.resilience.idempotency import IdempotencyGuard
from app.extraction.langextract.service import AsyncExtractionService
from app.connections.object_storage import StorageService
from app.observability.logging import logger
from app.utils import ServiceUnavailableException

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Coroutine


class DocumentWorkerResources(BaseModel):
    """Resources bound to one worker child and its persistent event loop."""

    model_config = ConfigDict(extra="forbid", frozen=True, arbitrary_types_allowed=True)

    engine: AsyncEngine
    session_local: async_sessionmaker[Any]
    graphiti: Graphiti
    ingestion_graph: CompiledStateGraph[Any]
    idempotency: IdempotencyGuard
    redis: Redis | None = None


_WORKER_PID: int | None = None
_WORKER_RUNNER: asyncio.Runner | None = None
_WORKER_RESOURCES: DocumentWorkerResources | None = None


async def _close_document_worker_resources(
    *,
    engine: AsyncEngine,
    graphiti: Graphiti | None,
    redis: Redis | None,
    raise_errors: bool,
) -> None:
    """Attempt every close operation, optionally surfacing the combined failures."""
    operations: list[tuple[str, Callable[[], Awaitable[None]]]] = [
        ("graphiti", lambda: close_graphiti(graphiti)),
    ]
    if redis is not None:
        operations.append(("redis", redis.aclose))
    operations.append(("database", engine.dispose))

    errors: list[Exception] = []
    for resource, close in operations:
        try:
            await close()
        except Exception as exc:  # noqa: BLE001 — cleanup must continue across resources
            logger.bind(resource=resource, error_type=type(exc).__name__).exception(
                "Document ingestion worker resource close failed"
            )
            errors.append(exc)
    if errors and raise_errors:
        message = "Document worker resource cleanup failed"
        raise ExceptionGroup(message, errors)


async def _provision_document_worker() -> DocumentWorkerResources:
    """Construct all async resources on the loop that will execute tasks."""
    settings = get_settings()
    engine, session_local = await init_db()
    graphiti: Graphiti | None = None
    redis: Redis | None = None
    try:
        redis_url = getattr(settings, "REDIS_URL", "")
        if redis_url:
            redis = create_redis_client(redis_url)
            await redis.ping()
        graphiti = await setup_graphiti(
            neo4j_uri=settings.NEO4J_URI,
            neo4j_user=settings.NEO4J_USERNAME,
            neo4j_password=settings.NEO4J_PASSWORD.get_secret_value(),
        )
        await setup_graphiti_indices(graphiti)
        object_store = StorageService.from_settings(settings=settings)
        idempotency = IdempotencyGuard(
            redis=redis,
            db_engine=engine,
            require_durable=True,
        )
        ingestion_graph = provide_document_ingestion_graph(
            settings=settings,
            object_store=object_store,
            graphiti=graphiti,
            ingest_document_fn=process_document_ingestion,
            extraction=AsyncExtractionService.from_settings(settings),
            graph_writer=graphiti_service(graphiti),
            idempotency=idempotency,
        )
        return DocumentWorkerResources(
            engine=engine,
            session_local=session_local,
            graphiti=graphiti,
            ingestion_graph=ingestion_graph,
            idempotency=idempotency,
            redis=redis,
        )
    except Exception:
        await _close_document_worker_resources(
            engine=engine,
            graphiti=graphiti,
            redis=redis,
            raise_errors=False,
        )
        raise


async def _release_document_worker(resources: DocumentWorkerResources) -> None:
    await _close_document_worker_resources(
        engine=resources.engine,
        graphiti=resources.graphiti,
        redis=resources.redis,
        raise_errors=True,
    )


@worker_process_init.connect
def initialize_document_worker(**_kwargs: object) -> None:
    """Reset inherited state; first task provisions resources after child startup."""
    global _WORKER_RESOURCES, _WORKER_RUNNER, _WORKER_PID  # noqa: PLW0603
    _WORKER_RESOURCES = None
    _WORKER_RUNNER = None
    _WORKER_PID = os.getpid()


def _ensure_document_worker() -> None:
    """Lazily provision once on the owning child's persistent loop."""
    global _WORKER_RESOURCES, _WORKER_RUNNER  # noqa: PLW0603
    if os.getpid() != _WORKER_PID:
        initialize_document_worker()
    if _WORKER_RESOURCES is not None:
        return
    _WORKER_RUNNER = asyncio.Runner()
    try:
        _WORKER_RESOURCES = _WORKER_RUNNER.run(_provision_document_worker())
    except Exception as exc:
        exc.add_note("capability=document, operation=worker_provision")
        _WORKER_RUNNER.close()
        _WORKER_RUNNER = None
        raise ServiceUnavailableException(detail="Document worker provisioning failed") from exc
    else:
        logger.info("Document ingestion worker resources initialized")


def get_document_worker_resources() -> DocumentWorkerResources:
    """Return provisioned resources or a typed, capability-naming failure."""
    _ensure_document_worker()
    if _WORKER_RESOURCES is None:
        raise ServiceUnavailableException(
            detail="Document ingestion worker is unavailable",
            data={"capability": "document_ingestion_graph"},
        )
    return _WORKER_RESOURCES


def run_on_document_worker_loop[T](
    coroutine_factory: Callable[[], Coroutine[Any, Any, T]],
) -> T:
    """Run task I/O on the same loop that created the process resources."""
    get_document_worker_resources()
    if _WORKER_RUNNER is None:
        message = "Worker resource guard returned without a runner"
        raise AssertionError(message)
    return _WORKER_RUNNER.run(coroutine_factory())


@worker_process_shutdown.connect
@worker_shutdown.connect
def shutdown_document_worker(**_kwargs: object) -> None:
    """Release each constructed resource once and close the persistent loop."""
    global _WORKER_RESOURCES, _WORKER_RUNNER  # noqa: PLW0603
    resources = _WORKER_RESOURCES
    runner = _WORKER_RUNNER
    _WORKER_RESOURCES = None
    _WORKER_RUNNER = None
    if resources is None:
        logger.info("Document ingestion worker had no resources to close")
        if runner is not None:
            runner.close()
        return
    try:
        if runner is None:
            logger.error("Document ingestion worker loop absent during resource shutdown")
            return
        runner.run(_release_document_worker(resources))
        logger.info("Document ingestion worker resources closed")
    finally:
        if runner is not None:
            runner.close()
