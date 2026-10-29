"""Celery child-process lifecycle for document ingestion resources."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any, override

from celery.signals import worker_process_init, worker_process_shutdown, worker_shutdown
from graphiti_core import Graphiti
from langgraph.graph.state import CompiledStateGraph
from pydantic import BaseModel, ConfigDict
from redis.asyncio import Redis
from sqlalchemy.ext.asyncio import (
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
from app.extraction.langextract.service import AsyncExtractionService
from app.features.documents.service import process_document_ingestion
from app.lifecycle.graphs import graphiti_service, provide_document_ingestion_graph
from app.observability.logging import logger
from app.resilience.idempotency import IdempotencyGuard
from app.utils import ServiceUnavailableException
from app.utils.storage import StorageService
from app.worker.capabilities.owner import (
    CapabilityLifecycle,
    CapabilityOwner,
    CapabilityUsageError,
)

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
            exc.add_note(f"capability=document, operation=resource_close, resource={resource}")
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
        redis_url = settings.REDIS_URL
        if redis_url:
            redis = create_redis_client(redis_url)
            await redis.ping()
        graphiti = await setup_graphiti(
            neo4j_uri=settings.NEO4J_URI,
            neo4j_user=settings.NEO4J_USERNAME,
            neo4j_password=settings.NEO4J_PASSWORD.get_secret_value(),
            neo4j_database=settings.NEO4J_DATABASE,
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
    except Exception as exc:
        exc.add_note("capability=document, operation=provision_cleanup")
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


class _DocumentCapability(CapabilityLifecycle[DocumentWorkerResources]):
    """Document adapter for the generic process/loop capability owner."""

    @override
    async def provision(self) -> DocumentWorkerResources:
        return await _provision_document_worker()

    @override
    async def release(self, resources: DocumentWorkerResources) -> None:
        async with asyncio.timeout(30):
            await _release_document_worker(resources)


_DOCUMENT_OWNER = CapabilityOwner(_DocumentCapability())


def _sync_document_compat_state() -> None:
    """Mirror owner state for older tests and external worker callers."""
    global _WORKER_PID, _WORKER_RUNNER, _WORKER_RESOURCES
    _WORKER_PID, _WORKER_RUNNER, _WORKER_RESOURCES = _DOCUMENT_OWNER._state()


@worker_process_init.connect
def initialize_document_worker(**_kwargs: object) -> None:
    """Reset inherited state; first task provisions resources after child startup."""
    _DOCUMENT_OWNER.reset_after_fork()
    _sync_document_compat_state()


def _ensure_document_worker() -> None:
    """Lazily provision once on the owning child's persistent loop."""
    try:
        _DOCUMENT_OWNER.ensure()
    except CapabilityUsageError:
        raise
    except Exception as exc:
        exc.add_note("capability=document, operation=worker_provision")
        _sync_document_compat_state()
        raise ServiceUnavailableException(detail="Document worker provisioning failed") from exc
    else:
        _sync_document_compat_state()
        logger.info("Document ingestion worker resources initialized")


def get_document_worker_resources() -> DocumentWorkerResources:
    """Return provisioned resources or a typed, capability-naming failure."""
    _ensure_document_worker()
    resources = _WORKER_RESOURCES
    if resources is None:
        raise ServiceUnavailableException(
            detail="Document ingestion worker is unavailable",
            data={"capability": "document_ingestion_graph"},
        )
    return resources


def run_on_document_worker_loop[T](
    coroutine_factory: Callable[[], Coroutine[Any, Any, T]],
) -> T:
    """Run task I/O on the same loop that created the process resources.

    A soft time limit escapes ``run_until_complete`` without cancelling the
    coroutine, which would otherwise resume on this loop alongside the next
    task while still holding DB sessions. Cancel and drain it first.
    """
    get_document_worker_resources()
    result = _DOCUMENT_OWNER.run(coroutine_factory)
    _sync_document_compat_state()
    return result


@worker_process_shutdown.connect
@worker_shutdown.connect
def shutdown_document_worker(**_kwargs: object) -> None:
    """Release each constructed resource once and close the persistent loop."""
    if _DOCUMENT_OWNER.peek() is None:
        logger.info("Document ingestion worker had no resources to close")
    try:
        _DOCUMENT_OWNER.shutdown()
        logger.info("Document ingestion worker resources closed")
    except CapabilityUsageError:
        raise
    except Exception as exc:  # noqa: BLE001 — shutdown must never raise
        exc.add_note("operation=shutdown_document_worker")
        logger.bind(error_type=type(exc).__name__).warning("document_worker_close_failed")
    finally:
        _sync_document_compat_state()
