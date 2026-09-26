"""Celery child-process lifecycle for document ingestion resources."""

from __future__ import annotations

import asyncio
import sys
from dataclasses import dataclass
from typing import TYPE_CHECKING

from celery.signals import worker_process_init, worker_process_shutdown

from app.config import get_settings
from app.connections import init_db
from app.connections.redis import create_redis_client
from app.features.documents.service import process_document_ingestion
from app.lifecycle.graphs import graphiti_service, provide_document_ingestion_graph
from app.shared.langchain_layer.agents.tools.idempotency import IdempotencyGuard
from app.shared.rag.graphiti import close_graphiti, setup_graphiti, setup_graphiti_indices
from app.shared.rag.langextract.service import AsyncExtractionService
from app.shared.services.storage import StorageService
from app.utils import ServiceUnavailableException, logger

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Coroutine, Sequence
    from typing import Any

    from graphiti_core import Graphiti
    from langgraph.graph.state import CompiledStateGraph
    from redis.asyncio import Redis
    from sqlalchemy.ext.asyncio import AsyncEngine, async_sessionmaker


@dataclass(frozen=True)
class DocumentWorkerResources:
    """Resources bound to one worker child and its persistent event loop."""

    engine: AsyncEngine
    session_local: async_sessionmaker[Any]
    graphiti: Graphiti
    ingestion_graph: CompiledStateGraph[Any]
    redis: Redis | None = None


_WORKER_RUNNER: asyncio.Runner | None = None
_WORKER_RESOURCES: DocumentWorkerResources | None = None


def _worker_consumes_queue(queue_name: str, argv: Sequence[str] | None = None) -> bool:
    """Return whether this worker command consumes ``queue_name``.

    Celery does not pass the consumer object to ``worker_process_init``. Pool
    children do retain the worker command line, so an explicit ``-Q``/``--queues``
    is the narrow source of truth. A command without a queue flag consumes the
    app's configured queues and must therefore provision conservatively.
    """
    arguments = list(argv if argv is not None else sys.argv)
    configured: list[str] = []
    for index, argument in enumerate(arguments):
        if argument in {"-Q", "--queues"} and index + 1 < len(arguments):
            configured.extend(arguments[index + 1].split(","))
        elif argument.startswith("--queues="):
            configured.extend(argument.partition("=")[2].split(","))
        elif argument.startswith("-Q") and len(argument) > 2:
            configured.extend(argument[2:].split(","))
    if not configured:
        return True
    return queue_name in {name.strip() for name in configured if name.strip()}


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
        ingestion_graph = provide_document_ingestion_graph(
            settings=settings,
            object_store=object_store,
            graphiti=graphiti,
            ingest_document_fn=process_document_ingestion,
            extraction=AsyncExtractionService.from_settings(settings),
            graph_writer=graphiti_service(graphiti),
            idempotency=(
                IdempotencyGuard(redis=redis, db_engine=engine) if redis is not None else None
            ),
        )
        return DocumentWorkerResources(
            engine=engine,
            session_local=session_local,
            graphiti=graphiti,
            ingestion_graph=ingestion_graph,
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
    """Provision once in each forked child, never in the Celery parent."""
    global _WORKER_RESOURCES, _WORKER_RUNNER  # noqa: PLW0603
    _WORKER_RESOURCES = None
    ingestion_queue = getattr(get_settings(), "CELERY_INGESTION_QUEUE", "ingestion")
    if not _worker_consumes_queue(ingestion_queue):
        logger.bind(queue=ingestion_queue).info(
            "Document ingestion resources skipped for unrelated worker"
        )
        return
    _WORKER_RUNNER = asyncio.Runner()
    try:
        _WORKER_RESOURCES = _WORKER_RUNNER.run(_provision_document_worker())
    except Exception as exc:  # noqa: BLE001 — failed optional capability degrades the worker
        logger.bind(error_type=type(exc).__name__).exception(
            "Document ingestion worker provisioning failed"
        )
        _WORKER_RUNNER.close()
        _WORKER_RUNNER = None
    else:
        logger.info("Document ingestion worker resources initialized")


def get_document_worker_resources() -> DocumentWorkerResources:
    """Return provisioned resources or a typed, capability-naming failure."""
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
    if _WORKER_RUNNER is None:
        get_document_worker_resources()
        message = "Worker resource guard returned without a runner"
        raise AssertionError(message)
    return _WORKER_RUNNER.run(coroutine_factory())


@worker_process_shutdown.connect
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
