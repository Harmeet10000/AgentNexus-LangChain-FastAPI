"""Scheduled work for the Agent Saul feature.

`agent_memory_consolidation` — nightly memory consolidation (band F).

`chat_transcript_sweep` — transcript reconciliation sweep (doc §6 trigger 3)
and checkpoint GC (§8): bounded batches, projector-only status changes, and
the §6 [v3] Redis-outage guard (derive nothing when aliveness is unknowable).
"""

from __future__ import annotations

import asyncio
from dataclasses import asdict
from typing import TYPE_CHECKING

from neo4j import AsyncGraphDatabase, basic_auth
from pydantic import BaseModel, ConfigDict, SecretStr

from app.ai.memory.agent_memory_service import (
    AgentMemoryError,
    AgentMemoryService,
    make_neo4j_procedures_probe,
    memory_partition,
)
from app.config import get_settings
from app.connections.checkpointer import setup_langgraph_checkpointer
from app.connections.postgres import init_db
from app.connections.redis import close_redis_client, create_redis_client
from app.features.agent_saul.application.reconcile import build_saul_read_graph, sweep_transcripts
from app.features.agent_saul.application.transcript.repository import ChatTranscriptRepository
from app.observability.logging import logger
from app.shared.jobs.names import AGENT_MEMORY_CONSOLIDATION, AGENT_SAUL_TRANSCRIPT_SWEEP
from app.worker.celery.app import get_worker_celery_app
from app.worker.celery.task import ResilientTask
from app.worker.execution.runtime import run

celery_app = get_worker_celery_app()

if TYPE_CHECKING:
    from typing import Any

    from neo4j import AsyncDriver

    from app.config import Settings
    from app.features.agent_saul.application.reconcile import SweepReport


class Neo4jCredentials(BaseModel):
    """Narrow frozen credentials for the consolidation probe's Neo4j driver."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    uri: str
    username: str
    password: SecretStr


@celery_app.task(
    name=AGENT_MEMORY_CONSOLIDATION,
    bind=True,
    base=ResilientTask,
    max_retries=0,
)
def agent_memory_consolidation(
    _self: ResilientTask,
    *,
    tenant_ids: list[str] | None = None,
    session_ids: list[str] | None = None,
) -> dict[str, Any]:
    """Consolidate conversation-scoped memory into the permanent graph.

    Runs nightly via beat. Each tenant's conversation partition is consolidated
    once; a precondition failure on any tenant is logged and recorded in the
    result rather than silently swallowed.
    """
    settings = get_settings()
    resolved_tenants = tenant_ids or []
    return run(
        _consolidate_async(
            settings=settings,
            resolved_tenants=resolved_tenants,
            session_ids=session_ids,
        )
    )


def _connect_graph_driver(credentials: Neo4jCredentials) -> AsyncDriver | None:
    """Open the consolidation probe's own Neo4j driver, or None when unavailable.

    Driver creation is lazy (no I/O), so a failure here means misconfiguration,
    never an unreachable host — the probe itself reports unreachability as False.
    """
    try:
        return AsyncGraphDatabase.driver(
            credentials.uri,
            auth=basic_auth(
                credentials.username,
                credentials.password.get_secret_value(),
            ),
        )
    except Exception as exc:  # noqa: BLE001 — no driver means the probe answers False
        exc.add_note("task=agent_memory_consolidation, stage=probe_driver")
        logger.bind(error=str(exc)).warning("agent_memory_probe_driver_unavailable")
        return None


async def _consolidate_async(
    *,
    settings: Settings,
    resolved_tenants: list[str],
    session_ids: list[str] | None,
) -> dict[str, Any]:
    """Consolidate each tenant with a live procedures probe, then close the driver."""
    driver = _connect_graph_driver(
        Neo4jCredentials(
            uri=settings.NEO4J_URI,
            username=settings.NEO4J_USERNAME,
            password=settings.NEO4J_PASSWORD,
        )
    )
    service = AgentMemoryService(
        partition_prefix=settings.COGNEE_DATASET_PREFIX,
        pending_sessions=set(session_ids or []),
        procedures_probe=make_neo4j_procedures_probe(driver),
    )
    try:
        consolidated: list[str] = []
        refused: list[dict[str, str]] = []
        for tenant_id in resolved_tenants:
            try:
                await service.consolidate(tenant_ids=[tenant_id])
                consolidated.append(tenant_id)
            except AgentMemoryError as exc:
                # Named precondition/identity refusal: logged and recorded
                # with its typed name, never silent, never stopping the rest.
                exc.add_note(f"task=agent_memory_consolidation, tenant={tenant_id}")
                logger.bind(tenant=tenant_id, error=str(exc)).warning(
                    "agent_memory_consolidation_refused"
                )
                refused.append({"tenant": tenant_id, "error": type(exc).__name__})
            except Exception as exc:  # noqa: BLE001 — one tenant's refusal must not stop the rest
                exc.add_note(f"task=agent_memory_consolidation, tenant={tenant_id}")
                logger.bind(tenant=tenant_id, error=str(exc)).warning(
                    "agent_memory_consolidation_refused"
                )
                refused.append({"tenant": tenant_id, "error": type(exc).__name__})

        logger.bind(
            consolidated=len(consolidated),
            refused=len(refused),
        ).info("agent_memory_consolidation_finished")
        return {
            "consolidated": consolidated,
            "refused": refused,
            "partitions": [
                memory_partition(tenant_id=t, kind="reports", prefix=settings.COGNEE_DATASET_PREFIX)
                for t in resolved_tenants
            ],
        }
    finally:
        if driver is not None:
            try:
                await driver.close()
            except Exception as exc:  # noqa: BLE001 — best-effort cleanup after results are recorded
                exc.add_note("task=agent_memory_consolidation, stage=probe_driver_close")
                logger.bind(error=str(exc)).warning("agent_memory_probe_driver_close_failed")


@celery_app.task(
    name=AGENT_SAUL_TRANSCRIPT_SWEEP,
    bind=True,
    base=ResilientTask,
    max_retries=0,
)
def chat_transcript_sweep(_self: ResilientTask) -> dict[str, Any]:
    """Bounded transcript sweep + checkpoint GC (doc §6 trigger 3, §8).

    Never raises: the sweep is a backstop, not a pipeline stage — the report
    is the result, and the next beat retries after any partial outcome.
    """
    return run(_transcript_sweep_async())


async def _transcript_sweep_async() -> dict[str, Any]:
    """Provision the sweep's own resources, run one bounded pass, tear down.

    Setup runs INSIDE the protected block and the teardown closes every
    resource that was successfully created: a ``setup_langgraph_checkpointer``
    failure after ``init_db`` must not leak the already-open SQLAlchemy
    engine — each scheduled attempt would otherwise grow this worker
    process's pools without bound.
    """
    settings: Settings = get_settings()
    engine: object | None = None
    redis_client: object | None = None
    checkpointer: object | None = None
    try:
        engine, session_local = await init_db()
        redis_client = create_redis_client(settings.REDIS_URL)
        checkpointer = await setup_langgraph_checkpointer()
        graph = build_saul_read_graph(checkpointer)
        repo = ChatTranscriptRepository(session_local)
        report: SweepReport = await sweep_transcripts(
            repo=repo,
            graph=graph,
            redis=redis_client,
            checkpointer=checkpointer,
        )
        return asdict(report)
    except Exception as exc:  # noqa: BLE001 — backstop task never raises into Celery retries
        exc.add_note("task=chat_transcript_sweep")
        logger.bind(error=str(exc)).error("transcript_sweep_task_failed")
        return {"skipped": "sweep_task_error", "error": type(exc).__name__}
    finally:
        await asyncio.gather(
            _aclose_checkpointer(checkpointer),
            close_redis_client(redis_client),
            _dispose_engine(engine),
            return_exceptions=True,
        )


async def _aclose_checkpointer(checkpointer: object) -> None:
    pool = getattr(checkpointer, "conn", None)
    if pool is not None:
        await pool.close()


async def _dispose_engine(engine: object) -> None:
    dispose = getattr(engine, "dispose", None)
    if dispose is not None:
        await dispose()
