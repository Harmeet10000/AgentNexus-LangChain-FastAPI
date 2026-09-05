"""Scheduled agent-memory consolidation (band F group 9).

The consolidation refuses loudly when the target graph lacks its required
procedures (:data:`ConsolidationPreconditionError`), which is the branch every
environment without APOC/GDS takes. The service is constructed here with a
live SHOW PROCEDURES probe over its own Neo4j driver, so an environment whose
graph does expose the procedures can actually consolidate; a driver or probe
failure still answers False and keeps the refusal branch.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from neo4j import AsyncGraphDatabase, basic_auth
from pydantic import Field

from app.config import get_settings
from app.connections.celery import CeleryTaskPayload, CeleryTaskRegistry, celery_app
from app.connections.celery_task_names import AGENT_MEMORY_CONSOLIDATION
from app.shared.langchain_layer.agents.memory.agent_memory_service import (
    AgentMemoryError,
    AgentMemoryService,
    make_neo4j_procedures_probe,
    memory_partition,
)
from app.utils import logger

if TYPE_CHECKING:
    from typing import Any

    from neo4j import AsyncDriver

    from app.config import Settings


class AgentMemoryConsolidationPayload(CeleryTaskPayload):
    """Tenant ids to consolidate; empty means the scheduler's default sweep."""

    tenant_ids: list[str] = Field(default_factory=list)


CeleryTaskRegistry.register(AGENT_MEMORY_CONSOLIDATION, AgentMemoryConsolidationPayload)


@celery_app.task(
    name="tasks.agent_memory_consolidation",
    bind=True,
    max_retries=0,
)
def agent_memory_consolidation(
    _self: Any,
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
    return asyncio.run(
        _consolidate_async(
            settings=settings,
            resolved_tenants=resolved_tenants,
            session_ids=session_ids,
        )
    )


def _connect_graph_driver(settings: Settings) -> AsyncDriver | None:
    """Open the consolidation probe's own Neo4j driver, or None when unavailable.

    Driver creation is lazy (no I/O), so a failure here means misconfiguration,
    never an unreachable host — the probe itself reports unreachability as False.
    """
    try:
        return AsyncGraphDatabase.driver(
            settings.NEO4J_URI,
            auth=basic_auth(
                settings.NEO4J_USERNAME,
                settings.NEO4J_PASSWORD.get_secret_value(),
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
    driver = _connect_graph_driver(settings)
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
