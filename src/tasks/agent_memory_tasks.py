"""Scheduled agent-memory consolidation (band F group 9).

Registration only — there is no worker and no beat service in the compose stack
(NG14, coordination point C-B), so this task is **inert on the day it lands**.
Nothing here may be read as evidence that a consolidation has ever run; group 10's
manual round-trip is the only such evidence available.

The consolidation itself refuses loudly when the target graph lacks its required
procedures (:data:`ConsolidationPreconditionError`), which is the branch every
current environment takes.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from pydantic import Field

from app.config import get_settings
from app.connections.celery import celery_app
from app.connections.celery_registry import CeleryTaskPayload, CeleryTaskRegistry
from app.connections.celery_task_names import AGENT_MEMORY_CONSOLIDATION
from app.shared.langchain_layer.agents.memory.agent_memory_service import (
    AgentMemoryService,
    memory_partition,
)
from app.utils import logger

if TYPE_CHECKING:
    from typing import Any


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
    service = AgentMemoryService(
        partition_prefix=settings.COGNEE_DATASET_PREFIX,
        pending_sessions=set(session_ids or []),
    )

    consolidated: list[str] = []
    refused: list[dict[str, str]] = []
    for tenant_id in resolved_tenants:
        try:
            asyncio.run(service.consolidate(tenant_ids=[tenant_id]))
            consolidated.append(tenant_id)
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
