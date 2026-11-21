"""Import-safe composition descriptor for the Agent Saul slice."""

from __future__ import annotations

from typing import TYPE_CHECKING

from app.shared.module import Module

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from typing import Any


def build_consumers() -> Sequence[Any]:
    """Bind Agent Saul-owned Celery consumers only in a worker process."""
    from .adapters.inbound.worker.tasks import (  # noqa: PLC0415
        agent_memory_consolidation,
        chat_transcript_sweep,
    )

    return (agent_memory_consolidation, chat_transcript_sweep)


def build_schedules() -> Mapping[str, Any]:
    """Build Agent Saul periodic work only in the scheduler process."""
    from celery.schedules import crontab  # noqa: PLC0415

    from app.shared.jobs.names import (  # noqa: PLC0415
        AGENT_MEMORY_CONSOLIDATION,
        AGENT_SAUL_TRANSCRIPT_SWEEP,
    )

    return {
        "agent-memory-consolidation-nightly": {
            "task": AGENT_MEMORY_CONSOLIDATION,
            "schedule": crontab(hour=3, minute=30),
            "kwargs": {"tenant_ids": []},
        },
        "agent-saul-transcript-sweep-periodic": {
            "task": AGENT_SAUL_TRANSCRIPT_SWEEP,
            "schedule": crontab(minute="*/15"),
            "options": {"expires": 60},
        },
    }


MODULE = Module(
    name="agent_saul",
    consumers=(build_consumers,),
    schedules=(build_schedules,),
)
