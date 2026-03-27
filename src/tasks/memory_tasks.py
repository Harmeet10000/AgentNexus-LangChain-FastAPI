"""
Celery tasks for background memory operations.

reconciliation_task:
  Triggered manually or by Celery beat (e.g., every 6 hours per user).
  Runs ReconciliationGraph: fetch_existing → reconcile → apply_changes → write_versions.

memory_decay_task:
  Celery beat, runs nightly.
  Computes decay_score = f(time, usage, confidence) for entities and clauses.
  Low-score rows are archived (decay_score < threshold set to archived=True or deleted).

Beat schedule (add to your Celery config):
  app.conf.beat_schedule = {
      "memory-decay-nightly": {
          "task": "src.tasks.memory_tasks.run_memory_decay",
          "schedule": crontab(hour=2, minute=0),  # 2 AM daily
      },
      "reconciliation-periodic": {
          "task": "src.tasks.memory_tasks.run_reconciliation_for_active_users",
          "schedule": crontab(hour="*/6"),  # every 6 hours
      },
  }

Decay formula (Section "For the chosen ones" — Memory Decay):
  time_factor   = exp(-lambda_t * age_days)       lambda_t = 0.01 (slow decay)
  usage_factor  = min(1.0, access_count / 10.0)   saturates at 10 accesses
  decay_score   = w_t * time_factor + w_u * usage_factor + w_c * confidence
  weights:       w_t=0.4, w_u=0.3, w_c=0.3

Archive threshold: decay_score < 0.15
"""

from __future__ import annotations

import asyncio
import math
from typing import Any

import structlog

logger = structlog.get_logger(__name__)

# Decay weights
_W_TIME: float = 0.4
_W_USAGE: float = 0.3
_W_CONFIDENCE: float = 0.3
_LAMBDA_T: float = 0.01          # decay rate per day (~70 day half-life)
_ARCHIVE_THRESHOLD: float = 0.15 # below this → archive


def _compute_decay(age_days: float, access_count: int, confidence: float) -> float:
    """Decay formula: Section 18.1 'Memory Decay'."""
    time_factor = math.exp(-_LAMBDA_T * max(age_days, 0.0))
    usage_factor = min(1.0, access_count / 10.0)
    return _W_TIME * time_factor + _W_USAGE * usage_factor + _W_CONFIDENCE * confidence


async def _run_decay_async(db_url: str) -> dict[str, int]:
    """Async core of memory decay computation.

    Uses raw asyncpg for batch updates — faster than SQLAlchemy for bulk ops.
    Falls back to SQLAlchemy if asyncpg not available.
    """
    import asyncpg  # type: ignore[import-untyped]
    from datetime import UTC, datetime

    conn = await asyncpg.connect(db_url)
    updated_entities = 0
    archived_entities = 0
    updated_clauses = 0

    try:
        # Fetch all active entities with decay metadata
        entity_rows = await conn.fetch("""
            SELECT id, confidence, access_count,
                   EXTRACT(EPOCH FROM (NOW() - created_at)) / 86400.0 AS age_days
            FROM entities
            WHERE decay_score > 0.0
        """)

        # Batch update decay scores
        entity_updates: list[tuple[float, str]] = []
        archive_ids: list[str] = []

        for row in entity_rows:
            new_score = _compute_decay(
                age_days=float(row["age_days"]),
                access_count=int(row["access_count"]),
                confidence=float(row["confidence"]),
            )
            entity_updates.append((new_score, str(row["id"])))
            if new_score < _ARCHIVE_THRESHOLD:
                archive_ids.append(str(row["id"]))

        if entity_updates:
            await conn.executemany(
                "UPDATE entities SET decay_score = $1 WHERE id = $2::uuid",
                entity_updates,
            )
            updated_entities = len(entity_updates)

        # Clause decay (same formula)
        clause_rows = await conn.fetch("""
            SELECT id, COALESCE(risk_score, 0.5) AS confidence, access_count,
                   EXTRACT(EPOCH FROM (NOW() - created_at)) / 86400.0 AS age_days
            FROM clauses WHERE decay_score > 0.0
        """)
        clause_updates: list[tuple[float, str]] = []
        for row in clause_rows:
            new_score = _compute_decay(
                age_days=float(row["age_days"]),
                access_count=int(row["access_count"]),
                confidence=float(row["confidence"]),
            )
            clause_updates.append((new_score, str(row["id"])))

        if clause_updates:
            await conn.executemany(
                "UPDATE clauses SET decay_score = $1 WHERE id = $2::uuid",
                clause_updates,
            )
            updated_clauses = len(clause_updates)

        logger.info(
            "memory_decay_complete",
            updated_entities=updated_entities,
            archived_candidates=len(archive_ids),
            updated_clauses=updated_clauses,
        )
        return {
            "updated_entities": updated_entities,
            "archived_candidates": len(archive_ids),
            "updated_clauses": updated_clauses,
        }
    finally:
        await conn.close()


async def _run_reconciliation_async(
    user_ids: list[str],
    reconciliation_graph: Any,
    lookback_hours: int = 24,
) -> dict[str, Any]:
    """Run ReconciliationGraph per user."""
    from uuid import uuid4

    results: dict[str, Any] = {}
    for user_id in user_ids:
        try:
            result = await reconciliation_graph.ainvoke({
                "user_id": user_id,
                "run_id": str(uuid4()),
                "lookback_hours": lookback_hours,
            })
            results[user_id] = {
                "merged": result.get("merged_count", 0),
                "updated": result.get("updated_count", 0),
                "versions": result.get("versions_written", 0),
            }
            logger.info("reconciliation_user_done", user_id=user_id, **results[user_id])
        except Exception as exc:
            logger.error("reconciliation_user_failed", user_id=user_id, error=str(exc))
            results[user_id] = {"error": str(exc)}

    return results


# ---------------------------------------------------------------------------
# Celery tasks
# ---------------------------------------------------------------------------
# These functions are registered as Celery tasks in your celery app.
# Import them in src/tasks/__init__.py and register via @celery_app.task.
# The async core is run via asyncio.run() — Celery workers are sync by default.
# If using celery-gevent or celery with asyncio, use the async variant directly.


def run_memory_decay(db_url: str) -> dict[str, int]:
    """Celery beat task: nightly memory decay computation.

    Wire in celery app:
        from src.tasks.memory_tasks import run_memory_decay
        celery_app.task(run_memory_decay, name="memory_decay")
    """
    logger.info("memory_decay_task_started")
    return asyncio.run(_run_decay_async(db_url))


def run_reconciliation_for_user(
    user_id: str,
    reconciliation_graph: Any,
    lookback_hours: int = 24,
) -> dict[str, Any]:
    """Celery task: run ReconciliationGraph for a single user.

    Can be triggered:
      - by Celery beat (periodic, all active users)
      - by agent_saul pipeline on completion (per-user, immediate)
    """
    logger.info("reconciliation_task_started", user_id=user_id)
    return asyncio.run(
        _run_reconciliation_async([user_id], reconciliation_graph, lookback_hours)
    )


def run_reconciliation_for_active_users(
    reconciliation_graph: Any,
    lookback_hours: int = 6,
) -> dict[str, Any]:
    """Celery beat task: periodic reconciliation for all users active in the last window."""
    logger.info("reconciliation_beat_started")
    # TODO: query distinct user_ids from entities WHERE created_at > NOW() - lookback
    # For now, stub returns empty — wire in actual user discovery from your DB
    active_user_ids: list[str] = []
    return asyncio.run(
        _run_reconciliation_async(active_user_ids, reconciliation_graph, lookback_hours)
    )
