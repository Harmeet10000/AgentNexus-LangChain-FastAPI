"""Background helpers for memory decay and reconciliation workflows."""

from __future__ import annotations

import asyncio
import math
from typing import TYPE_CHECKING, Protocol, TypedDict
from uuid import uuid4

import asyncpg

from app.utils import logger

if TYPE_CHECKING:
    from collections.abc import Mapping

_W_TIME = 0.4
_W_USAGE = 0.3
_W_CONFIDENCE = 0.3
_LAMBDA_T = 0.01
_ARCHIVE_THRESHOLD = 0.15


class DecayStats(TypedDict):
    """Summary returned by the decay workflow."""

    updated_entities: int
    archived_candidates: int
    updated_clauses: int


class ReconciliationSummary(TypedDict, total=False):
    """Per-user reconciliation result."""

    merged: int
    updated: int
    versions: int
    error: str


class ReconciliationGraph(Protocol):
    """Minimal async interface required by the reconciliation task helpers."""

    async def ainvoke(
        self,
        payload: dict[str, str | int],
    ) -> Mapping[str, object]:
        """Run the reconciliation graph for a single user."""


def _compute_decay(age_days: float, access_count: int, confidence: float) -> float:
    """Compute the weighted decay score for an entity or clause."""
    bounded_confidence = max(0.0, min(confidence, 1.0))
    bounded_age = max(age_days, 0.0)
    bounded_access_count = max(access_count, 0)

    time_factor = math.exp(-_LAMBDA_T * bounded_age)
    usage_factor = min(1.0, bounded_access_count / 10.0)
    return (_W_TIME * time_factor) + (_W_USAGE * usage_factor) + (
        _W_CONFIDENCE * bounded_confidence
    )


async def _run_decay_async(db_url: str) -> DecayStats:
    """Recompute decay scores for memory rows using asyncpg bulk updates."""

    conn = await asyncpg.connect(db_url)
    updated_entities = 0
    archived_candidates = 0
    updated_clauses = 0

    try:
        entity_rows = await conn.fetch(
            """
            SELECT id, confidence, access_count,
                   EXTRACT(EPOCH FROM (NOW() - created_at)) / 86400.0 AS age_days
            FROM entities
            WHERE decay_score > 0.0
            """
        )

        entity_updates: list[tuple[float, str]] = []
        for row in entity_rows:
            new_score = _compute_decay(
                age_days=float(row["age_days"]),
                access_count=int(row["access_count"]),
                confidence=float(row["confidence"]),
            )
            entity_updates.append((new_score, str(row["id"])))
            if new_score < _ARCHIVE_THRESHOLD:
                archived_candidates += 1

        if entity_updates:
            await conn.executemany(
                "UPDATE entities SET decay_score = $1 WHERE id = $2::uuid",
                entity_updates,
            )
            updated_entities = len(entity_updates)

        clause_rows = await conn.fetch(
            """
            SELECT id, COALESCE(risk_score, 0.5) AS confidence, access_count,
                   EXTRACT(EPOCH FROM (NOW() - created_at)) / 86400.0 AS age_days
            FROM clauses
            WHERE decay_score > 0.0
            """
        )

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

        logger.bind(
            updated_entities=updated_entities,
            archived_candidates=archived_candidates,
            updated_clauses=updated_clauses,
        ).info("Memory decay completed")

        if archived_candidates:
            logger.bind(archived_candidates=archived_candidates).warning(
                "Archive candidates detected, but no archive column exists in the current schema"
            )

        return {
            "updated_entities": updated_entities,
            "archived_candidates": archived_candidates,
            "updated_clauses": updated_clauses,
        }
    finally:
        await conn.close()


async def _run_reconciliation_async(
    user_ids: list[str],
    reconciliation_graph: ReconciliationGraph,
    lookback_hours: int = 24,
) -> dict[str, ReconciliationSummary]:
    """Run reconciliation sequentially for each user id."""
    results: dict[str, ReconciliationSummary] = {}

    for user_id in user_ids:
        try:
            result = await reconciliation_graph.ainvoke(
                {
                    "user_id": user_id,
                    "run_id": str(uuid4()),
                    "lookback_hours": lookback_hours,
                }
            )
            user_result: ReconciliationSummary = {
                "merged": int(result.get("merged_count", 0)),
                "updated": int(result.get("updated_count", 0)),
                "versions": int(result.get("versions_written", 0)),
            }
            results[user_id] = user_result
            logger.bind(user_id=user_id, **user_result).info(
                "User reconciliation completed"
            )
        except Exception as exc:  # noqa: BLE001
            error_message = str(exc)
            results[user_id] = {"error": error_message}
            logger.bind(user_id=user_id, error=error_message).exception(
                "User reconciliation failed"
            )

    return results


def run_memory_decay(db_url: str) -> DecayStats:
    """Run memory decay from a synchronous task runner."""
    logger.info("Memory decay task started")
    return asyncio.run(_run_decay_async(db_url))


def run_reconciliation_for_user(
    user_id: str,
    reconciliation_graph: ReconciliationGraph,
    lookback_hours: int = 24,
) -> dict[str, ReconciliationSummary]:
    """Run reconciliation for a single user."""
    logger.bind(user_id=user_id, lookback_hours=lookback_hours).info(
        "Single-user reconciliation started"
    )
    return asyncio.run(
        _run_reconciliation_async([user_id], reconciliation_graph, lookback_hours)
    )


def run_reconciliation_for_active_users(
    reconciliation_graph: ReconciliationGraph,
    lookback_hours: int = 6,
    active_user_ids: list[str] | None = None,
) -> dict[str, ReconciliationSummary]:
    """Run reconciliation for an already-resolved set of active users."""
    user_ids = active_user_ids or []
    logger.bind(
        lookback_hours=lookback_hours,
        active_user_count=len(user_ids),
    ).info("Active-user reconciliation started")
    return asyncio.run(
        _run_reconciliation_async(user_ids, reconciliation_graph, lookback_hours)
    )


