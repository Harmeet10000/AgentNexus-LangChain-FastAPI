"""Shared checkpoint-to-transcript synchronization boundary.

Both the live turn recorder and the recovery sweep supply their independently
acquired evidence/context here. Projection and persistence policy stay single-
source so a sweep cannot derive different transcript rows from a live turn.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

from returns.result import Failure, Success

from app.features.agent_saul.transcript.projection import project
from app.features.agent_saul.transcript.repository import ApplyOutcome

_ASSISTANT_ROW_KINDS = frozenset({"plan_prompt", "clarification_prompt", "final_summary"})

if TYPE_CHECKING:
    from collections.abc import Mapping
    from datetime import datetime
    from typing import Any

    from app.features.agent_saul.transcript.errors import TranscriptResult
    from app.features.agent_saul.transcript.projection import Projection, SnapshotView, TurnRows


class ProjectionRepository(Protocol):
    """Minimum persistence seam required by transcript synchronization."""

    async def apply_projection(
        self,
        projection: Projection,
        *,
        user_id: str,
        session_id: str,
        usage: Mapping[str, int] | None,
        usage_metadata: Mapping[str, Mapping[str, Any]] | None,
        prior_checkpoint_id: str | None,
    ) -> TranscriptResult[ApplyOutcome]: ...


@dataclass(frozen=True)
class PreparedSnapshot:
    """Pure projection plus the side-input maps that must be applied with it."""

    projection: Projection
    usage: Mapping[str, int] | None = None
    usage_metadata: Mapping[str, Mapping[str, Any]] | None = None


@dataclass(frozen=True)
class SnapshotSync:
    projection: Projection
    applied: ApplyOutcome


def prepare_snapshot(
    *,
    view: SnapshotView,
    existing_keys: frozenset[str],
    turn_rows: TurnRows,
    now: datetime,
    aliveness: bool,
    usage_total: int = 0,
    usage_metadata: Mapping[str, Any] | None = None,
) -> PreparedSnapshot:
    """Purely derive one snapshot and prepare its fill-once usage side input."""
    projection = project(
        view,
        existing_keys=existing_keys,
        turn_rows=turn_rows,
        now=now,
        aliveness=aliveness,
    )
    if projection.is_no_op or (not usage_total and not usage_metadata):
        return PreparedSnapshot(projection=projection)
    usage_key = next(
        (row.source_key for row in reversed(projection.rows) if row.kind in _ASSISTANT_ROW_KINDS),
        next(
            (
                row.source_key
                for row in reversed(turn_rows.rows)
                if row.kind in _ASSISTANT_ROW_KINDS
            ),
            None,
        ),
    )
    if usage_key is None:
        return PreparedSnapshot(projection=projection)
    # A zero-token segment can still carry tool-call facts. Keep the zero in
    # the usage map so a metadata-only segment remains a durable usage target;
    # repository application is fill-once and therefore replay-safe.
    usage = {usage_key: usage_total}
    metadata = {usage_key: dict(usage_metadata)} if usage_metadata is not None else None
    return PreparedSnapshot(
        projection=projection,
        usage=usage,
        usage_metadata=metadata,
    )


async def apply_snapshot(
    *,
    repo: ProjectionRepository,
    prepared: PreparedSnapshot,
    user_id: str,
    session_id: str,
    prior_checkpoint_id: str | None,
) -> TranscriptResult[SnapshotSync]:
    """Atomically apply a prepared snapshot without masking repository bugs."""
    if prepared.projection.is_no_op:
        noop = prepared.projection.no_op
        return Success(
            SnapshotSync(
                projection=prepared.projection,
                applied=ApplyOutcome(
                    applied=False,
                    reason=f"noop:{noop.reason if noop is not None else 'unspecified'}",
                ),
            )
        )
    result = await repo.apply_projection(
        prepared.projection,
        user_id=user_id,
        session_id=session_id,
        usage=prepared.usage,
        usage_metadata=prepared.usage_metadata,
        prior_checkpoint_id=prior_checkpoint_id,
    )
    if isinstance(result, Failure):
        return result
    return Success(SnapshotSync(projection=prepared.projection, applied=result.unwrap()))
