"""ChatTranscriptRepository — the only writer of transcript rows (doc §10).

Every write enters through :meth:`apply_projection` from a snapshot-derived
projection. There is deliberately no ``close_turn``: every terminal row —
including every ``turn_closed`` row — exits ``apply_projection`` (chain-of-custody
rule, §1). The repository's one transaction moves session watermark, row
inserts, and the status cache together so batch- and item-level idempotency
(§4.3) both hold.

Errors are Results; driver exceptions are mapped to sanitized typed errors at
this edge and never cross to the service (§10).
"""

from __future__ import annotations

import re

# Runtime-required: pydantic field annotations resolve at class definition
# (SessionMeta/TranscriptMessage) — do NOT move these to TYPE_CHECKING.
from datetime import datetime
from traceback import format_exc
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict, Field
from returns.result import Failure, Success
from sqlalchemy import (
    TIMESTAMP,
    String,
    Text,
    and_,
    bindparam,
    cast,
    func,
    literal_column,
    or_,
    select,
    tuple_,
    update,
)
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.exc import SQLAlchemyError

from app.features.agent_saul.adapters.persistence.models import (
    MESSAGE_SCHEMA_VERSION,
    ChatMessage,
    ChatSession,
)
from app.features.agent_saul.application.transcript.errors import (
    TranscriptInfrastructureError,
    TranscriptNotFoundError,
    TranscriptSchemaError,
)
from app.features.agent_saul.application.transcript.projection import (
    MESSAGE_KINDS,
    TRANSCRIPT_SCHEMA_VERSION,
    VERDICT_ROW_KINDS,
    SessionStatus,
    TurnRow,
    TurnRows,
)
from app.kernel.result import add_database_error_note
from app.observability.logging import logger, trace_layer

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

    from app.features.agent_saul.application.transcript.errors import (
        TranscriptError,
        TranscriptResult,
    )
    from app.features.agent_saul.application.transcript.projection import (
        ProjectedRow,
        Projection,
    )

    # NOT moved into this block — pydantic needs them at class-definition time
    # for SessionMeta / TranscriptMessage / ApplyOutcome field annotations
    # (datetime, Any) and TranscriptTurnContext's runtime defaults (SessionStatus,
    # TurnRow, TurnRows, MESSAGE_KINDS, TRANSCRIPT_SCHEMA_VERSION); async_sessionmaker
    # is the constructor signature. They stay runtime imports below.

_SESSION_ID_LEN = 100
_USER_ID_LEN = 100
_GC_STAMP_KEY = "gc_swept_at"  # §8 GC audit stamp (extra_metadata)
_GC_PENDING_KEY = "gc_pending"  # true ⇒ deletion stamped but not yet confirmed
_GC_PENDING_RETRY_MINS = 15  # a pending stamp older than this re-enters the list
_PLAN_DECISION_KEY_RE = re.compile(r"^interrupt:[A-Za-z0-9][A-Za-z0-9._:-]{0,199}$")
_CHECKPOINT_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,255}$")
_REVIEWER_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:@-]{0,199}$")


class ApplyOutcome(BaseModel):
    """What ``apply_projection`` did in its single transaction."""

    model_config = ConfigDict(frozen=True)

    applied: bool
    reason: str = "applied"
    rows_inserted: int = 0
    turn_seq: int | None = None
    turn_opened: bool = False


class TranscriptTurnContext(BaseModel):
    """Everything the projector/recorder needs for one turn (§4.1 turn_rows)."""

    model_config = ConfigDict(frozen=True)

    existing_keys: frozenset[str] = frozenset()
    turn_rows: TurnRows = TurnRows()
    session_status: SessionStatus | None = None
    turn_count: int = 0
    last_checkpoint_id: str | None = None
    last_checkpoint_ts: datetime | None = None


class TranscriptMessage(BaseModel):
    """Read model for transcript reads/export (D3 order: turn_seq, ordinal).

    ``row_id`` carries the storage row's id so a keyset cursor can compare
    the FULL ordering key (``(turn_seq, ordinal, id)`` — the id tiebreak is
    in the ORDER BY) across page boundaries. Not part of the display payload
    contract: treat it as the cursor token.
    """

    model_config = ConfigDict(frozen=True)

    row_id: int
    turn_seq: int
    ordinal: int
    kind: str
    role: str
    content: str | None
    model: str | None = None
    tokens_used: int | None = None
    redacted: bool = False
    created_at: datetime | None = None
    extra_metadata: dict[str, Any] = Field(default_factory=dict)


class SessionMeta(BaseModel):
    """Read model for session rows (list views / export head)."""

    model_config = ConfigDict(frozen=True)

    session_id: str
    user_id: str
    title: str | None = None
    status: str = SessionStatus.ACTIVE.value
    schema_version: int = MESSAGE_SCHEMA_VERSION
    turn_count: int = 0
    last_checkpoint_id: str | None = None
    last_checkpoint_ts: datetime | None = None
    gc_pending: bool = False  # §8: deletion stamped but not yet confirmed
    created_at: datetime | None = None
    updated_at: datetime | None = None


async def _resolve_turn(
    session: AsyncSession,
    session_id: str,
    q_key: str | None,
    session_row: ChatSession,
) -> tuple[int, bool]:
    """An existing query row keys the turn; otherwise the next slot opens one."""
    if q_key:
        existing = (
            await session.execute(
                select(ChatMessage.turn_seq).where(
                    ChatMessage.session_id == session_id,
                    ChatMessage.source_key == q_key,
                )
            )
        ).scalar_one_or_none()
        if existing is not None:
            return existing, False
    return session_row.turn_count + 1, True


def _session_set_values(
    session_row: ChatSession,
    projection: Projection,
    checkpoint_id: str | None,
    turn_opened: bool,
    status: str,
) -> dict[str, Any]:
    """Build the trailing ChatSession UPDATE payload (watermark + counters)."""
    values: dict[str, Any] = {"updated_at": func.now()}
    if session_row.title is None and projection.query_source_key:
        first_query = next((row.content for row in projection.rows if row.kind == "query"), None)
        if first_query:
            values["title"] = str(first_query)[:120]
    if checkpoint_id is not None:
        values["last_checkpoint_id"] = checkpoint_id
    if projection.watermark_ts is not None:
        values["last_checkpoint_ts"] = projection.watermark_ts
    if turn_opened:
        values["turn_count"] = session_row.turn_count + 1
    values["status"] = status
    if (
        projection.watermark_checkpoint_id
        and session_row.pending_plan_decision_checkpoint_id == projection.watermark_checkpoint_id
        and any(
            (
                _plan_decision_prompt_key(row.source_key) is not None
                and _plan_decision_prompt_key(row.source_key)
                == session_row.pending_plan_decision_key
            )
            for row in projection.rows
        )
    ):
        values["pending_plan_decision_key"] = None
        values["pending_plan_decision_modified"] = None
        values["pending_plan_decision_checkpoint_id"] = None
    return values


def _plan_decision_prompt_key(source_key: str) -> str | None:
    """Return the pending-fact key proven by a decision row, if well formed."""
    prefix = "decision:"
    if not source_key.startswith(prefix):
        return None
    interrupt_id = source_key.removeprefix(prefix)
    if not interrupt_id:
        return None
    return f"interrupt:{interrupt_id}"


def _valid_plan_decision_key(decision_key: object) -> bool:
    return (
        isinstance(decision_key, str) and _PLAN_DECISION_KEY_RE.fullmatch(decision_key) is not None
    )


def _valid_checkpoint_id(checkpoint_id: object) -> bool:
    return isinstance(checkpoint_id, str) and _CHECKPOINT_ID_RE.fullmatch(checkpoint_id) is not None


def _plan_decision_row_is_authorized(
    row: ProjectedRow,
    *,
    session_row: ChatSession,
    checkpoint_id: str | None,
) -> bool:
    """Require the decision row to carry the exact proof written for its prompt.

    A ``decision:<interrupt>`` key is not authority on its own: checkpoint
    identifiers are generation fences.  This guard lives at the repository
    boundary as well as in the pure projector because recovery callers can
    supply a handcrafted projection.
    """
    if row.kind != "plan_decision":
        return True
    prompt_key = _plan_decision_prompt_key(row.source_key)
    return bool(
        prompt_key
        and session_row.pending_plan_decision_key == prompt_key
        and session_row.pending_plan_decision_checkpoint_id
        and checkpoint_id
        and session_row.pending_plan_decision_checkpoint_id == checkpoint_id
    )


async def _load_session(session: AsyncSession, session_id: str, user_id: str) -> ChatSession | None:
    return (
        await session.execute(
            select(ChatSession).where(
                ChatSession.session_id == session_id,
                ChatSession.user_id == user_id,
            )
        )
    ).scalar_one_or_none()


async def _load_session_for_update(
    session: AsyncSession,
    session_id: str,
    user_id: str,
) -> ChatSession | None:
    return (
        await session.execute(
            select(ChatSession)
            .where(
                ChatSession.session_id == session_id,
                ChatSession.user_id == user_id,
            )
            .with_for_update()
        )
    ).scalar_one_or_none()


class ChatTranscriptRepository:
    """Owns chat_session/chat_message persistence. No per-run state kept."""

    def __init__(self, session_factory: async_sessionmaker[AsyncSession]) -> None:
        self._session_factory = session_factory

    # ------------------------------------------------------------------
    # Writers
    # ------------------------------------------------------------------

    @trace_layer("repository")
    async def ensure_session(
        self,
        *,
        session_id: str,
        user_id: str,
    ) -> TranscriptResult[bool]:
        """Ensure the owner-bound discovery anchor exists.

        This is intentionally narrower than :meth:`apply_projection`: it may
        create only the identity/discovery row and never derives status,
        advances a checkpoint watermark, or writes transcript messages.
        Returns ``True`` only when this call inserted the row.
        """
        sid = session_id[:_SESSION_ID_LEN]
        uid = user_id[:_USER_ID_LEN]
        try:
            async with self._session_factory() as session, session.begin():
                inserted_id = (
                    await session.execute(
                        pg_insert(ChatSession)
                        .values(
                            session_id=sid,
                            user_id=uid,
                            schema_version=TRANSCRIPT_SCHEMA_VERSION,
                            status=SessionStatus.ACTIVE.value,
                        )
                        .on_conflict_do_nothing(index_elements=[ChatSession.session_id])
                        .returning(ChatSession.id)
                    )
                ).scalar_one_or_none()
                session_row = (
                    await session.execute(
                        select(ChatSession).where(ChatSession.session_id == sid).with_for_update()
                    )
                ).scalar_one()
                if session_row.user_id != uid:
                    logger.bind(session_id=sid).warning("transcript_owner_mismatch_refused")
                    return Failure(
                        TranscriptNotFoundError(
                            message="transcript session not found",
                            session_id=session_id,
                        )
                    )
                schema_failure = _schema_gate(
                    session_row.schema_version,
                    operation="ensure_session",
                )
                if schema_failure is not None:
                    return Failure(schema_failure)
                return Success(inserted_id is not None)
        except SQLAlchemyError as exc:
            await session.rollback()
            add_database_error_note(
                exc,
                table="chat_sessions",
                operation="ensure_session",
                context={"session_id": sid},
            )
            exc.add_note(format_exc())
            logger.bind(
                operation="ensure_session",
                session_id=sid,
                error_notes=list(getattr(exc, "__notes__", ())),
            ).warning("transcript_repository_failed")
            return Failure(
                TranscriptInfrastructureError(
                    message="transcript storage operation failed",
                    source="chat_transcript_repository",
                    operation="ensure_session",
                )
            )

    @trace_layer("repository")
    async def record_plan_decision_fact(
        self,
        *,
        session_id: str,
        user_id: str,
        decision_key: str,
        modified: bool,
        checkpoint_id: str | None = None,
    ) -> TranscriptResult[bool]:
        """Durably record validated resume intent before graph resumption.

        The prompt source key is supplied by the fresh checkpoint and written
        under the session lock. No plan text enters SQL; wrong owner fails
        closed. The prompt row may still be a recoverable hole.
        """
        if (
            not _valid_plan_decision_key(decision_key)
            or type(modified) is not bool
            or not _valid_checkpoint_id(checkpoint_id)
        ):
            return Failure(
                TranscriptSchemaError(
                    message="invalid plan decision evidence",
                    source="chat_transcript_repository",
                    operation="record_plan_decision_fact",
                    expected_version=TRANSCRIPT_SCHEMA_VERSION,
                    found_version=None,
                )
            )
        try:
            async with self._session_factory() as session, session.begin():
                session_row = await _load_session_for_update(session, session_id, user_id)
                if session_row is None:
                    return Failure(
                        TranscriptNotFoundError(
                            message="transcript session not found",
                            session_id=session_id,
                        )
                    )
                schema_failure = _schema_gate(
                    session_row.schema_version,
                    operation="record_plan_decision_fact",
                )
                if schema_failure is not None:
                    return Failure(schema_failure)
                latest_prompt = (
                    await session.execute(
                        select(ChatMessage.source_key, ChatMessage.source_checkpoint_id)
                        .where(
                            ChatMessage.session_id == session_id,
                            ChatMessage.user_id == session_row.user_id,
                            ChatMessage.kind == "plan_prompt",
                        )
                        .order_by(
                            ChatMessage.turn_seq.desc(),
                            ChatMessage.ordinal.desc(),
                            ChatMessage.id.desc(),
                        )
                        .limit(1)
                    )
                ).one_or_none()
                if (
                    latest_prompt is None
                    or latest_prompt[0] != decision_key
                    or latest_prompt[1] != checkpoint_id
                    or session_row.last_checkpoint_id != checkpoint_id
                ):
                    return Failure(
                        TranscriptNotFoundError(
                            message="current plan prompt evidence is stale",
                            session_id=session_id,
                        )
                    )
                decision_source_key = f"decision:{decision_key.removeprefix('interrupt:')}"
                existing_decision = (
                    await session.execute(
                        select(ChatMessage.id).where(
                            ChatMessage.session_id == session_id,
                            ChatMessage.source_key == decision_source_key,
                        )
                    )
                ).scalar_one_or_none()
                if existing_decision is not None:
                    return Success(True)
                if session_row.pending_plan_decision_key is not None:
                    if (
                        session_row.pending_plan_decision_key == decision_key
                        and session_row.pending_plan_decision_modified == modified
                        and session_row.pending_plan_decision_checkpoint_id == checkpoint_id
                    ):
                        return Success(True)
                    return Failure(
                        TranscriptNotFoundError(
                            message="plan decision evidence is already claimed",
                            session_id=session_id,
                        )
                    )
                session_row.pending_plan_decision_key = decision_key
                session_row.pending_plan_decision_modified = modified
                session_row.pending_plan_decision_checkpoint_id = checkpoint_id
                return Success(True)
        except SQLAlchemyError as exc:
            return _sanitize_db_error(exc, "record_plan_decision_fact")

    @trace_layer("repository")
    async def record_review_decision_fact(
        self,
        *,
        session_id: str,
        user_id: str,
        action: str,
        reviewer_id: str,
        reviewer_role: str,
        self_reviewed: bool,
        checkpoint_id: str | None = None,
    ) -> TranscriptResult[bool]:
        """Append one idempotent, content-free review decision audit fact."""
        if (
            action not in {"approve", "reject", "modify"}
            or not _REVIEWER_ID_RE.fullmatch(reviewer_id)
            or not reviewer_role
            or type(self_reviewed) is not bool
        ):
            return Failure(
                TranscriptSchemaError(
                    message="invalid review decision evidence",
                    source="chat_transcript_repository",
                    operation="record_review_decision_fact",
                    expected_version=TRANSCRIPT_SCHEMA_VERSION,
                    found_version=None,
                )
            )
        try:
            async with self._session_factory() as session, session.begin():
                session_row = await _load_session_for_update(session, session_id, user_id)
                if session_row is None:
                    return Failure(
                        TranscriptNotFoundError(
                            message="transcript session not found",
                            session_id=session_id,
                        )
                    )
                schema_failure = _schema_gate(
                    session_row.schema_version,
                    operation="record_review_decision_fact",
                )
                if schema_failure is not None:
                    return Failure(schema_failure)
                turn_seq = session_row.turn_count or 1
                max_ordinal = (
                    await session.execute(
                        select(func.max(ChatMessage.ordinal)).where(
                            ChatMessage.session_id == session_id,
                            ChatMessage.turn_seq == turn_seq,
                        )
                    )
                ).scalar_one()
                source_key = f"review:{turn_seq}:{action}:{reviewer_id}"
                result = await session.execute(
                    pg_insert(ChatMessage)
                    .values(
                        session_id=session_id,
                        user_id=user_id,
                        role="user",
                        content=action,
                        turn_seq=turn_seq,
                        ordinal=int(max_ordinal or 0) + 1,
                        kind="review_decision",
                        source_key=source_key,
                        source_checkpoint_id=checkpoint_id or "review",
                        extra_metadata={
                            "reviewer_id": reviewer_id,
                            "reviewer_role": reviewer_role,
                            "self_reviewed": self_reviewed,
                        },
                    )
                    .on_conflict_do_nothing(
                        index_elements=[ChatMessage.session_id, ChatMessage.source_key]
                    )
                    .returning(ChatMessage.id)
                )
                return Success(result.scalar_one_or_none() is not None)
        except SQLAlchemyError as exc:
            return _sanitize_db_error(exc, "record_review_decision_fact")

    @trace_layer("repository")
    async def apply_projection(  # noqa: PLR0914 — transaction boundary keeps all lineage locals together
        self,
        projection: Projection,
        *,
        user_id: str,
        session_id: str,
        usage: Mapping[str, int] | None = None,
        usage_metadata: Mapping[str, Mapping[str, Any]] | None = None,
        prior_checkpoint_id: str | None,
    ) -> TranscriptResult[ApplyOutcome]:
        """One boundary = one transaction: watermark + rows + status (§5.1).

        ``usage`` maps ``source_key -> tokens``; on conflict only the one-time
        fill ``tokens_used = COALESCE(chat_messages.tokens_used, EXCLUDED.tokens_used)``
        happens, so replay can never double-bill (§6).
        ``prior_checkpoint_id`` is REQUIRED and is the transcript watermark
        the caller read BEFORE projecting (``None`` when it observed no row).
        When the locked row's watermark no longer equals it, a newer writer
        landed between that read and this apply — the whole apply answers a
        no-op so its status/watermark can never be overwritten backward (row
        locking only orders writes, it doesn't reject them). Required, not
        defaulted: an omitted argument must not silently disable the guard.
        """
        if projection.is_no_op:
            noop = projection.no_op
            return Success(
                ApplyOutcome(
                    applied=False,
                    reason=f"noop:{noop.reason if noop is not None else 'unspecified'}",
                )
            )
        if projection.session_status is None:
            return Success(ApplyOutcome(applied=False, reason="no_derived_status"))
        # §9: projector bug / unknown item shape → Open. Log with ids only,
        # alert through the failure result, never an exception.
        bad_kind = next(
            (row.kind for row in projection.rows if row.kind not in MESSAGE_KINDS), None
        )
        if bad_kind is not None:
            logger.bind(
                session_id=session_id,
                checkpoint_id=projection.watermark_checkpoint_id,
            ).error("transcript_unknown_row_kind")
            return Failure(
                TranscriptInfrastructureError(
                    message="projection produced an unknown row kind",
                    source="chat_transcript_repository",
                    operation="apply_projection",
                )
            )

        tokens_for = dict(usage or {})
        metadata_for = {key: dict(value) for key, value in (usage_metadata or {}).items() if value}
        session_ctx = self._session_factory()
        try:
            async with session_ctx as session, session.begin():
                # Session upsert — first time only; the row then carries state.
                await session.execute(
                    pg_insert(ChatSession)
                    .values(
                        session_id=session_id[:_SESSION_ID_LEN],
                        user_id=user_id[:_USER_ID_LEN],
                        schema_version=TRANSCRIPT_SCHEMA_VERSION,
                        status=projection.session_status.value,
                    )
                    .on_conflict_do_nothing(index_elements=[ChatSession.session_id])
                )
                session_row = (
                    await session.execute(
                        select(ChatSession)
                        .where(ChatSession.session_id == session_id)
                        .with_for_update()
                    )
                ).scalar_one()

                # §9 ownership: the locked row must BELONG to the caller. A
                # mismatched user_id never writes rows or moves the watermark —
                # it answers not-found, sanitized, like any other wrong-owner read.
                if session_row.user_id != user_id[:_USER_ID_LEN]:
                    logger.bind(session_id=session_id).warning("transcript_owner_mismatch_refused")
                    return Failure(
                        TranscriptNotFoundError(
                            message="transcript session not found",
                            session_id=session_id,
                        )
                    )
                schema_failure = _schema_gate(
                    session_row.schema_version,
                    operation="apply_projection",
                )
                if schema_failure is not None:
                    return Failure(schema_failure)

                # Stale-projection guard (§5.3): the locked row's watermark
                # must still be the one this caller OBSERVED before
                # projecting — a mismatch means a newer writer landed between
                # the read and this apply. None==None means the row was
                # freshly created and nobody else claimed it; a watermark
                # appearing under a None observation is exactly the theft
                # this guard rejects.
                if session_row.last_checkpoint_id != prior_checkpoint_id:
                    logger.bind(
                        session_id=session_id,
                        observed_prior=prior_checkpoint_id,
                        current=session_row.last_checkpoint_id,
                    ).debug("transcript_apply_stale_projection")
                    return Success(ApplyOutcome(applied=False, reason="stale_projection"))
                # Batch-level no-op (§4.3): the common replay is free. Rows are
                # still allowed through (item level) when this projection has
                # rows, so crash-healing replays of the same checkpoint heal.
                if (
                    projection.watermark_checkpoint_id is not None
                    and session_row.last_checkpoint_id == projection.watermark_checkpoint_id
                    and not projection.rows
                    and not (tokens_for or metadata_for)
                ):
                    return Success(ApplyOutcome(applied=False, reason="same_checkpoint"))
                if (
                    projection.watermark_checkpoint_id is not None
                    and session_row.last_checkpoint_id == projection.watermark_checkpoint_id
                    and not projection.rows
                ):
                    filled = await self._fill_usage_only(
                        session,
                        session_id=session_id,
                        tokens_for=tokens_for,
                        metadata_for=metadata_for,
                        projected_keys=set(),
                    )
                    return Success(
                        ApplyOutcome(
                            applied=filled > 0,
                            reason="usage_filled" if filled > 0 else "same_checkpoint",
                        )
                    )

                turn_seq, turn_opened = await _resolve_turn(
                    session, session_id, projection.query_source_key, session_row
                )

                inserted = 0
                projected_keys: set[str] = set()
                # A stale projection may contain a decision row derived from
                # a previous checkpoint generation.  Do not let it become a
                # durable fact (or clear the newer proof) merely because the
                # source key happens to match.
                rows_to_apply = tuple(
                    row
                    for row in projection.rows
                    if _plan_decision_row_is_authorized(
                        row,
                        session_row=session_row,
                        checkpoint_id=projection.watermark_checkpoint_id,
                    )
                )
                for row in rows_to_apply:
                    projected_keys.add(row.source_key)
                    attached_metadata = metadata_for.get(row.source_key)
                    row_extra = dict(row.extra_metadata or {})
                    if attached_metadata is not None:
                        # §7: content-free usage/tool facts ride the segment's
                        # assistant row — merged at insert so they never get
                        # lost between the collector reset and a later pass.
                        row_extra.update(attached_metadata)
                    inserted += await self._insert_row(
                        session,
                        row,
                        user_id=user_id,
                        session_id=session_id,
                        turn_seq=turn_seq,
                        checkpoint_id=row_checkpoint(projection),
                        tokens=tokens_for.get(row.source_key),
                        extra_metadata=row_extra or None,
                    )

                # §7 usage-only fill: segment totals whose target row already
                # exists (the projector dedupes it away) still land once, via
                # the same COALESCE fill-once rule. Existing NULLs fill;
                # anything already billed never changes. The metadata breakdown
                # merge follows the same fill-once gate (usage not yet present).
                await self._fill_usage_only(
                    session,
                    session_id=session_id,
                    tokens_for=tokens_for,
                    metadata_for=metadata_for,
                    projected_keys=projected_keys,
                )

                set_values = _session_set_values(
                    session_row,
                    projection.model_copy(update={"rows": rows_to_apply}),
                    projection.watermark_checkpoint_id,
                    turn_opened,
                    status=projection.session_status.value,
                )
                await session.execute(
                    update(ChatSession).where(ChatSession.id == session_row.id).values(**set_values)
                )
                return Success(
                    ApplyOutcome(
                        applied=True,
                        rows_inserted=inserted,
                        turn_seq=turn_seq,
                        turn_opened=turn_opened,
                    )
                )
        except SQLAlchemyError as exc:
            add_database_error_note(exc, table="chat_messages")
            await session_ctx.rollback()
            logger.bind(
                session_id=session_id,
                checkpoint_id=projection.watermark_checkpoint_id,
                operation="apply_projection",
            ).warning("transcript_apply_projection_failed")
            return Failure(
                TranscriptInfrastructureError(
                    message="transcript projection could not be applied",
                    source="chat_transcript_repository",
                    operation="apply_projection",
                )
            )

    @staticmethod
    async def _fill_usage_only(
        session: AsyncSession,
        *,
        session_id: str,
        tokens_for: Mapping[str, Any],
        metadata_for: Mapping[str, Mapping[str, Any]],
        projected_keys: set[str],
    ) -> int:
        """§7 usage-only fill — same-transaction, fill-once (target deduped).

        The projector drops rows already keyed in the saved turn context, so
        the segment's tokens land on the EXISTING assistant row instead.
        Read-modify-write UNDER the session's row lock (the lock at the top
        of this transaction serializes every writer for this session):
        ``tokens_used`` fills only NULLs (replay cannot double-bill), while
        metadata fills only absent top-level facts. The target set is the
        union of token and metadata keys, so a zero-token tool segment is not
        lost. This read-modify-write also avoids json/jsonb operator mismatch
        (``extra_metadata`` is a ``json`` column and ``||`` is jsonb-only).
        """
        target_keys = (set(tokens_for) | set(metadata_for)) - projected_keys
        filled = 0
        for source_key in target_keys:
            tokens = tokens_for.get(source_key)
            attached = metadata_for.get(source_key)
            if tokens is None and attached is None:
                continue
            target_row = (
                await session.execute(
                    select(ChatMessage)
                    .where(
                        ChatMessage.session_id == session_id,
                        ChatMessage.source_key == source_key,
                        ChatMessage.role == "assistant",
                    )
                    .with_for_update()
                )
            ).scalar_one_or_none()
            if target_row is None:
                continue  # missing target or wrong row kind — fill-once holds
            if tokens is not None and target_row.tokens_used is None:
                target_row.tokens_used = tokens
                filled += 1
            if attached is not None:
                existing_metadata = dict(target_row.extra_metadata or {})
                merged_metadata = {
                    **{
                        key: value
                        for key, value in dict(attached).items()
                        if key not in existing_metadata
                    },
                    **existing_metadata,
                }
                if merged_metadata != existing_metadata:
                    target_row.extra_metadata = merged_metadata
                    filled += 1
        return filled

    @staticmethod
    async def _insert_row(
        session: AsyncSession,
        row: ProjectedRow,
        *,
        user_id: str,
        session_id: str,
        turn_seq: int,
        checkpoint_id: str,
        tokens: int | None,
        extra_metadata: Mapping[str, Any] | None = None,
    ) -> int:
        tokens_used = tokens if tokens is not None else row.tokens_used
        merged_metadata = dict(
            extra_metadata if extra_metadata is not None else row.extra_metadata or {}
        )
        base = pg_insert(ChatMessage).values(
            session_id=session_id[:_SESSION_ID_LEN],
            user_id=user_id[:_USER_ID_LEN],
            role=row.role.value,
            content=row.content,
            model=row.model,
            tokens_used=tokens_used,
            extra_metadata=merged_metadata or None,
            turn_seq=turn_seq,
            ordinal=row.ordinal,
            kind=row.kind,
            source_key=row.source_key,
            source_checkpoint_id=checkpoint_id,
        )
        stmt = base.on_conflict_do_update(
            index_elements=[ChatMessage.session_id, ChatMessage.source_key],
            # Fill-once: never overwrite; replay cannot double-bill (§6).
            set_={
                "tokens_used": func.coalesce(ChatMessage.tokens_used, base.excluded.tokens_used),
            },
        ).returning(
            # The xmax trick: a freshly inserted row has xmax 0, a
            # conflict-updated row carries the other writer's transaction id —
            # so this counts REAL insertions, not conflict-updates, and
            # ApplyOutcome reports the rows that actually landed (§13 healing).
            literal_column("(xmax = 0)").label("did_insert"),
        )
        result = await session.execute(stmt)
        return sum(1 for did_insert in result.scalars().all() if did_insert)

    @trace_layer("repository")
    async def insert_rows_only(
        self,
        projection: Projection,
        *,
        user_id: str,
        session_id: str,
    ) -> TranscriptResult[ApplyOutcome]:
        """Message-row-only application — the §6 hole-fill vehicle (recovery).

        Inserts any not-yet-persisted keyed rows and nothing else: no
        watermark, no cached status, no verdict rows. A historical snapshot's
        derived *outcomes* (turn_closed / final_summary) must never ride a
        replay judged at today's time; only their row facts recover.
        """
        session_ctx = self._session_factory()
        try:
            async with session_ctx as session, session.begin():
                session_row = (
                    await session.execute(
                        select(ChatSession)
                        .where(
                            ChatSession.session_id == session_id,
                            ChatSession.user_id == user_id[:_USER_ID_LEN],
                        )
                        .with_for_update()
                    )
                ).scalar_one_or_none()
                if session_row is None:
                    return Failure(
                        TranscriptNotFoundError(
                            message="transcript session not found",
                            session_id=session_id,
                        )
                    )
                schema_failure = _schema_gate(
                    session_row.schema_version,
                    operation="insert_rows_only",
                )
                if schema_failure is not None:
                    return Failure(schema_failure)
                turn_seq, turn_opened = await _resolve_turn(
                    session, session_id, projection.query_source_key, session_row
                )
                recoverable = tuple(
                    row
                    for row in projection.rows
                    if row.kind not in VERDICT_ROW_KINDS
                    and _plan_decision_row_is_authorized(
                        row,
                        session_row=session_row,
                        checkpoint_id=projection.watermark_checkpoint_id,
                    )
                )
                inserted = 0
                for row in recoverable:
                    inserted += await self._insert_row(
                        session,
                        row,
                        user_id=user_id,
                        session_id=session_id,
                        turn_seq=turn_seq,
                        checkpoint_id=row_checkpoint(projection),
                        tokens=None,
                    )
                if any(
                    _plan_decision_row_is_authorized(
                        row,
                        session_row=session_row,
                        checkpoint_id=projection.watermark_checkpoint_id,
                    )
                    and _plan_decision_prompt_key(row.source_key)
                    == session_row.pending_plan_decision_key
                    for row in recoverable
                ):
                    # Hole-fill can be the first writer of the decision row
                    # after a crash. Clear the proof in this same transaction,
                    # even when the keyed insert was an idempotent conflict.
                    await session.execute(
                        update(ChatSession)
                        .where(ChatSession.id == session_row.id)
                        .values(
                            pending_plan_decision_key=None,
                            pending_plan_decision_modified=None,
                            pending_plan_decision_checkpoint_id=None,
                        )
                    )
                if turn_opened and inserted > 0:
                    # §6 recovery reservation: a recovered query row OPENS the
                    # turn it belongs to — persist turn_count (and nothing
                    # else) so the next apply_projection cannot hand the same
                    # slot to a later turn. Status and checkpoint timestamps
                    # stay untouched; recovery never rewrites consensus.
                    await session.execute(
                        update(ChatSession)
                        .where(ChatSession.id == session_row.id)
                        .values(turn_count=session_row.turn_count + 1, updated_at=func.now())
                    )
                return Success(
                    ApplyOutcome(
                        applied=inserted > 0,
                        rows_inserted=inserted,
                        turn_seq=turn_seq,
                        turn_opened=turn_opened and inserted > 0,
                    )
                )
        except SQLAlchemyError as exc:
            add_database_error_note(exc, table="chat_messages")
            await session_ctx.rollback()
            logger.bind(session_id=session_id, operation="insert_rows_only").warning(
                "transcript_insert_rows_only_failed"
            )
            return Failure(
                TranscriptInfrastructureError(
                    message="hole-fill recovery could not be applied",
                    source="chat_transcript_repository",
                    operation="insert_rows_only",
                )
            )

    # ------------------------------------------------------------------
    # Sweep readers — trusted actor, still fail-closed (§6 trigger 3 / §8)
    # ------------------------------------------------------------------

    @trace_layer("repository")
    async def get_sweep_context(
        self,
        *,
        session_id: str,
    ) -> TranscriptResult[tuple[str, TranscriptTurnContext]]:
        """Owner + turn context for the trusted beat sweep (no user gate).

        The sweep is a platform actor, not a client view: it needs whose turn
        it is so the write (apply_projection) can re-run the owner gate. All
        other §9 guards — schema gate, unknown-kind, watermark sanity — stay
        exactly as they are for connection-scoped reads.
        """
        try:
            async with self._session_factory() as session:
                session_row = (
                    await session.execute(
                        select(ChatSession)
                        .where(ChatSession.session_id == session_id)
                        .with_for_update()
                    )
                ).scalar_one_or_none()
            if session_row is None:
                return Failure(
                    TranscriptNotFoundError(
                        message="transcript session not found",
                        session_id=session_id,
                    )
                )
        except SQLAlchemyError as exc:
            return _sanitize_db_error(exc, "get_sweep_context")
        context_result = await self.get_turn_context(
            session_id=session_id, user_id=session_row.user_id
        )
        if isinstance(context_result, Failure):
            return context_result
        return Success((session_row.user_id, context_result.unwrap()))

    async def peek_session_checkpoint(
        self,
        *,
        session_id: str,
    ) -> TranscriptResult[tuple[str | None, datetime | None]]:
        """The session row's CURRENT checkpoint identity — the GC fence (§8).

        Returns ``(last_checkpoint_id, last_checkpoint_ts)`` as read NOW: a
        sweep compares both under its thread lock — id mismatch means the
        thread moved and the verdict is stale; a ts at-or-after the retention
        deadline means a new turn arrived.
        """
        try:
            async with self._session_factory() as session:
                row = (
                    await session.execute(
                        select(
                            ChatSession.last_checkpoint_id,
                            ChatSession.last_checkpoint_ts,
                        ).where(ChatSession.session_id == session_id)
                    )
                ).one_or_none()
        except SQLAlchemyError as exc:
            return _sanitize_db_error(exc, "peek_session_checkpoint")
        if row is None:
            return Success((None, None))
        return Success((row[0], row[1]))

    @trace_layer("repository")
    async def list_stale_sessions(
        self,
        *,
        older_than: datetime,
        terminal_statuses: frozenset[str],
        limit: int = 20,
    ) -> TranscriptResult[Sequence[SessionMeta]]:
        """Non-terminal sessions idle past their orphan-grace boundary (§5.3).

        The bounded sweep has two lanes because a missing persisted query is
        ambiguous: fixture debris and a real checkpoint whose first transcript
        write failed have the same SQL shape.  For limits of at least two, one
        slot is reserved for the freshest ambiguous candidate and the rest for
        oldest-first query-backed work; unused quota is filled from either
        lane.  A one-row sweep is query-first and falls back to the freshest
        ambiguous candidate.  This is deterministic priority, not strict
        fairness among immortal ambiguous rows (which needs durable attempt
        state).
        """
        if limit <= 0:
            return Success(())
        has_persisted_query = (
            select(ChatMessage.id)
            .where(
                ChatMessage.session_id == ChatSession.session_id,
                ChatMessage.kind == "query",
            )
            .exists()
        )
        try:
            async with self._session_factory() as session:
                common_filters = (
                    ChatSession.status.not_in(terminal_statuses),
                    ChatSession.last_checkpoint_ts.is_not(None),
                    ChatSession.last_checkpoint_ts <= older_than,
                )
                query_backed = (
                    (
                        await session.execute(
                            select(ChatSession)
                            .where(*common_filters, has_persisted_query)
                            .order_by(
                                ChatSession.last_checkpoint_ts,
                                ChatSession.session_id,
                            )
                            .limit(limit)
                        )
                    )
                    .scalars()
                    .all()
                )
                ambiguous = (
                    (
                        await session.execute(
                            select(ChatSession)
                            .where(*common_filters, ~has_persisted_query)
                            # Fresh first-write failures jump ahead of old,
                            # immortal integration debris in this lane.
                            .order_by(
                                ChatSession.last_checkpoint_ts.desc(),
                                ChatSession.session_id,
                            )
                            .limit(limit)
                        )
                    )
                    .scalars()
                    .all()
                )
                scheduled = _schedule_stale_session_rows(
                    query_backed=query_backed,
                    ambiguous=ambiguous,
                    limit=limit,
                )
                return Success(tuple(_session_meta_of(row) for row in scheduled))
        except SQLAlchemyError as exc:
            return _sanitize_db_error(exc, "list_stale_sessions")

    @trace_layer("repository")
    async def list_gc_candidates(
        self,
        *,
        older_than: datetime,
        limit: int = 20,
    ) -> TranscriptResult[Sequence[SessionMeta]]:
        """Terminal sessions past the retention horizon — GC input (§8).

        Exclusion is stamp-generation aware: a CONFIRMED ``gc_swept_at`` at
        or after ``last_checkpoint_ts`` means that checkpoint generation was
        already audited and survives. A session with no stamp, or with a
        confirmed stamp OLDER than its (later re-created) checkpoint, is a
        fresh candidate — the oldest audited batch cannot clog the list and
        starve newer deletions. A PENDING stamp (pre-delete crash window)
        is excluded only while FRESH — another sweep may legitimately be
        inside its critical section — and re-enters the list once stale so
        the idempotent delete + confirm retry can complete (§8).
        """
        terminal = frozenset(
            (
                str(SessionStatus.COMPLETED.value),
                str(SessionStatus.FAILED.value),
                str(SessionStatus.ABANDONED_EXPIRED.value),
            )
        )
        try:
            async with self._session_factory() as session:
                # Stamp-aware include predicate, expression form of the former
                # text() block: include a terminal session when (a) no sweep
                # stamp exists yet, (b) the stamp is CONFIRMED (pending =
                # 'false') and OLDER than the current checkpoint generation, or
                # (c) the stamp is PENDING but stale enough that another
                # sweep's critical section is over (§8). The timestamptz casts
                # and make_interval(mins => n) mirror the original verbatim.
                # `return_type=Text` is load-bearing: a bare op("->>") inherits
                # the JSON left operand's type, so the 'false'/'true'
                # comparisons would bind through the JSON serializer
                # ('"false"') and asyncpg would render a $n::JSON cast —
                # `text = json`, an operator error or a never-matching
                # comparison. Text typing keeps the literals plain text binds.
                stamp_text = ChatSession.extra_metadata.op("->>", return_type=Text)(
                    bindparam("stamp_key", value=_GC_STAMP_KEY, type_=String)
                )
                pending_text = ChatSession.extra_metadata.op("->>", return_type=Text)(
                    bindparam("pending_key", value=_GC_PENDING_KEY, type_=String)
                )
                stamp_ts = cast(stamp_text, TIMESTAMP(timezone=True))
                include_predicate = or_(
                    stamp_text.is_(None),
                    and_(pending_text == "false", stamp_ts < ChatSession.last_checkpoint_ts),
                    and_(
                        pending_text == "true",
                        stamp_ts
                        < func.now()
                        # make_interval's named argument has no func() kwarg
                        # form; the mins value is a module-owned int constant.
                        - func.make_interval(literal_column(f"mins => {_GC_PENDING_RETRY_MINS}")),
                    ),
                )
                rows = (
                    (
                        await session.execute(
                            select(ChatSession)
                            .where(
                                ChatSession.status.in_(terminal),
                                ChatSession.last_checkpoint_ts.is_not(None),
                                ChatSession.last_checkpoint_ts <= older_than,
                                include_predicate,
                            )
                            .order_by(ChatSession.last_checkpoint_ts)
                            .limit(limit)
                        )
                    )
                    .scalars()
                    .all()
                )
                return Success(tuple(_session_meta_of(row) for row in rows))
        except SQLAlchemyError as exc:
            return _sanitize_db_error(exc, "list_gc_candidates")

    @trace_layer("repository")
    async def stamp_gc_audit(
        self,
        *,
        session_id: str,
        deleted_checkpoint_id: str | None,
        swept_at: datetime,
        pending: bool = True,
    ) -> TranscriptResult[bool]:
        """§8 GC audit: the only proof that a catch-up verify preceded GC."""
        try:
            async with self._session_factory() as session, session.begin():
                session_row = (
                    await session.execute(
                        select(ChatSession)
                        .where(ChatSession.session_id == session_id)
                        .with_for_update()
                    )
                ).scalar_one_or_none()
                if session_row is None:
                    return Failure(
                        TranscriptNotFoundError(
                            message="transcript session not found",
                            session_id=session_id,
                        )
                    )
                audit: dict[str, Any] = {
                    _GC_STAMP_KEY: swept_at.isoformat(),
                    _GC_PENDING_KEY: pending,
                    "gc_deleted_checkpoint_id": deleted_checkpoint_id,
                }
                existing = dict(session_row.extra_metadata or {})
                await session.execute(
                    update(ChatSession)
                    .where(ChatSession.id == session_row.id)
                    .values(extra_metadata={**existing, **audit})
                )
                return Success(True)
        except SQLAlchemyError as exc:
            return _sanitize_db_error(exc, "stamp_gc_audit")

    async def confirm_gc_audit(
        self,
        *,
        session_id: str,
        confirmed_at: datetime,
    ) -> TranscriptResult[bool]:
        """Close the pending deletion record — deletion succeeded (§8).

        The pre-delete stamp marked the generation PENDING so a worker that
        died before ``adelete_thread`` would age back into the candidate
        list; a confirmed delete simply clears the flag (stamps stay for the
        audit trail). A confirm that never lands leaves pending-retry self-
        healing in place.
        """
        try:
            async with self._session_factory() as session, session.begin():
                session_row = (
                    await session.execute(
                        select(ChatSession)
                        .where(ChatSession.session_id == session_id)
                        .with_for_update()
                    )
                ).scalar_one_or_none()
                if session_row is None:
                    return Failure(
                        TranscriptNotFoundError(
                            message="transcript session not found",
                            session_id=session_id,
                        )
                    )
                existing = dict(session_row.extra_metadata or {})
                updated = {
                    **existing,
                    _GC_PENDING_KEY: False,
                    _GC_STAMP_KEY: confirmed_at.isoformat(),
                }
                await session.execute(
                    update(ChatSession)
                    .where(ChatSession.id == session_row.id)
                    .values(extra_metadata=updated)
                )
                return Success(True)
        except SQLAlchemyError as exc:
            return _sanitize_db_error(exc, "confirm_gc_audit")

    # ------------------------------------------------------------------
    # Turn context for the projector
    # ------------------------------------------------------------------

    @trace_layer("repository")
    async def get_turn_context(
        self,
        *,
        session_id: str,
        user_id: str,
    ) -> TranscriptResult[TranscriptTurnContext]:
        """Rows + keys + session cache for the CURRENT turn (§4.1 input)."""
        try:
            async with self._session_factory() as session:
                session_row = (
                    await session.execute(
                        select(ChatSession).where(
                            ChatSession.session_id == session_id,
                            ChatSession.user_id == user_id,
                        )
                    )
                ).scalar_one_or_none()
                if session_row is None:
                    return Success(TranscriptTurnContext())
                # §9 schema gate before ANY read the projector acts on: a
                # mixed-version/rollback deployment must modify nothing it
                # refuses to export, and the turn context is the projector's
                # write-planning input.
                gate = _schema_gate(
                    session_row.schema_version,
                    operation="get_turn_context",
                )
                if gate is not None:
                    return Failure(gate)
                latest_turn = (
                    await session.execute(
                        select(func.max(ChatMessage.turn_seq)).where(
                            ChatMessage.session_id == session_id
                        )
                    )
                ).scalar_one()
                if latest_turn is None:
                    keys: set[str] = set()
                    rows: tuple[TurnRow, ...] = ()
                else:
                    row_dicts = (
                        await session.execute(
                            select(
                                ChatMessage.source_key,
                                ChatMessage.kind,
                                ChatMessage.ordinal,
                            )
                            .where(
                                ChatMessage.session_id == session_id,
                                ChatMessage.turn_seq == latest_turn,
                            )
                            .order_by(ChatMessage.ordinal, ChatMessage.id)
                        )
                    ).all()
                    keys = {entry.source_key for entry in row_dicts}
                    rows = tuple(
                        TurnRow(
                            kind=entry.kind,
                            source_key=entry.source_key,
                            ordinal=entry.ordinal,
                        )
                        for entry in row_dicts
                    )
                return Success(
                    TranscriptTurnContext(
                        existing_keys=frozenset(keys),
                        turn_rows=TurnRows(
                            rows=rows,
                            pending_plan_decision_key=session_row.pending_plan_decision_key,
                            pending_plan_decision_modified=session_row.pending_plan_decision_modified,
                            pending_plan_decision_checkpoint_id=session_row.pending_plan_decision_checkpoint_id,
                        ),
                        session_status=SessionStatus(session_row.status),
                        turn_count=session_row.turn_count,
                        last_checkpoint_id=session_row.last_checkpoint_id,
                        last_checkpoint_ts=session_row.last_checkpoint_ts,
                    )
                )
        except SQLAlchemyError as exc:
            return _sanitize_db_error(exc, "get_turn_context")

    # ------------------------------------------------------------------
    # Readers — schema-version gated (§9 fail closed)
    # ------------------------------------------------------------------

    @trace_layer("repository")
    async def get_session_meta(
        self,
        *,
        session_id: str,
        user_id: str,
    ) -> TranscriptResult[SessionMeta]:
        try:
            async with self._session_factory() as session:
                session_row = await _load_session(session, session_id, user_id)
                if session_row is None:
                    return Failure(
                        TranscriptNotFoundError(
                            message="transcript session not found",
                            session_id=session_id,
                        )
                    )
                version_failure = _schema_gate(
                    session_row.schema_version,
                    operation="get_session_meta",
                )
                if version_failure is not None:
                    return Failure(version_failure)
                return Success(_session_meta_of(session_row))
        except SQLAlchemyError as exc:
            return _sanitize_db_error(exc, "get_session_meta")

    @trace_layer("repository")
    async def list_sessions(
        self,
        *,
        user_id: str,
        limit: int = 50,
        cursor_updated_at: datetime | None = None,
    ) -> TranscriptResult[Sequence[SessionMeta]]:
        """Chat list ordered by last activity (D3 index). Schema gated (§9)."""
        try:
            async with self._session_factory() as session:
                statement = (
                    select(ChatSession)
                    .where(ChatSession.user_id == user_id)
                    .order_by(ChatSession.updated_at.desc(), ChatSession.id.desc())
                    .limit(limit)
                )
                if cursor_updated_at is not None:
                    statement = statement.where(ChatSession.updated_at < cursor_updated_at)
                rows = (await session.execute(statement)).scalars().all()
                for session_row in rows:
                    version_failure = _schema_gate(
                        session_row.schema_version,
                        operation="list_sessions",
                    )
                    if version_failure is not None:
                        return Failure(version_failure)
                return Success(tuple(_session_meta_of(row) for row in rows))
        except SQLAlchemyError as exc:
            return _sanitize_db_error(exc, "list_sessions")

    @trace_layer("repository")
    async def read_transcript(
        self,
        *,
        session_id: str,
        user_id: str,
        limit: int = 500,
        after_turn_seq: int | None = None,
        after_ordinal: int | None = None,
        after_id: int | None = None,
    ) -> TranscriptResult[Sequence[TranscriptMessage]]:
        """Keyset pagination: display order (turn_seq, ordinal), stable reads.

        The cursor is the (turn_seq, ordinal, id) triple the caller last saw
        — the FULL ordering key, so rows sharing turn_seq and ordinal (a
        legacy import can produce them; the id tiebreaker is in the ORDER
        BY) are never skipped at a page boundary. OFFSET rereads the whole
        table on every page and can repeat rows when exports and appends
        overlap; keyset paging never revisits rows (D3).
        """
        try:
            async with self._session_factory() as session:
                session_row = await _load_session(session, session_id, user_id)
                if session_row is None:
                    return Failure(
                        TranscriptNotFoundError(
                            message="transcript session not found",
                            session_id=session_id,
                        )
                    )
                version_failure = _schema_gate(
                    session_row.schema_version,
                    operation="read_transcript",
                )
                if version_failure is not None:
                    return Failure(version_failure)
                key = (after_turn_seq, after_ordinal, after_id)
                statement = select(ChatMessage).where(ChatMessage.session_id == session_id)
                if all(part is not None for part in key):
                    statement = statement.where(
                        tuple_(ChatMessage.turn_seq, ChatMessage.ordinal, ChatMessage.id) > key
                    )
                rows = (
                    (
                        await session.execute(
                            statement.order_by(
                                ChatMessage.turn_seq, ChatMessage.ordinal, ChatMessage.id
                            ).limit(limit)
                        )
                    )
                    .scalars()
                    .all()
                )
                return Success(
                    tuple(
                        TranscriptMessage(
                            row_id=row.id,
                            turn_seq=row.turn_seq,
                            ordinal=row.ordinal,
                            kind=row.kind,
                            role=row.role,
                            content=row.content,
                            model=row.model,
                            tokens_used=row.tokens_used,
                            redacted=row.redacted_at is not None,
                            created_at=row.created_at,
                            extra_metadata=dict(row.extra_metadata or {}),
                        )
                        for row in rows
                    )
                )
        except SQLAlchemyError as exc:
            return _sanitize_db_error(exc, "read_transcript")

    # ------------------------------------------------------------------
    # Error sanitization — driver detail never leaves this edge (§10)
    # ------------------------------------------------------------------


def _sanitize_db_error(exc: SQLAlchemyError, operation: str) -> Failure[TranscriptError]:
    add_database_error_note(exc, table="chat_messages", operation=operation)
    exc.add_note(format_exc())
    logger.bind(
        operation=operation,
        error_notes=list(getattr(exc, "__notes__", ())),
    ).warning("transcript_repository_failed")
    return Failure(
        TranscriptInfrastructureError(
            message="transcript storage operation failed",
            source="chat_transcript_repository",
            operation=operation,
        )
    )


def row_checkpoint(projection: Projection) -> str:
    return projection.watermark_checkpoint_id or "unknown-checkpoint"


def _schema_gate(
    found: int | None,
    *,
    operation: str,
) -> TranscriptSchemaError | None:
    """Unknown transcript version ⇒ refuse at the read boundary (§9)."""
    if found == TRANSCRIPT_SCHEMA_VERSION:
        return None
    return TranscriptSchemaError(
        message="unrecognized transcript schema version",
        source="chat_transcript_repository",
        operation=operation,
        expected_version=TRANSCRIPT_SCHEMA_VERSION,
        found_version=found,
    )


def _schedule_stale_session_rows(
    *,
    query_backed: Sequence[ChatSession],
    ambiguous: Sequence[ChatSession],
    limit: int,
) -> tuple[ChatSession, ...]:
    """Merge pre-ordered sweep lanes without exceeding the caller's bound."""
    if limit <= 0:
        return ()
    if limit == 1:
        return tuple(query_backed[:1] or ambiguous[:1])

    query_quota = limit - 1
    selected = [*query_backed[:query_quota], *ambiguous[:1]]
    remaining = limit - len(selected)
    if remaining:
        query_fill = query_backed[query_quota : query_quota + remaining]
        selected.extend(query_fill)
        remaining -= len(query_fill)
    if remaining:
        selected.extend(ambiguous[1 : 1 + remaining])
    return tuple(selected)


def _session_meta_of(row: ChatSession) -> SessionMeta:
    return SessionMeta(
        session_id=row.session_id,
        user_id=row.user_id,
        title=row.title,
        status=row.status,
        schema_version=row.schema_version,
        turn_count=row.turn_count,
        last_checkpoint_id=row.last_checkpoint_id,
        last_checkpoint_ts=row.last_checkpoint_ts,
        gc_pending=(row.extra_metadata or {}).get(_GC_PENDING_KEY) is True,
        created_at=row.created_at,
        updated_at=row.updated_at,
    )
