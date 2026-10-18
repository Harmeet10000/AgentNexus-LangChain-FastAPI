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

# Runtime-required: pydantic field annotations resolve at class definition
# (SessionMeta/TranscriptMessage) — do NOT move these to TYPE_CHECKING.
from datetime import datetime  # noqa: TC003 — pydantic field annotation
from typing import TYPE_CHECKING, Any  # noqa: TC003 — pydantic field annotation

from pydantic import BaseModel, ConfigDict, Field
from returns.result import Failure, Success
from sqlalchemy import func, literal_column, select, text, tuple_, update
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.exc import SQLAlchemyError

from app.features.agent_saul.models import MESSAGE_SCHEMA_VERSION, ChatMessage, ChatSession
from app.features.agent_saul.transcript.errors import (
    TranscriptInfrastructureError,
    TranscriptNotFoundError,
    TranscriptSchemaError,
)
from app.features.agent_saul.transcript.projection import (
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

    from app.features.agent_saul.transcript.errors import (
        TranscriptError,
        TranscriptResult,
    )
    from app.features.agent_saul.transcript.projection import (
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
    """Read model for transcript reads/export (D3 order: turn_seq, ordinal)."""

    model_config = ConfigDict(frozen=True)

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
    return values


async def _load_session(session: AsyncSession, session_id: str, user_id: str) -> ChatSession | None:
    return (
        await session.execute(
            select(ChatSession).where(
                ChatSession.session_id == session_id,
                ChatSession.user_id == user_id,
            )
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
    async def apply_projection(
        self,
        projection: Projection,
        *,
        user_id: str,
        session_id: str,
        usage: Mapping[str, int] | None = None,
    ) -> TranscriptResult[ApplyOutcome]:
        """One boundary = one transaction: watermark + rows + status (§5.1).

        ``usage`` maps ``source_key -> tokens``; on conflict only the one-time
        fill ``tokens_used = COALESCE(chat_messages.tokens_used, EXCLUDED.tokens_used)``
        happens, so replay can never double-bill (§6).
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

                checkpoint_id = projection.watermark_checkpoint_id
                # Batch-level no-op (§4.3): the common replay is free. Rows are
                # still allowed through (item level) when this projection has
                # rows, so crash-healing replays of the same checkpoint heal.
                if (
                    checkpoint_id is not None
                    and session_row.last_checkpoint_id == checkpoint_id
                    and not projection.rows
                ):
                    return Success(ApplyOutcome(applied=False, reason="same_checkpoint"))

                turn_seq, turn_opened = await _resolve_turn(
                    session, session_id, projection.query_source_key, session_row
                )

                inserted = 0
                projected_keys: set[str] = set()
                for row in projection.rows:
                    projected_keys.add(row.source_key)
                    inserted += await self._insert_row(
                        session,
                        row,
                        user_id=user_id,
                        session_id=session_id,
                        turn_seq=turn_seq,
                        checkpoint_id=row_checkpoint(projection),
                        tokens=tokens_for.get(row.source_key),
                    )

                # §7 usage-only fill: segment totals whose target row already
                # exists (the projector dedupes it away) still land once, via
                # the same COALESCE fill-once rule. Existing NULLs fill;
                # anything already billed never changes.
                for source_key, tokens in tokens_for.items():
                    if source_key in projected_keys or tokens is None:
                        continue
                    await session.execute(
                        update(ChatMessage)
                        .where(
                            ChatMessage.session_id == session_id,
                            ChatMessage.source_key == source_key,
                            ChatMessage.tokens_used.is_(None),
                        )
                        .values(tokens_used=tokens)
                    )

                set_values = _session_set_values(
                    session_row,
                    projection,
                    checkpoint_id,
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
    async def _insert_row(
        session: AsyncSession,
        row: ProjectedRow,
        *,
        user_id: str,
        session_id: str,
        turn_seq: int,
        checkpoint_id: str,
        tokens: int | None,
    ) -> int:
        tokens_used = tokens if tokens is not None else row.tokens_used
        base = pg_insert(ChatMessage).values(
            session_id=session_id[:_SESSION_ID_LEN],
            user_id=user_id[:_USER_ID_LEN],
            role=row.role.value,
            content=row.content,
            model=row.model,
            tokens_used=tokens_used,
            extra_metadata=row.extra_metadata or None,
            turn_seq=turn_seq,
            ordinal=row.ordinal,
            kind=row.kind,
            source_key=row.source_key,
            source_checkpoint_id=checkpoint_id,
        )
        stmt = base.on_conflict_do_update(
            index_elements=[ChatMessage.session_id, ChatMessage.source_key],
            # Fill-once: never overwrite; replay cannot double-bill (§6).
            set_={"tokens_used": func.coalesce(ChatMessage.tokens_used, base.excluded.tokens_used)},
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
                turn_seq, turn_opened = await _resolve_turn(
                    session, session_id, projection.query_source_key, session_row
                )
                recoverable = tuple(
                    row for row in projection.rows if row.kind not in VERDICT_ROW_KINDS
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

    @trace_layer("repository")
    async def list_stale_sessions(
        self,
        *,
        older_than: datetime,
        terminal_statuses: frozenset[str],
        limit: int = 20,
    ) -> TranscriptResult[Sequence[SessionMeta]]:
        """Non-terminal sessions idle past their orphan-grace boundary (§5.3)."""
        try:
            async with self._session_factory() as session:
                rows = (
                    (
                        await session.execute(
                            select(ChatSession)
                            .where(
                                ChatSession.status.not_in(terminal_statuses),
                                ChatSession.last_checkpoint_ts.is_not(None),
                                ChatSession.last_checkpoint_ts <= older_than,
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
            return _sanitize_db_error(exc, "list_stale_sessions")

    @trace_layer("repository")
    async def list_gc_candidates(
        self,
        *,
        older_than: datetime,
        limit: int = 20,
    ) -> TranscriptResult[Sequence[SessionMeta]]:
        """Terminal sessions past the retention horizon — GC input (§8).

        A session is excluded only while its sweep stamp is up to date with
        its checkpoint: an ``gc_swept_at`` at-or-after ``last_checkpoint_ts``
        means that checkpoint generation was already audited. A session with
        no stamp, or with a stamp OLDER than its (later re-created)
        checkpoint, is a fresh candidate — the oldest audited batch cannot
        clog the list and starve newer deletions.
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
                rows = (
                    (
                        await session.execute(
                            select(ChatSession)
                            .where(
                                ChatSession.status.in_(terminal),
                                ChatSession.last_checkpoint_ts.is_not(None),
                                ChatSession.last_checkpoint_ts <= older_than,
                                # No stamp, or a stamp from a previous
                                # checkpoint generation (§8; _GC_STAMP_KEY).
                                text(
                                    "chat_sessions.extra_metadata ->> :stamp_key IS NULL "
                                    "OR (chat_sessions.extra_metadata ->> :stamp_key)"
                                    "::timestamptz < chat_sessions.last_checkpoint_ts"
                                ).params(stamp_key=_GC_STAMP_KEY),
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
                            select(ChatMessage.source_key, ChatMessage.kind, ChatMessage.ordinal)
                            .where(
                                ChatMessage.session_id == session_id,
                                ChatMessage.turn_seq == latest_turn,
                            )
                            .order_by(ChatMessage.ordinal, ChatMessage.id)
                        )
                    ).all()
                    keys = {entry.source_key for entry in row_dicts}
                    rows = tuple(
                        TurnRow(kind=entry.kind, source_key=entry.source_key, ordinal=entry.ordinal)
                        for entry in row_dicts
                    )
                return Success(
                    TranscriptTurnContext(
                        existing_keys=frozenset(keys),
                        turn_rows=TurnRows(rows=rows),
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
    ) -> TranscriptResult[SessionMeta | None]:
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
                version_failure = _schema_gate(session_row.schema_version)
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
                    version_failure = _schema_gate(session_row.schema_version)
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
    ) -> TranscriptResult[Sequence[TranscriptMessage]]:
        """Keyset pagination: display order (turn_seq, ordinal), stable reads.

        The cursor is the (turn_seq, ordinal) pair the caller last saw —
        OFFSET rereads the whole table on every page and can repeat rows
        when exports and appends overlap; keyset paging never revisits rows
        (D3). The leading ``session_id`` equality still drives the index.
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
                version_failure = _schema_gate(session_row.schema_version)
                if version_failure is not None:
                    return Failure(version_failure)
                cursor = (after_turn_seq, after_ordinal)
                statement = select(ChatMessage).where(ChatMessage.session_id == session_id)
                if cursor[0] is not None and cursor[1] is not None:
                    statement = statement.where(
                        tuple_(ChatMessage.turn_seq, ChatMessage.ordinal) > cursor
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
    add_database_error_note(exc, table="chat_messages")
    logger.bind(operation=operation).warning("transcript_repository_failed")
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
) -> TranscriptSchemaError | None:
    """Unknown transcript version ⇒ refuse at the read boundary (§9)."""
    if found == TRANSCRIPT_SCHEMA_VERSION:
        return None
    return TranscriptSchemaError(
        message="unrecognized transcript schema version",
        source="chat_transcript_repository",
        operation="read_transcript",
        expected_version=TRANSCRIPT_SCHEMA_VERSION,
        found_version=found,
    )


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
        created_at=row.created_at,
        updated_at=row.updated_at,
    )
