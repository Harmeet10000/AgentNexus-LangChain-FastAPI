"""Deep transcript export operation.

This module owns the complete "export transcript" invariant:
- Verifies ownership and schema-version compatibility (fail-closed) BEFORE streaming
- Streams transcript rows in display order (turn_seq, ordinal)
- Manages keyset pagination cursors internally
- Handles concurrent-append consistency under monotonic ordering
- Bounds memory by page size
- Surfaces mid-stream failures as typed events (never silent truncation)
- HTTP transport adapter encodes as NDJSON; CLI/worker adapters can consume typed events
"""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, ConfigDict, Field
from returns.result import Failure, Success

from app.features.agent_saul.transcript.errors import TranscriptNotFoundError

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from returns.result import Result

    from app.features.agent_saul.transcript.errors import TranscriptError
    from app.features.agent_saul.transcript.repository import ChatTranscriptRepository


class ExportTranscript(BaseModel):
    """Request to export a transcript — only caller-owned facts."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    session_id: str
    requesting_user_id: str


class TranscriptExportRow(BaseModel):
    """Single row in the exported transcript.

    Snake_case fields preserved for wire compatibility. CamelCase aliases
    available via model_dump(by_alias=True) if a consumer prefers them.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    row_id: int = Field(serialization_alias="rowId")
    turn_seq: int = Field(serialization_alias="turnSeq")
    ordinal: int
    kind: str
    role: str
    content: str | None
    model: str | None = None
    tokens_used: int | None = Field(default=None, serialization_alias="tokensUsed")
    redacted: bool
    created_at: datetime | None = Field(default=None, serialization_alias="createdAt")
    extra_metadata: dict[str, object] = Field(
        default_factory=dict, serialization_alias="extraMetadata"
    )


class TranscriptExportError(BaseModel):
    """Terminal error event emitted inside the stream (after preflight succeeded)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    type: Literal["error"] = "error"
    code: str
    message: str


# Discriminated union for the event stream after preflight
TranscriptExportEvent = TranscriptExportRow | TranscriptExportError


class PreparedTranscriptExport:
    """Preflight-successful export ready to be streamed.

    Owns the pagination logic. Each `events()` invocation owns an independent
    cursor, so a prepared export can be replayed or consumed concurrently.
    """

    _EXPORT_PAGE_ROWS = 500  # Bounded read size; pages stream until short page

    def __init__(
        self,
        repository: ChatTranscriptRepository,
        session_id: str,
        user_id: str,
    ) -> None:
        self._repository = repository
        self._session_id = session_id
        self._user_id = user_id

    async def events(self) -> AsyncIterator[TranscriptExportEvent]:
        """Stream transcript events (rows or mid-stream errors).

        Yields:
            - TranscriptExportRow for each transcript row
            - TranscriptExportError for mid-stream repository failures

        Raises:
            Never — all failures are yielded as TranscriptExportError events.
        """
        cursor_seq: int | None = None
        cursor_ord: int | None = None
        cursor_id: int | None = None

        while True:
            result = await self._repository.read_transcript(
                session_id=self._session_id,
                user_id=self._user_id,
                limit=self._EXPORT_PAGE_ROWS,
                after_turn_seq=cursor_seq,
                after_ordinal=cursor_ord,
                after_id=cursor_id,
            )
            if isinstance(result, Failure):
                error = result.failure()
                yield TranscriptExportError(
                    code=str(error.code),
                    message=error.message,
                )
                return

            page_rows = list(result.unwrap())
            if not page_rows:
                return

            for item in page_rows:
                yield TranscriptExportRow(
                    row_id=item.row_id,
                    turn_seq=item.turn_seq,
                    ordinal=item.ordinal,
                    kind=item.kind,
                    role=item.role,
                    content=item.content,
                    model=item.model,
                    tokens_used=item.tokens_used,
                    redacted=item.redacted,
                    created_at=item.created_at,
                    extra_metadata=item.extra_metadata,
                )

            # Termination detection: short page means we're done
            if len(page_rows) < self._EXPORT_PAGE_ROWS:
                return

            # Advance keyset cursor using FULL ordering key (turn_seq, ordinal, id)
            last = page_rows[-1]
            cursor_seq, cursor_ord, cursor_id = last.turn_seq, last.ordinal, last.row_id


class TranscriptExporter:
    """Deep export operation — preflight then typed event stream.

    Separates rejectable preflight (ownership, schema) from streaming body.
    HTTP adapter calls `prepare()` first, renders failures with correct status,
    then streams typed events encoded as NDJSON.
    """

    def __init__(self, repository: ChatTranscriptRepository) -> None:
        self._repository = repository

    async def prepare(
        self,
        request: ExportTranscript,
    ) -> Result[PreparedTranscriptExport, TranscriptError]:
        """Preflight: verify ownership and schema compatibility.

        Returns:
            - Success(PreparedTranscriptExport) if preflight passes
            - Failure(TranscriptNotFoundError) if session not found or wrong owner
            - Failure(TranscriptSchemaError) if schema version unsupported
            - Failure(TranscriptInfrastructureError) on DB error
        """
        meta_result = await self._repository.get_session_meta(
            session_id=request.session_id,
            user_id=request.requesting_user_id,
        )
        if isinstance(meta_result, Failure):
            return Failure(meta_result.failure())
        meta = meta_result.unwrap()
        if meta is None:
            # Defensive guard for non-conforming adapters: a successful
            # preflight without metadata must never open an HTTP stream.
            return Failure(
                TranscriptNotFoundError(
                    message="transcript session not found",
                    session_id=request.session_id,
                )
            )

        # Preflight passed — ownership and schema verified
        return Success(
            PreparedTranscriptExport(
                repository=self._repository,
                session_id=request.session_id,
                user_id=request.requesting_user_id,
            )
        )
