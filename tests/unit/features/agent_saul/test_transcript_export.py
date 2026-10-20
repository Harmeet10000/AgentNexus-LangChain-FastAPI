"""Behavioural tests for the transport-neutral transcript export module."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from returns.result import Failure, Success

from app.features.agent_saul.transcript.errors import (
    TranscriptInfrastructureError,
    TranscriptNotFoundError,
    TranscriptSchemaError,
)
from app.features.agent_saul.transcript.export import (
    ExportTranscript,
    PreparedTranscriptExport,
    TranscriptExporter,
    TranscriptExportError,
    TranscriptExportRow,
)
from app.features.agent_saul.transcript.repository import SessionMeta, TranscriptMessage

if TYPE_CHECKING:
    from typing import Any


type Cursor = tuple[int | None, int | None, int | None]

NOW = datetime(2026, 1, 1, 12, 0, tzinfo=UTC)
SESSION_ID = "session-1"
USER_ID = "user-1"


class _FakeTranscriptRepository:
    """Small repository adapter whose responses are keyed by keyset cursor."""

    def __init__(self, responses: dict[Cursor, Any], meta_result: Any | None = None) -> None:
        self.responses = responses
        self.meta_result = (
            meta_result
            if meta_result is not None
            else Success(SessionMeta(session_id=SESSION_ID, user_id=USER_ID))
        )
        self.meta_calls: list[tuple[str, str]] = []
        self.read_calls: list[tuple[int, Cursor]] = []

    async def get_session_meta(self, *, session_id: str, user_id: str) -> Any:
        self.meta_calls.append((session_id, user_id))
        return self.meta_result

    async def read_transcript(
        self,
        *,
        session_id: str,
        user_id: str,
        limit: int,
        after_turn_seq: int | None = None,
        after_ordinal: int | None = None,
        after_id: int | None = None,
    ) -> Any:
        assert (session_id, user_id) == (SESSION_ID, USER_ID)
        cursor = (after_turn_seq, after_ordinal, after_id)
        self.read_calls.append((limit, cursor))
        return self.responses.get(cursor, Success(()))


def _message(
    row_id: int,
    *,
    turn_seq: int = 1,
    ordinal: int = 0,
    content: str | None = "content",
    redacted: bool = False,
    metadata: dict[str, object] | None = None,
) -> TranscriptMessage:
    return TranscriptMessage(
        row_id=row_id,
        turn_seq=turn_seq,
        ordinal=ordinal,
        kind="assistant",
        role="assistant",
        content=content,
        model="gpt-test",
        tokens_used=42,
        redacted=redacted,
        created_at=NOW,
        extra_metadata=metadata or {},
    )


def _prepared(
    responses: dict[Cursor, Any],
    *,
    page_rows: int | None = None,
) -> tuple[PreparedTranscriptExport, _FakeTranscriptRepository]:
    repository = _FakeTranscriptRepository(responses)
    prepared = PreparedTranscriptExport(
        repository=repository,
        session_id=SESSION_ID,
        user_id=USER_ID,
    )
    # Boundary tests may opt into a small page size; ordinary tests use the
    # production page size and do not couple themselves to this implementation detail.
    if page_rows is not None:
        prepared._EXPORT_PAGE_ROWS = page_rows
    return prepared, repository


async def _events(
    prepared: PreparedTranscriptExport,
) -> list[TranscriptExportRow | TranscriptExportError]:
    return [event async for event in prepared.events()]


async def test_prepare_ownership_failure_happens_before_transcript_reads() -> None:
    repository = _FakeTranscriptRepository(
        {},
        meta_result=Failure(
            TranscriptNotFoundError(message="transcript session not found", session_id=SESSION_ID)
        ),
    )

    result = await TranscriptExporter(repository).prepare(
        ExportTranscript(session_id=SESSION_ID, requesting_user_id=USER_ID)
    )

    assert isinstance(result, Failure)
    assert isinstance(result.failure(), TranscriptNotFoundError)
    assert repository.read_calls == []


async def test_prepare_schema_failure_happens_before_transcript_reads() -> None:
    repository = _FakeTranscriptRepository(
        {},
        meta_result=Failure(
            TranscriptSchemaError(
                message="unsupported transcript schema",
                operation="get_session_meta",
                expected_version=1,
                found_version=99,
            )
        ),
    )

    result = await TranscriptExporter(repository).prepare(
        ExportTranscript(session_id=SESSION_ID, requesting_user_id=USER_ID)
    )

    assert isinstance(result, Failure)
    assert isinstance(result.failure(), TranscriptSchemaError)
    assert repository.read_calls == []


async def test_prepare_successful_none_is_rejected_before_streaming() -> None:
    repository = _FakeTranscriptRepository({}, meta_result=Success(None))

    result = await TranscriptExporter(repository).prepare(
        ExportTranscript(session_id=SESSION_ID, requesting_user_id=USER_ID)
    )

    assert isinstance(result, Failure)
    assert isinstance(result.failure(), TranscriptNotFoundError)
    assert repository.read_calls == []


async def test_empty_export_reads_one_empty_page() -> None:
    prepared, repository = _prepared({(None, None, None): Success(())})

    assert await _events(prepared) == []
    assert repository.read_calls == [(500, (None, None, None))]


async def test_partial_page_terminates_without_an_extra_read() -> None:
    row = _message(1)
    prepared, repository = _prepared({(None, None, None): Success((row,))})

    events = await _events(prepared)

    assert events == [TranscriptExportRow.model_validate(row.model_dump())]
    assert repository.read_calls == [(500, (None, None, None))]


async def test_full_and_multi_page_export_uses_full_ordering_cursor_without_duplicates() -> None:
    rows = (
        _message(1, turn_seq=1, ordinal=0),
        _message(2, turn_seq=1, ordinal=1),
        _message(3, turn_seq=2, ordinal=0),
        _message(4, turn_seq=2, ordinal=1),
    )
    prepared, repository = _prepared(
        {
            (None, None, None): Success(rows[:2]),
            (1, 1, 2): Success(rows[2:]),
            (2, 1, 4): Success(()),
        },
        page_rows=2,
    )

    events = await _events(prepared)

    assert [event.row_id for event in events if isinstance(event, TranscriptExportRow)] == [
        1,
        2,
        3,
        4,
    ]
    assert repository.read_calls == [
        (2, (None, None, None)),
        (2, (1, 1, 2)),
        (2, (2, 1, 4)),
    ]


async def test_mid_stream_failure_emits_one_typed_terminal_error() -> None:
    failure = Failure(
        TranscriptInfrastructureError(
            message="transcript read failed",
            operation="read_transcript",
        )
    )
    prepared, repository = _prepared(
        {
            (None, None, None): Success((_message(1), _message(2, ordinal=1))),
            (1, 1, 2): failure,
        },
        page_rows=2,
    )

    events = await _events(prepared)

    assert [event.row_id for event in events if isinstance(event, TranscriptExportRow)] == [1, 2]
    assert events[-1] == TranscriptExportError(
        code="TRANSCRIPT_DATABASE_ERROR",
        message="transcript read failed",
    )
    assert repository.read_calls == [
        (2, (None, None, None)),
        (2, (1, 1, 2)),
    ]


async def test_serialization_preserves_snake_case_redaction_and_metadata() -> None:
    row = _message(
        9,
        content="[REDACTED]",
        redacted=True,
        metadata={"source": "policy", "nested": {"reason": "secret"}},
    )
    prepared, _ = _prepared({(None, None, None): Success((row,))})

    event = (await _events(prepared))[0]
    assert isinstance(event, TranscriptExportRow)
    assert event.model_dump() == {
        "row_id": 9,
        "turn_seq": 1,
        "ordinal": 0,
        "kind": "assistant",
        "role": "assistant",
        "content": "[REDACTED]",
        "model": "gpt-test",
        "tokens_used": 42,
        "redacted": True,
        "created_at": NOW,
        "extra_metadata": {"source": "policy", "nested": {"reason": "secret"}},
    }
    assert event.model_dump(by_alias=True)["rowId"] == 9
    assert event.model_dump(by_alias=True)["tokensUsed"] == 42
    assert event.model_dump(by_alias=True)["extraMetadata"] == row.extra_metadata


async def test_monotonic_append_after_full_page_is_seen_once_by_keyset_cursor() -> None:
    first_page = (_message(1, turn_seq=1, ordinal=0), _message(2, turn_seq=1, ordinal=1))
    appended = _message(3, turn_seq=2, ordinal=0)
    prepared, _repository = _prepared(
        {
            (None, None, None): Success(first_page),
            (1, 1, 2): Success((appended,)),
        },
        page_rows=2,
    )

    events = await _events(prepared)

    assert [event.row_id for event in events if isinstance(event, TranscriptExportRow)] == [1, 2, 3]
    assert len({event.row_id for event in events if isinstance(event, TranscriptExportRow)}) == 3


async def test_events_invocations_have_independent_replayable_cursors() -> None:
    rows = (_message(1),)
    prepared, repository = _prepared({(None, None, None): Success(rows)})

    first = await _events(prepared)
    second = await _events(prepared)

    assert first == second
    assert repository.read_calls == [
        (500, (None, None, None)),
        (500, (None, None, None)),
    ]


class _ConcurrentTranscriptRepository(_FakeTranscriptRepository):
    """Repository that forces two consumers to overlap across two pages."""

    def __init__(
        self,
        first_pages: dict[str, tuple[TranscriptMessage, ...]],
        second_pages: dict[str, TranscriptMessage],
        first_barrier: asyncio.Barrier,
        second_barrier: asyncio.Barrier,
    ) -> None:
        super().__init__({})
        self._first_pages = first_pages
        self._second_pages = second_pages
        self._first_barrier = first_barrier
        self._second_barrier = second_barrier
        self.second_cursors: dict[str, Cursor] = {}

    async def read_transcript(
        self,
        *,
        session_id: str,
        user_id: str,
        limit: int,
        after_turn_seq: int | None = None,
        after_ordinal: int | None = None,
        after_id: int | None = None,
    ) -> Any:
        assert (session_id, user_id) == (SESSION_ID, USER_ID)
        task = asyncio.current_task()
        assert task is not None
        consumer = task.get_name()
        cursor = (after_turn_seq, after_ordinal, after_id)
        self.read_calls.append((limit, cursor))
        if cursor == (None, None, None):
            await self._first_barrier.wait()
            return Success(self._first_pages[consumer])

        self.second_cursors[consumer] = cursor
        await self._second_barrier.wait()
        return Success((self._second_pages[consumer],))


async def test_one_prepared_export_supports_true_concurrent_consumers() -> None:
    first_pages = {
        "consumer-a": (_message(1, turn_seq=1, ordinal=0), _message(2, turn_seq=1, ordinal=1)),
        "consumer-b": (_message(10, turn_seq=5, ordinal=0), _message(11, turn_seq=5, ordinal=1)),
    }
    second_pages = {
        "consumer-a": _message(3, turn_seq=2, ordinal=0),
        "consumer-b": _message(12, turn_seq=6, ordinal=0),
    }
    repository = _ConcurrentTranscriptRepository(
        first_pages,
        second_pages,
        asyncio.Barrier(2),
        asyncio.Barrier(2),
    )
    prepared = PreparedTranscriptExport(
        repository=repository,
        session_id=SESSION_ID,
        user_id=USER_ID,
    )
    prepared._EXPORT_PAGE_ROWS = 2

    first_task = asyncio.create_task(_events(prepared), name="consumer-a")
    second_task = asyncio.create_task(_events(prepared), name="consumer-b")
    first, second = await asyncio.gather(first_task, second_task)

    assert [event.row_id for event in first if isinstance(event, TranscriptExportRow)] == [1, 2, 3]
    assert [event.row_id for event in second if isinstance(event, TranscriptExportRow)] == [
        10,
        11,
        12,
    ]
    assert repository.second_cursors == {
        "consumer-a": (1, 1, 2),
        "consumer-b": (5, 1, 11),
    }
