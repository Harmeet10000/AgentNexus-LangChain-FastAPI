"""HTTP contract tests for the Agent Saul transcript export adapter."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from returns.result import Failure, Success

from app.features.agent_saul.dependencies import get_transcript_exporter
from app.features.agent_saul.router import router
from app.features.agent_saul.transcript.errors import (
    TranscriptNotFoundError,
    TranscriptSchemaError,
)
from app.features.agent_saul.transcript.export import (
    ExportTranscript,
    TranscriptExportError,
    TranscriptExportRow,
)
from app.features.auth.api.http.dependencies import get_token_claims
from app.security.tokens import TokenClaims

if TYPE_CHECKING:
    from collections.abc import AsyncIterator
    from typing import Any

    from returns.result import Result

    from app.features.agent_saul.transcript.export import TranscriptExportEvent

pytestmark = pytest.mark.unit

SESSION_ID = "session-1"
USER_ID = "user-1"


class _PreparedExport:
    def __init__(self, events: tuple[TranscriptExportEvent, ...]) -> None:
        self._events = events
        self.events_calls = 0

    async def events(self) -> AsyncIterator[TranscriptExportEvent]:
        self.events_calls += 1
        for event in self._events:
            yield event


class _FakeExporter:
    def __init__(self, result: Result[Any, Any]) -> None:
        self._result = result
        self.prepare_calls: list[ExportTranscript] = []

    async def prepare(self, request: ExportTranscript) -> Result[Any, Any]:
        self.prepare_calls.append(request)
        return self._result


def _claims() -> TokenClaims:
    return TokenClaims(
        sub=USER_ID,
        jti="jti-1",
        sid="session-token-1",
        role="user",
        permissions=[],
        token_type="access",
    )


def _client(exporter: _FakeExporter) -> TestClient:
    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[get_transcript_exporter] = lambda: exporter
    app.dependency_overrides[get_token_claims] = _claims
    return TestClient(app)


def test_ownership_preflight_returns_not_found_before_streaming_or_page_read() -> None:
    exporter = _FakeExporter(
        Failure(
            TranscriptNotFoundError(
                message="transcript session not found",
                session_id=SESSION_ID,
            )
        )
    )

    with _client(exporter) as client:
        response = client.get(f"/agent-saul/sessions/{SESSION_ID}/export")

    assert response.status_code == 404
    assert response.json()["error"]["code"] == "TRANSCRIPT_NOT_FOUND"
    assert exporter.prepare_calls == [
        ExportTranscript(session_id=SESSION_ID, requesting_user_id=USER_ID)
    ]


def test_schema_preflight_returns_unprocessable_before_streaming_or_page_read() -> None:
    exporter = _FakeExporter(
        Failure(
            TranscriptSchemaError(
                message="unsupported transcript schema",
                operation="get_session_meta",
                expected_version=1,
                found_version=99,
            )
        )
    )

    with _client(exporter) as client:
        response = client.get(f"/agent-saul/sessions/{SESSION_ID}/export")

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "TRANSCRIPT_SCHEMA_VERSION"


def test_successful_export_is_ndjson_download_with_snake_case_keys() -> None:
    prepared = _PreparedExport(
        (
            TranscriptExportRow(
                row_id=7,
                turn_seq=2,
                ordinal=1,
                kind="assistant",
                role="assistant",
                content="[REDACTED]",
                tokens_used=42,
                redacted=True,
                extra_metadata={"source": "policy"},
            ),
        )
    )
    exporter = _FakeExporter(Success(prepared))

    with _client(exporter) as client:
        response = client.get(f"/agent-saul/sessions/{SESSION_ID}/export")

    assert response.status_code == 200
    assert response.headers["content-type"] == "application/x-ndjson"
    assert response.headers["content-disposition"] == (
        'attachment; filename="transcript-session-1.ndjson"'
    )
    row = response.text.strip().splitlines()[0]
    assert '"row_id": 7' in row
    assert '"turn_seq": 2' in row
    assert '"tokens_used": 42' in row
    assert '"rowId"' not in row
    assert '"content": "[REDACTED]"' in row
    assert prepared.events_calls == 1


def test_mid_stream_failure_is_emitted_as_typed_error_ndjson() -> None:
    prepared = _PreparedExport(
        (
            TranscriptExportRow(
                row_id=1,
                turn_seq=1,
                ordinal=0,
                kind="user",
                role="user",
                content="question",
                redacted=False,
            ),
            TranscriptExportError(
                code="TRANSCRIPT_DATABASE_ERROR",
                message="transcript read failed",
            ),
        )
    )
    exporter = _FakeExporter(Success(prepared))

    with _client(exporter) as client:
        response = client.get(f"/agent-saul/sessions/{SESSION_ID}/export")

    lines = response.text.strip().splitlines()
    assert response.status_code == 200
    assert lines[-1] == (
        '{"type": "error", "code": "TRANSCRIPT_DATABASE_ERROR", '
        '"message": "transcript read failed"}'
    )
    assert prepared.events_calls == 1
