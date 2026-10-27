"""Admission gating for Agent Saul document ingestion state."""

from __future__ import annotations

from typing import TYPE_CHECKING
from uuid import uuid4

import pytest
from returns.result import Success

from app.features.agent_saul.document_source import DocumentReadiness
from app.features.agent_saul.exceptions import (
    AgentSaulDocumentIngestionFailedError,
    AgentSaulDocumentNotReadyError,
)
from app.features.agent_saul.service import AgentSaulService

if TYPE_CHECKING:
    from app.features.agent_saul.document_source import DocumentReadinessState

pytestmark = pytest.mark.unit

_USER = "user-1"
_DOC_A = str(uuid4())
_DOC_B = str(uuid4())


class _ReadinessSource:
    def __init__(self, statuses: dict[str, DocumentReadiness]) -> None:
        self.statuses = statuses
        self.checked: list[str] = []

    async def ensure_document(self, *, user_id: str, doc_id: str) -> object:
        assert user_id == _USER
        return Success(None)

    async def check_readiness(self, *, user_id: str, doc_id: str) -> object:
        assert user_id == _USER
        self.checked.append(doc_id)
        return Success(self.statuses[doc_id])


async def _create(source: _ReadinessSource, ids: list[str]) -> object:
    return await AgentSaulService.create_session_for_document(
        websocket_url="wss://example.test/ws/thread-1",
        thread_id="thread-1",
        doc_id=ids[0],
        allowed_doc_ids=ids,
        user_id=_USER,
        document_source=source,
    )


async def test_completed_document_is_admissible() -> None:
    source = _ReadinessSource({_DOC_A: DocumentReadiness(status="completed")})

    result = await _create(source, [_DOC_A])

    assert isinstance(result, Success)
    assert result.unwrap().thread_id == "thread-1"
    assert source.checked == [_DOC_A]


async def test_completed_with_warnings_is_admissible_and_keeps_warning_details() -> None:
    source = _ReadinessSource(
        {
            _DOC_A: DocumentReadiness(
                status="completed_with_warnings",
                warning_count=1,
                warnings=({"code": "LOW_TEXT_CONFIDENCE", "message": "scan quality"},),
            )
        }
    )

    result = await _create(source, [_DOC_A])

    assert result.unwrap().thread_id == "thread-1"


@pytest.mark.parametrize(
    ("status", "expected"),
    [
        ("not_ready", AgentSaulDocumentNotReadyError),
        ("failed", AgentSaulDocumentIngestionFailedError),
    ],
)
async def test_non_terminal_or_failed_document_is_rejected(
    status: DocumentReadinessState, expected: type[object]
) -> None:
    source = _ReadinessSource(
        {
            _DOC_A: DocumentReadiness(
                status=status,
                error_code="PARSE_FAILED" if status == "failed" else None,
                error_message="parse failed" if status == "failed" else None,
            )
        }
    )

    result = await _create(source, [_DOC_A])

    assert result.failure() is not None
    assert isinstance(result.failure(), expected)


async def test_mixed_multi_document_scope_rejects_before_session_creation() -> None:
    source = _ReadinessSource(
        {
            _DOC_A: DocumentReadiness(status="completed"),
            _DOC_B: DocumentReadiness(status="not_ready"),
        }
    )

    result = await _create(source, [_DOC_A, _DOC_B])

    assert isinstance(result.failure(), AgentSaulDocumentNotReadyError)
    assert source.checked == [_DOC_A, _DOC_B]


async def test_rechecking_same_scope_after_status_change_rejects_resume_like_retry() -> None:
    source = _ReadinessSource({_DOC_A: DocumentReadiness(status="completed")})

    first = await _create(source, [_DOC_A])
    source.statuses[_DOC_A] = DocumentReadiness(status="failed", error_message="worker stopped")
    second = await _create(source, [_DOC_A])

    assert first.unwrap().thread_id == "thread-1"
    assert isinstance(second.failure(), AgentSaulDocumentIngestionFailedError)
