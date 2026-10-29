"""Document-content source discipline: injection, node mapping, preflight.

Covers the canonical-text seam (the 2026-10 ingestion blocker): the ingestion
node must feed normalization from the injected ``DocumentContentSource`` and
fail CLEANLY (state errors, never a raise, never silent ``""``) when it
cannot; ``create_session_for_document`` must reject a missing/foreign doc_id
with the typed ``AgentSaulDocumentNotFoundError``. Test doubles follow the
runtime-wiring tests' SimpleNamespace/parity discipline — no TestClient.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING, cast
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest
from returns.result import Failure, Success
from sqlalchemy.exc import SQLAlchemyError

from app.features.agent_saul.adapters.document_source import (
    DOCUMENT_SOURCE_CONFIG_KEY,
    DocumentTextFailure,
    DocumentTextFailureCode,
    PostgresDocumentContentSource,
    build_latest_version_statement,
    document_content_source_from_config,
    parse_document_id,
)
from app.features.agent_saul.api.dto import CreateSessionResponse
from app.features.agent_saul.application.service import AgentSaulService, _document_preflight_error
from app.features.agent_saul.domain.errors import (
    AgentSaulDocumentNotFoundError,
    AgentSaulInfrastructureError,
    AgentSaulValidationError,
)
from app.features.agent_saul.domain.state import WorkflowStatus
from app.features.agent_saul.graph.nodes import make_ingestion_node
from app.features.documents.domain.errors import DocumentDatabaseError, DocumentNotFoundError

if TYPE_CHECKING:
    from collections.abc import Callable

pytestmark = pytest.mark.unit

_USER = "user-1"
_DOC_ID = str(uuid4())


# ---------------------------------------------------------------------------
# Harness: duck-typed session/repository doubles (the repo's own test style)
# ---------------------------------------------------------------------------


class _FakeSession:
    def __init__(self, contents: list[str] | None = None) -> None:
        if contents is None:
            rows: object = SimpleNamespace(scalars=lambda: SimpleNamespace(all=list))
        else:
            rows = SimpleNamespace(scalars=lambda: SimpleNamespace(all=lambda: contents))
        self.execute = AsyncMock(return_value=rows)
        self.rollbacks = 0

    async def __aenter__(self) -> _FakeSession:
        return self

    async def __aexit__(self, *exc: object) -> bool:
        return False

    async def rollback(self) -> None:
        self.rollbacks += 1


def _session_factory(session: object | None = None) -> MagicMock:
    factory = MagicMock()
    if session is not None:
        factory.return_value = session
    return factory


class _RepoDouble:
    """Replays one pre-decided repository read — the doc state under test."""

    def __init__(self, outcome: object) -> None:
        self._outcome = outcome

    async def get_document_by_id(self, *, user_id: str, document_id: str) -> object:
        return self._outcome


def _repository_factory(outcome: object) -> Callable[[object], object]:
    return lambda _session: _RepoDouble(outcome)


_NOT_FOUND_REPO = Failure(
    DocumentNotFoundError(message="Document not found for the given user and document ID")
)
_DB_DOWN_REPO = Failure(
    DocumentDatabaseError(message="Database error while fetching document by ID")
)


def _state(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "user_id": _USER,
        "doc_id": _DOC_ID,
        "retry_count": 0,
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# Injection seam: config resolution
# ---------------------------------------------------------------------------


def test_config_resolver_returns_injected_source() -> None:
    source = PostgresDocumentContentSource(_session_factory())
    config: dict[str, object] = {"configurable": {DOCUMENT_SOURCE_CONFIG_KEY: source}}

    assert document_content_source_from_config(cast("dict[str, object]", config)) is source


def test_config_resolver_without_key_or_configurable_resolves_none() -> None:
    assert document_content_source_from_config({}) is None
    assert document_content_source_from_config({"configurable": {"thread_id": "t"}}) is None


# ---------------------------------------------------------------------------
# Adapter: Postgres reads through fakes (no database)
# ---------------------------------------------------------------------------


async def test_fetch_text_rejoins_latest_version_contents_in_index_order() -> None:
    session = _FakeSession(contents=["Second chunk", "", "First chunk"])
    source = PostgresDocumentContentSource(
        _session_factory(session),  # type: ignore[arg-type] — duck session under test
        repository_factory=_repository_factory(Success(1)),  # type: ignore[arg-type]
    )

    result = await source.fetch_text(user_id=_USER, doc_id=_DOC_ID)

    assert isinstance(result, Success)
    assert result.unwrap() == "Second chunk\n\nFirst chunk"


async def test_fetch_text_missing_document_maps_to_not_found() -> None:
    session = _FakeSession()
    source = PostgresDocumentContentSource(
        _session_factory(session),  # type: ignore[arg-type] — duck session under test
        repository_factory=_repository_factory(_NOT_FOUND_REPO),  # type: ignore[arg-type]
    )

    result = await source.fetch_text(user_id=_USER, doc_id=_DOC_ID)

    assert isinstance(result, Failure)
    assert result.failure().code is DocumentTextFailureCode.NOT_FOUND


async def test_fetch_text_document_without_chunks_maps_to_empty() -> None:
    session = _FakeSession(contents=[])
    source = PostgresDocumentContentSource(
        _session_factory(session),  # type: ignore[arg-type] — duck session under test
        repository_factory=_repository_factory(Success(1)),  # type: ignore[arg-type]
    )

    result = await source.fetch_text(user_id=_USER, doc_id=_DOC_ID)

    assert isinstance(result, Failure)
    assert result.failure().code is DocumentTextFailureCode.EMPTY


async def test_fetch_text_outage_maps_to_unavailable() -> None:
    outage = SQLAlchemyError("postgres down")

    class _ExplodingSession(_FakeSession):
        async def __aenter__(self) -> _FakeSession:
            raise outage

    source = PostgresDocumentContentSource(
        _session_factory(_ExplodingSession()),  # type: ignore[arg-type] — duck session under test
        repository_factory=_repository_factory(Success(1)),  # type: ignore[arg-type]
    )

    result = await source.fetch_text(user_id=_USER, doc_id=_DOC_ID)

    assert isinstance(result, Failure)
    assert result.failure().code is DocumentTextFailureCode.UNAVAILABLE


async def test_fetch_text_invalid_uuid_refuses_without_opening_a_session() -> None:
    factory = _session_factory(None)
    factory.side_effect = AssertionError("session must not open for an invalid id")
    source = PostgresDocumentContentSource(factory)  # type: ignore[arg-type]

    result = await source.fetch_text(user_id=_USER, doc_id="not-a-uuid")

    assert isinstance(result, Failure)
    assert result.failure().code is DocumentTextFailureCode.NOT_FOUND


def test_parse_document_id_semantics() -> None:
    assert parse_document_id(_DOC_ID) is not None
    assert parse_document_id("") is None
    assert parse_document_id("1e40b7c1-fake-uuid") is None


def test_latest_version_statement_compiles_with_the_truth_columns() -> None:
    from sqlalchemy.dialects import postgresql

    sql = str(build_latest_version_statement(_USER, uuid4()).compile(dialect=postgresql.dialect()))
    assert "content" in sql
    assert "chunk_index" in sql
    assert "max" in sql.lower()  # latest-version subquery drives the filter


# ---------------------------------------------------------------------------
# Ingestion node maps provider outcomes to the existing state-error taxonomy
# ---------------------------------------------------------------------------


def _node_config(source_or_none: object | None) -> dict[str, object]:
    if source_or_none is None:
        return {}
    return {"configurable": {DOCUMENT_SOURCE_CONFIG_KEY: source_or_none}}


class _StaticSource:
    def __init__(self, outcome: object) -> None:
        self._outcome = outcome

    async def ensure_document(self, *, user_id: str, doc_id: str) -> object:  # pragma: no cover
        return Success(None)

    async def fetch_text(self, *, user_id: str, doc_id: str) -> object:
        return self._outcome


class _CoughingSource(_StaticSource):
    async def fetch_text(self, *, user_id: str, doc_id: str) -> object:
        leak = RuntimeError("provider exploded")
        raise leak


async def test_node_success_feeds_document_text_and_status_with_no_errors() -> None:
    node = make_ingestion_node()
    source = _StaticSource(Success("MASTER SERVICES AGREEMENT\nsignature (~) clause"))

    output = await node(_state(), _node_config(source))  # type: ignore[arg-type]

    assert output["document_text"] == "MASTER SERVICES AGREEMENT\nsignature (~) clause"
    assert output["status"] is WorkflowStatus.NORMALIZING
    assert "errors" not in output


@pytest.mark.parametrize(
    ("code", "expected_state_code", "retryable"),
    [
        (DocumentTextFailureCode.NOT_FOUND, "DOCUMENT_NOT_FOUND", False),
        (DocumentTextFailureCode.EMPTY, "DOCUMENT_EMPTY_TEXT", False),
        (DocumentTextFailureCode.UNAVAILABLE, "DOCUMENT_SOURCE_UNAVAILABLE", True),
    ],
)
async def test_node_failure_shapes_return_clean_state_errors(
    code: DocumentTextFailureCode,
    expected_state_code: str,
    retryable: bool,
) -> None:
    node = make_ingestion_node()
    source = _StaticSource(
        Failure(DocumentTextFailure(code=code, message=f"failure {code}", doc_id=_DOC_ID))
    )

    output = await node(_state(), _node_config(source))  # type: ignore[arg-type]

    error = output["errors"][0]
    assert output["status"] is WorkflowStatus.FAILED
    assert error.node == "ingestion"
    assert error.code == expected_state_code
    assert error.retryable is retryable


async def test_node_missing_injection_is_a_clean_state_error_not_a_raise() -> None:
    node = make_ingestion_node()

    output = await node(_state(), _node_config(None))  # type: ignore[arg-type]

    error = output["errors"][0]
    assert output["status"] is WorkflowStatus.FAILED
    assert error.code == "DOCUMENT_SOURCE_NOT_WIRED"


async def test_node_leaked_provider_failure_is_counted_not_raised() -> None:
    node = make_ingestion_node()
    source = _CoughingSource(Success("never"))

    output = await node(_state(), _node_config(source))  # type: ignore[arg-type]

    error = output["errors"][0]
    assert output["status"] is WorkflowStatus.FAILED
    assert error.code == "DOCUMENT_SOURCE_UNAVAILABLE"
    assert error.retryable is True


async def test_node_still_respects_the_retry_budget_before_any_read() -> None:
    node = make_ingestion_node()
    source = _StaticSource(Success("text"))

    output = await node(_state(retry_count=3), _node_config(source))  # type: ignore[arg-type]

    assert output["errors"][0].code == "MAX_RETRIES_EXCEEDED"


# ---------------------------------------------------------------------------
# Session pre-flight: doc existence/ownership typed validation path
# ---------------------------------------------------------------------------


class _EnsureOnlySource:
    def __init__(self, outcome: object) -> None:
        self._outcome = outcome

    async def ensure_document(self, *, user_id: str, doc_id: str) -> object:
        assert user_id == _USER
        return self._outcome

    async def fetch_text(self, **kwargs: object) -> object:  # pragma: no cover
        contract = AssertionError("session creation must not read document text")
        raise contract


async def test_session_creation_validates_doc_and_returns_response_on_check() -> None:
    source = _EnsureOnlySource(Success(None))

    result = await AgentSaulService.create_session_for_document(
        websocket_url="wss://example.test/ws/thread-1",
        thread_id="thread-1",
        doc_id=_DOC_ID,
        user_id=_USER,
        document_source=source,
    )

    assert isinstance(result, Success)
    assert result.unwrap() == CreateSessionResponse(
        thread_id="thread-1", ws_url="wss://example.test/ws/thread-1"
    )


async def test_session_creation_missing_or_foreign_doc_is_typed_not_500() -> None:
    source = _EnsureOnlySource(
        Failure(
            DocumentTextFailure(
                code=DocumentTextFailureCode.NOT_FOUND,
                message="Document not found for the given user and document ID",
                doc_id=_DOC_ID,
            )
        )
    )

    result = await AgentSaulService.create_session_for_document(
        websocket_url="wss://example.test/ws/thread-1",
        thread_id="thread-1",
        doc_id=_DOC_ID,
        user_id=_USER,
        document_source=source,
    )

    assert isinstance(result, Failure)
    failure = result.failure()
    assert isinstance(failure, AgentSaulDocumentNotFoundError)
    assert failure.doc_id == _DOC_ID
    assert failure.user_id == _USER


async def test_session_creation_source_outage_is_typed_not_500() -> None:
    source = _EnsureOnlySource(
        Failure(
            DocumentTextFailure(
                code=DocumentTextFailureCode.UNAVAILABLE,
                message="Document content source is unavailable",
                doc_id=_DOC_ID,
            )
        )
    )

    result = await AgentSaulService.create_session_for_document(
        websocket_url="wss://example.test/ws/thread-1",
        thread_id="thread-1",
        doc_id=_DOC_ID,
        user_id=_USER,
        document_source=source,
    )

    assert isinstance(result, Failure)
    assert isinstance(result.failure(), AgentSaulInfrastructureError)


def test_document_preflight_error_mapping_covers_the_union() -> None:
    not_found = _document_preflight_error(
        DocumentTextFailure(
            code=DocumentTextFailureCode.NOT_FOUND, message="missing", doc_id=_DOC_ID
        ),
        user_id=_USER,
        doc_id=_DOC_ID,
    )
    unavailable = _document_preflight_error(
        DocumentTextFailure(
            code=DocumentTextFailureCode.UNAVAILABLE, message="down", doc_id=_DOC_ID
        ),
        user_id=_USER,
        doc_id=_DOC_ID,
    )
    empty = _document_preflight_error(
        DocumentTextFailure(code=DocumentTextFailureCode.EMPTY, message="void", doc_id=_DOC_ID),
        user_id=_USER,
        doc_id=_DOC_ID,
    )

    assert isinstance(not_found, AgentSaulDocumentNotFoundError)
    assert isinstance(unavailable, AgentSaulInfrastructureError)
    assert isinstance(empty, AgentSaulValidationError)
