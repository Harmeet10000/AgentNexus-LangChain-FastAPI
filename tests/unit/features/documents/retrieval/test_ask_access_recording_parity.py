"""Access-recording parity (mount-retrieval-graph task 2.3).

The service wrapper around the ask invocation records ``document.read`` for
the same scope/outcome combinations the inline ask wrapper does — success,
failure, and the denied empty scope — and the graph itself never records
audit (the recording stays at the service boundary per the design). This
gates the cutover: whatever serves the graph must keep this wrapper's
recording behaviour.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from returns.result import Failure, Success

import app.features.documents.application.queries.documents as documents_module
from app.features.documents.api.http.dto import UnifiedAskRequest
from app.features.documents.application.access_audit import (
    InMemoryDocumentAccessAuditRecorder,
)
from app.features.documents.application.queries.documents import DocumentQueryService
from app.features.documents.application.retrieval.graph import RetrievalEngine
from app.features.documents.application.retrieval.graph.state import GeneratedAnswer, QueryPlan
from app.features.documents.domain.errors import DocumentDatabaseError

if TYPE_CHECKING:
    from typing import Any

_DOC_ID = "doc-a"
_TENANT_A = "tenant-a"


class _OkRepository:
    async def bm25_search(self, **_kwargs: object) -> Success[list[dict[str, Any]]]:
        return Success([{"chunk_id": "chunk-1", "score": 2.0}])

    async def vector_search(self, **_kwargs: object) -> Success[list[dict[str, Any]]]:
        return Success([])

    async def trigram_search(self, **_kwargs: object) -> Success[list[dict[str, Any]]]:
        return Success([])

    async def fetch_chunks_by_ids(
        self, chunk_ids: list[str], **_kwargs: object
    ) -> Success[dict[str, dict[str, Any]]]:
        return Success(
            {
                chunk_id: {
                    "document_id": _DOC_ID,
                    "title": "t",
                    "content": "content",
                    "preamble": "",
                    "document_version": 1,
                    "contract_index": 0,
                    "pipeline_version": "canonical",
                    "chunk_index": 0,
                    "chunk_kind": "legal_contract",
                    "clause_type": "payment",
                    "chunk_metadata": {},
                    "quality_warnings": [],
                    "graphiti_verified": False,
                    "graph_indexed_at": None,
                }
                for chunk_id in chunk_ids
            }
        )

    async def resolve_scope_versions(
        self, *, tenant_id: str, document_ids: tuple[str, ...]
    ) -> tuple[tuple[str, int, str | None], ...]:
        return ((_DOC_ID, 1, None),) if document_ids else ()


class _FailingRepository(_OkRepository):
    async def bm25_search(self, **_kwargs: object) -> Failure[DocumentDatabaseError]:
        return Failure(
            DocumentDatabaseError(message="fused retrieval failed", details={"branch": "bm25"})
        )


class _Handler:
    def __init__(self, schema: type) -> None:
        self._schema = schema

    async def ainvoke(self, _messages: list[Any]) -> Any:
        if self._schema is QueryPlan:
            return QueryPlan(rewritten_query="payment obligations")
        return GeneratedAnswer(answer="answer", citations=[], confidence="high")


class _ScriptedAskLLM:
    def with_structured_output(self, schema: type) -> _Handler:
        return _Handler(schema)


class _PassThroughReranker:
    async def rerank(self, _query: str, chunks: list[Any], *, limit: int) -> list[Any]:
        return chunks[:limit]


async def _fake_embed(*_args: object, **_kwargs: object) -> list[float]:
    return [0.1, 0.2, 0.3]


@pytest.fixture
def _wired(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(documents_module, "embed_text", _fake_embed)
    import app.features.documents.application.retrieval.graph.nodes as graph_nodes

    monkeypatch.setattr(graph_nodes, "embed_text", _fake_embed)
    monkeypatch.setattr(graph_nodes, "get_configured_reranker", _PassThroughReranker)


def _service(repo: Any) -> DocumentQueryService:
    """The graph-served ask path (mount-retrieval-graph 4.x): the recording
    wrapper is unchanged, the engine is the compile-once graph."""
    return DocumentQueryService(
        repo,  # ty: ignore[invalid-argument-type]
        _ScriptedAskLLM,
        None,
        None,
        audit_recorder=InMemoryDocumentAccessAuditRecorder(),
        retrieval_engine=RetrievalEngine(
            session_factory=None,  # ty: ignore[invalid-argument-type]
            redis=None,
            graphiti=None,
            llm_factory=_ScriptedAskLLM,
        ),
    )


def _payload() -> UnifiedAskRequest:
    return UnifiedAskRequest(query="payment obligations", bypass_cache=True)


@pytest.mark.usefixtures("_wired")
async def test_successful_ask_records_document_read_for_the_scope() -> None:
    service = _service(_OkRepository())
    result = await service.ask(
        user_id="user-1",
        tenant_id=_TENANT_A,
        allowed_doc_ids=[_DOC_ID],
        payload=_payload(),
        require_graphiti_verified=False,
    )
    assert isinstance(result, Success)
    events = service.audit_recorder.events  # ty: ignore[union-attr]
    assert len(events) == 1
    event = events[0]
    assert event.operation == "document.read"
    assert event.requested_document_ids == (_DOC_ID,)
    assert event.document_ids == (_DOC_ID,)
    assert event.tenant_id == _TENANT_A
    assert event.principal_id == "user-1"
    assert event.outcome == "ok"


@pytest.mark.usefixtures("_wired")
async def test_denied_empty_scope_records_the_denial() -> None:
    service = _service(_OkRepository())
    result = await service.ask(
        user_id="user-1",
        tenant_id=_TENANT_A,
        allowed_doc_ids=(),
        payload=_payload(),
        require_graphiti_verified=False,
    )
    assert isinstance(result, Success)
    events = service.audit_recorder.events  # ty: ignore[union-attr]
    assert len(events) == 1
    event = events[0]
    assert event.operation == "document.read"
    assert event.requested_document_ids == ()
    assert event.document_ids == ()
    assert event.outcome == "denied"
    assert event.reason == "requested document scope is empty"


@pytest.mark.usefixtures("_wired")
async def test_failed_ask_records_a_denied_read() -> None:
    service = _service(_FailingRepository())
    result = await service.ask(
        user_id="user-1",
        tenant_id=_TENANT_A,
        allowed_doc_ids=[_DOC_ID],
        payload=_payload(),
        require_graphiti_verified=False,
    )
    assert isinstance(result, Failure)
    events = service.audit_recorder.events  # ty: ignore[union-attr]
    assert len(events) == 1
    event = events[0]
    assert event.operation == "document.read"
    assert event.outcome == "denied"
    assert event.document_ids == ()
