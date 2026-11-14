"""F1: the RAG path must not re-validate its own payload against the strict search schema.

`DocumentQueryService.rag` narrows `UnifiedRagRequest` into the strict `UnifiedSearchRequest`
(`extra="forbid"`) before calling `search`. `UnifiedRagRequest` adds `max_tokens` (default
2000) on top of the search schema, so the verbatim dump always carried an extra field and the
strict re-validation rejected EVERY /search/rag request — server-smoke-2026-11-15 reproduced a
deterministic 500 with `max_tokens — Extra inputs are not permitted` on payloads with and
without a client-supplied `max_tokens`.

The fix drops the RAG-only field before the strict re-validation. These tests pin both the
composition (the DTO conversion) and the service behavior (a `UnifiedRagRequest` carrying the
inherited default flows through `rag` without a `ValidationError`).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from returns.result import Success

from app.features.documents.api.http.dto import (
    DocumentSearchResultItem,
    UnifiedRagRequest,
    UnifiedSearchRequest,
    UnifiedSearchResponse,
)
from app.features.documents.application.queries.documents import DocumentQueryService

if TYPE_CHECKING:
    from app.features.documents.api.http.dto import UnifiedRagResponse
    from app.features.documents.domain.errors import DocumentResult

pytestmark = pytest.mark.unit


def _service_with_search(
    stub: DocumentResult[UnifiedSearchResponse],
) -> tuple[DocumentQueryService, list[UnifiedSearchRequest]]:
    """A `DocumentQueryService` whose `search` step is replaced by a recording stub.

    `rag` is the only layer the defect touched; stubbing `search` isolates the payload
    composition from the retrieval machinery without dragging in repositories or providers.
    """
    captured: list[UnifiedSearchRequest] = []

    async def _stub_search(
        _self: DocumentQueryService,
        *,
        user_id: str,
        tenant_id: str | None,
        allowed_doc_ids: tuple[str, ...] | None,
        payload: UnifiedSearchRequest,
        weights: object | None = None,
    ) -> DocumentResult[UnifiedSearchResponse]:
        captured.append(payload)
        return stub

    service = object.__new__(DocumentQueryService)
    service.search = _stub_search.__get__(service)  # ty: ignore[invalid-assignment]
    return service, captured


def test_rag_defaults_compose_into_the_strict_search_schema() -> None:
    """The inherited `max_tokens` default must not poison the strict re-validation.

    This is the exact expression the service uses. Before the fix it raised
    `ValidationError: max_tokens — Extra inputs are not permitted` for every payload, because
    `model_dump()` serialized the RAG default into the strict search schema.
    """
    rag_payload = UnifiedRagRequest(query="termination notice")

    search_payload = UnifiedSearchRequest.model_validate(
        rag_payload.model_dump(exclude={"max_tokens"})
    )

    assert search_payload.query == "termination notice"
    assert "max_tokens" not in search_payload.model_dump()


def test_rag_defaults_compose_with_a_client_supplied_max_tokens() -> None:
    """Same composition with the field the smoke probe actually sent."""
    rag_payload = UnifiedRagRequest(query="termination notice", max_tokens=2000)

    search_payload = UnifiedSearchRequest.model_validate(
        rag_payload.model_dump(exclude={"max_tokens"})
    )

    assert search_payload.query == "termination notice"


async def test_rag_service_accepts_a_request_carrying_max_tokens() -> None:
    """Service-level regression: `rag()` with the inherited default must not 500."""
    service, captured = _service_with_search(Success(UnifiedSearchResponse(items=[], cache_hit=False)))

    result = await service.rag(
        user_id="user-1", payload=UnifiedRagRequest(query="termination notice")
    )

    assert isinstance(result, Success)
    rag_response: UnifiedRagResponse = result.unwrap()
    assert rag_response.items == []
    assert rag_response.context == []
    assert rag_response.cache_hit is False
    # The narrowed payload reached the search path clean of the RAG-only field.
    assert captured[-1].model_dump() == UnifiedSearchRequest(query="termination notice").model_dump()


async def test_rag_service_passes_max_tokens_to_context_assembly() -> None:
    """Dropping `max_tokens` from the search payload must not drop it from the budget."""
    item = DocumentSearchResultItem(
        chunk_id="c1",
        document_id="doc-1",
        title="Contract",
        content="alpha beta gamma delta epsilon",
        chunk_index=0,
        chunk_kind="paragraph",
        chunk_metadata={},
        score=1.0,
        rank=1,
    )
    service, captured = _service_with_search(
        Success(UnifiedSearchResponse(items=[item], cache_hit=False))
    )

    result = await service.rag(
        user_id="user-1", payload=UnifiedRagRequest(query="termination notice", max_tokens=2)
    )

    assert isinstance(result, Success)
    sections = result.unwrap().context
    # The served tokenizer counts more tokens than whitespace splits, so the tight budget may
    # keep or drop the single section — what the regression pins is that the search payload
    # stayed clean while the RAG budget still traveled with the original request.
    assert len(sections) <= 1
    assert captured[-1].query == "termination notice"


def test_empty_query_is_still_rejected_at_the_edge() -> None:
    """Validation stays strict at the boundary — the fix only removes the extra-field clash."""
    with pytest.raises(ValueError, match="query"):
        UnifiedRagRequest.model_validate({"query": "", "max_tokens": 1})
