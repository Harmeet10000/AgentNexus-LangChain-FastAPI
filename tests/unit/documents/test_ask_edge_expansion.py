"""The ask path consumes typed relationship edges end-to-end.

These tests drive the live inline `ask` loop (`DocumentQueryService._ask`) with
hermetic doubles: a fake Graphiti that emulates the provider's group scoping, a
stub repository, and a structured-output LLM double. They prove:

- a query touching edges gets edge-grounded citations carrying the same
  server-derived provenance discipline as LLM citations (document identity
  from the hydrated row, never from edge text or model output);
- a tenant cannot expand through another tenant's edges — the service only
  ever requests the authorized tenant's graph partitions;
- a query that touches no edges keeps the exact pre-edge contract: no
  ``grounding_edges`` key in the generator payload and no extra citations.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest
from returns.result import Success

import app.features.documents.application.queries.documents as documents_module
from app.features.documents.api.http.dto import UnifiedAskRequest
from app.features.documents.application.queries.documents import DocumentQueryService
from app.features.documents.retrieval.graph import (
    Citation,
    ContextGrade,
    GeneratedAnswer,
    QueryPlan,
    RetrievedChunk,
)
from app.retrieval.graphiti import client as graphiti_client_module

if TYPE_CHECKING:
    from typing import Any

_CHUNK_ID = "chunk-1"
_DOC_ID = "doc-a"
_TENANT_A = "tenant-a"
_TENANT_B = "tenant-b"

_EDGE_BODY = (
    "Acme Corp obliges GlobalTech Ltd. Source clause: 2.1. "
    "Citation: clause 2.1. Confidence: 0.90. Grounding: chars 1042-1110."
)


def _edge_source(doc_id: str) -> str:
    return json.dumps(
        {"type": "relationship_edge", "doc_id": doc_id, "clause_id": "2.1", "confidence": 0.9}
    )


def _episode(
    episode_id: str,
    *,
    group_id: str,
    content: str,
    source_description: str,
) -> Any:
    from types import SimpleNamespace

    return SimpleNamespace(
        uuid=episode_id,
        name=episode_id,
        content=content,
        source_description=source_description,
        group_id=group_id,
        score=0.9,
        created_at=None,
    )


class _FakeGraphiti:
    """Fake client: search (chunk filter) + driver-backed episode reads."""

    def __init__(self, episodes: list[Any]) -> None:
        self._episodes = episodes
        self.graph_driver: object = object()
        self.calls: list[dict[str, Any]] = []

    async def search(self, *, query: str, group_ids: list[str], num_results: int) -> list[Any]:
        self.calls.append({"surface": "search", "group_ids": list(group_ids), "limit": num_results})
        return [episode for episode in self._episodes if episode.group_id in group_ids][
            :num_results
        ]


def _install_episode_source(monkeypatch: pytest.MonkeyPatch, episodes: list[Any]) -> _FakeGraphiti:
    """Patch the episode read surface; the reader reads episodes by group."""
    graphiti = _FakeGraphiti(episodes)

    async def fake_read(_driver: object, *, group_ids: list[str], limit: int) -> list[Any]:
        graphiti.calls.append({"surface": "episodes", "group_ids": list(group_ids), "limit": limit})
        return [episode for episode in episodes if episode.group_id in group_ids][:limit]

    monkeypatch.setattr(graphiti_client_module, "_read_episodes_by_group", fake_read)
    return graphiti


class _StubRepository:
    """Branch + hydration doubles returning exactly one hydrated chunk."""

    async def bm25_search(self, **_kwargs: object) -> Success[list[dict[str, Any]]]:
        return Success([{"chunk_id": _CHUNK_ID, "score": 2.0}])

    async def vector_search(self, **_kwargs: object) -> Success[list[dict[str, Any]]]:
        return Success([])

    async def trigram_search(self, **_kwargs: object) -> Success[list[dict[str, Any]]]:
        return Success([])

    async def fetch_chunks_by_ids(
        self, chunk_ids: list[str], **_kwargs: object
    ) -> Success[dict[str, dict[str, Any]]]:
        if _CHUNK_ID not in chunk_ids:
            return Success({})
        return Success(
            {
                _CHUNK_ID: {
                    "document_id": _DOC_ID,
                    "title": "Services Agreement",
                    "content": "The Employer shall pay the monthly charge. Acme Corp signs.",
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
            }
        )

    async def resolve_scope_versions(
        self, *, tenant_id: str, document_ids: tuple[str, ...]
    ) -> tuple[tuple[str, int, str | None], ...]:
        if tenant_id != _TENANT_A or _DOC_ID not in document_ids:
            return ()
        return ((_DOC_ID, 1, None),)

    async def resolve_scope_contracts(
        self, *, tenant_id: str, document_ids: tuple[str, ...]
    ) -> tuple[tuple[str, int, tuple[int | None, ...], bool], ...]:
        if tenant_id != _TENANT_A or _DOC_ID not in document_ids:
            return ()
        return ((_DOC_ID, 1, (0,), False),)


class _Handler:
    def __init__(self, schema: type, prompts: list[str]) -> None:
        self._schema = schema
        self._prompts = prompts

    async def ainvoke(self, messages: list[Any]) -> Any:
        if self._schema is QueryPlan:
            return QueryPlan(rewritten_query="payment obligations")
        if self._schema is ContextGrade:
            return ContextGrade(sufficient=True, missing_aspects=[])
        if self._schema is GeneratedAnswer:
            self._prompts.append(str(messages[-1].content))
            return GeneratedAnswer(
                answer="Acme Corp must pay the monthly charge.",
                citations=[
                    Citation(
                        chunk_id=_CHUNK_ID,
                        clause_type="payment",
                        claim="an obligation to pay exists",
                    )
                ],
                confidence="high",
            )
        raise AssertionError(f"unexpected structured schema {self._schema}")  # noqa: EM102


class _FakeStructuredLLM:
    def __init__(self) -> None:
        self.generator_prompts: list[str] = []

    def with_structured_output(self, schema: type) -> _Handler:
        return _Handler(schema, self.generator_prompts)


class _PassThroughReranker:
    async def rerank(
        self, _query: str, chunks: list[RetrievedChunk], *, limit: int
    ) -> list[RetrievedChunk]:
        return chunks[:limit]


def _tenant_group(tenant_id: str, doc_id: str) -> str:
    from app.retrieval.graphiti.grouping import build_graphiti_group_id

    return build_graphiti_group_id(
        tenant_id=tenant_id, document_id=doc_id, document_version=1, contract_index=0
    )


def _service(graphiti: Any, llm: _FakeStructuredLLM | None = None) -> DocumentQueryService:
    llm = llm or _FakeStructuredLLM()
    return DocumentQueryService(
        _StubRepository(),  # ty: ignore[invalid-argument-type]
        lambda: llm,
        None,
        graphiti,
    )


def _payload() -> UnifiedAskRequest:
    return UnifiedAskRequest(query="payment obligations", bypass_cache=True)


async def _fake_embed(*_args: object, **_kwargs: object) -> list[float]:
    return [0.1, 0.2, 0.3]


def _pass_through_reranker() -> _PassThroughReranker:
    return _PassThroughReranker()


@pytest.fixture
def _wired(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(documents_module, "embed_text", _fake_embed)
    monkeypatch.setattr(documents_module, "get_shared_reranker", _pass_through_reranker)


@pytest.mark.usefixtures("_wired")
async def test_ask_cites_edges_with_server_derived_provenance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    graphiti = _install_episode_source(
        monkeypatch,
        [
            _episode(
                "edge-1",
                group_id=_tenant_group(_TENANT_A, _DOC_ID),
                content=_EDGE_BODY,
                source_description=_edge_source(_DOC_ID),
            )
        ],
    )
    service = _service(graphiti)

    result = await service.ask(
        user_id="user-1",
        tenant_id=_TENANT_A,
        allowed_doc_ids=[_DOC_ID],
        payload=_payload(),
        require_graphiti_verified=False,
    )

    assert isinstance(result, Success)
    response = result.unwrap()
    edge_citations = [
        citation for citation in response.citations if "Grounded by" in citation.claim
    ]
    assert len(edge_citations) == 1
    citation = edge_citations[0]
    assert citation.chunk_id == _CHUNK_ID
    assert "OBLIGES edge 'Acme Corp' -> 'GlobalTech Ltd'" in citation.claim
    assert "chars 1042-1110" in citation.claim
    assert citation.clause_type == "payment"
    # Provenance comes from the hydrated row, never from the edge text.
    assert citation.document_id == _DOC_ID
    assert citation.document_version == 1
    assert citation.contract_index == 0


@pytest.mark.usefixtures("_wired")
async def test_ask_feeds_edge_grounding_to_the_generator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    llm = _FakeStructuredLLM()
    graphiti = _install_episode_source(
        monkeypatch,
        [
            _episode(
                "edge-1",
                group_id=_tenant_group(_TENANT_A, _DOC_ID),
                content=_EDGE_BODY,
                source_description=_edge_source(_DOC_ID),
            )
        ],
    )
    service = _service(graphiti, llm)

    result = await service.ask(
        user_id="user-1",
        tenant_id=_TENANT_A,
        allowed_doc_ids=[_DOC_ID],
        payload=_payload(),
        require_graphiti_verified=False,
    )

    assert isinstance(result, Success)
    prompts = llm.generator_prompts
    assert len(prompts) == 1
    assert "grounding_edges" in prompts[0]
    assert "chars 1042-1110" in prompts[0]


@pytest.mark.usefixtures("_wired")
async def test_ask_cannot_expand_through_another_tenant_edges(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tenant_a_group = _tenant_group(_TENANT_A, _DOC_ID)
    tenant_b_group = _tenant_group(_TENANT_B, "doc-b")
    graphiti = _install_episode_source(
        monkeypatch,
        [
            _episode(
                "edge-a",
                group_id=tenant_a_group,
                content=_EDGE_BODY,
                source_description=_edge_source(_DOC_ID),
            ),
            _episode(
                "edge-b",
                group_id=tenant_b_group,
                content="BetaCorp obliges Gamma LLC. Source clause: 7.2. "
                "Citation: clause 7.2. Confidence: 0.90. Grounding: chars 5-90.",
                source_description=_edge_source("doc-b"),
            ),
        ],
    )
    service = _service(graphiti)

    result = await service.ask(
        user_id="user-1",
        tenant_id=_TENANT_A,
        allowed_doc_ids=[_DOC_ID],
        payload=_payload(),
        require_graphiti_verified=False,
    )

    assert isinstance(result, Success)
    response = result.unwrap()
    assert all("BetaCorp" not in citation.claim for citation in response.citations)
    assert all("Gamma" not in citation.claim for citation in response.citations)
    # The service never even asked for tenant B's graph partition.
    assert graphiti.calls
    for call in graphiti.calls:
        assert tenant_b_group not in call["group_ids"]


@pytest.mark.usefixtures("_wired")
async def test_ask_without_edges_keeps_the_pre_edge_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A graphiti holding only non-edge episodes must behave exactly like no
    # graphiti at all: no grounding_edges payload key, no extra citations.
    bare_graphiti = _install_episode_source(
        monkeypatch,
        [
            _episode(
                "clause-1",
                group_id=_tenant_group(_TENANT_A, _DOC_ID),
                content="The Employer shall pay within fourteen days.",
                source_description=json.dumps({"type": "clause_episode"}),
            )
        ],
    )
    bare_llm = _FakeStructuredLLM()
    bare_service = _service(bare_graphiti, bare_llm)
    bare_response = await bare_service.ask(
        user_id="user-1",
        tenant_id=_TENANT_A,
        allowed_doc_ids=[_DOC_ID],
        payload=_payload(),
        require_graphiti_verified=False,
    )

    none_llm = _FakeStructuredLLM()
    none_service = _service(None, none_llm)
    none_response = await none_service.ask(
        user_id="user-1",
        tenant_id=_TENANT_A,
        allowed_doc_ids=[_DOC_ID],
        payload=_payload(),
        require_graphiti_verified=False,
    )

    assert isinstance(bare_response, Success)
    assert isinstance(none_response, Success)
    assert [citation.claim for citation in bare_response.unwrap().citations] == [
        citation.claim for citation in none_response.unwrap().citations
    ]
    assert len(bare_llm.generator_prompts) == len(none_llm.generator_prompts) == 1
    assert "grounding_edges" not in bare_llm.generator_prompts[0]
    assert "grounding_edges" not in none_llm.generator_prompts[0]


async def test_edge_grounding_helper_fails_closed_without_explicit_scope(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    graphiti = _install_episode_source(
        monkeypatch,
        [
            _episode(
                "edge-1",
                group_id=_tenant_group(_TENANT_A, _DOC_ID),
                content=_EDGE_BODY,
                source_description=_edge_source(_DOC_ID),
            )
        ],
    )

    result = await documents_module._edge_grounding_for_reranked(
        graphiti=graphiti,
        query="payment",
        tenant_id=_TENANT_A,
        scope_ids=(),
        scope_versions=(("doc-a", 1, None),),
        scope_contracts=(("doc-a", 1, (0,), False),),
        contract_index=None,
        reranked=[
            RetrievedChunk(
                chunk_id=_CHUNK_ID,
                chunk_text="Acme Corp signs.",
                preamble="",
                clause_type="payment",
                parent_doc_id=_DOC_ID,
                metadata_={},
                custom_metadata={},
                score=1.0,
            )
        ],
    )

    assert result == []
    assert graphiti.calls == []
