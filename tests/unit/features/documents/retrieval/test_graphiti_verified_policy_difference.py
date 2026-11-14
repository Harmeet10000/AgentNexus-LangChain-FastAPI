"""The deliberate ``graphiti_verified`` difference (harden task 5.11).

``/legal/ask`` keeps the ``graph_indexed_only`` policy: its retrieval leg
filters chunks on the persisted ``graphiti_verified`` column, so a chunk that
was never graph-indexed cannot enter its served evidence. Agent Saul's
canonical-chunk path runs under ``mechanically_verified`` (the qna engine pins
``require_graphiti_verified=False`` in its entry mapping, and the evidence
tools default ``require_graph_indexed=False``): a canonical but non-indexed
chunk is still SERVED, with the missing graph verification disclosed as
coverage (``graph_unavailable_documents``, ``graph_indexed`` provenance).

This suite asserts BOTH behaviours side by side against the same two-chunk
corpus through the same compiled retrieval graph, proving the difference is
deliberate rather than drift. The repository double below mirrors exactly one
predicate — the real repository's SQL filter
(``document_repository._FILTER_SQL``:
``CAST(:require_graphiti_verified AS boolean) IS FALSE OR c.graphiti_verified IS TRUE``),
which the mechanism pin in
``test_ask_inline_graph_parity.test_graphiti_verified_filter_threads_into_hybrid_params``
keeps bound — so the exclusion asserted here is the predicate the real
database enforces.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from returns.result import Success
from tests.unit.features.documents.retrieval.test_ask_inline_graph_parity import (
    _CHUNK_ID,
    _DOC_ID,
    _TENANT_A,
    _graph_to_ask_response,
    _payload,
    _RecordingRepository,
    _run_graph,
    _ScriptedAskLLM,
    _service,
    _sufficient,
    _wired,  # noqa: F401 — re-exported so pytest collects the fixture here
)

from app.features.documents.adapters.persistence.postgres.document_repository import _FILTER_SQL
from app.features.documents.retrieval.graph.state import Citation, GeneratedAnswer

if TYPE_CHECKING:
    from typing import Any

pytestmark = pytest.mark.unit

# The canonical-only chunk is the parity suite's main row (graphiti_verified
# is hard False there); the graph-indexed chunk is the extra row.
_CANONICAL_DOC = _DOC_ID
_GRAPH_DOC = "doc-b"
_GRAPH_CHUNK_ID = "chunk-graph"
_CANONICAL_BODY = "The Employer shall pay the monthly charge. Acme Corp signs."
_GRAPH_BODY = "The parties recorded the payment schedule in Graphiti."


class _TwoChunkPolicyRepository(_RecordingRepository):
    """The parity repository double plus the ask policy predicate.

    Mirrors the real repository's ``require_graphiti_verified`` SQL predicate
    in Python so a corpus containing BOTH a canonical-only and a
    graph-indexed chunk can be run through the graph under either policy.
    Every applied filter bind is still recorded for the bind assertions.
    """

    def __init__(self) -> None:
        super().__init__(
            extra_row={
                "chunk_id": _GRAPH_CHUNK_ID,
                "document_id": _GRAPH_DOC,
                "content": _GRAPH_BODY,
                "clause_type": "payment",
                "graphiti_verified": True,
            }
        )
        self.fetched_chunk_ids: list[list[str]] = []

    async def bm25_search(self, **kwargs: Any) -> Success[list[dict[str, Any]]]:
        self.seen_filter_params.append(dict(kwargs.get("filter_params", {})))
        require = bool(kwargs.get("filter_params", {}).get("require_graphiti_verified", False))
        results = [
            {"chunk_id": _CHUNK_ID, "score": 2.0},
            {"chunk_id": _GRAPH_CHUNK_ID, "score": 1.5},
        ]
        if require:
            results = [row for row in results if row["chunk_id"] == _GRAPH_CHUNK_ID]
        return Success(results)

    async def fetch_chunks_by_ids(
        self, chunk_ids: list[str], **kwargs: Any
    ) -> Success[dict[str, dict[str, Any]]]:
        self.fetched_chunk_ids.append([str(chunk_id) for chunk_id in chunk_ids])
        return await super().fetch_chunks_by_ids(chunk_ids, **kwargs)

    async def resolve_scope_versions(
        self, *, tenant_id: str, document_ids: tuple[str, ...]
    ) -> tuple[tuple[str, int, str | None], ...]:
        if tenant_id != _TENANT_A:
            return ()
        return tuple(
            (doc_id, 1, None) for doc_id in (_CANONICAL_DOC, _GRAPH_DOC) if doc_id in document_ids
        )

    async def resolve_scope_contracts(
        self, *, tenant_id: str, document_ids: tuple[str, ...]
    ) -> tuple[tuple[str, int, tuple[int | None, ...], bool], ...]:
        if tenant_id != _TENANT_A:
            return ()
        return tuple(
            (doc_id, 1, (0,), False)
            for doc_id in (_CANONICAL_DOC, _GRAPH_DOC)
            if doc_id in document_ids
        )


class _CitingAskLLM(_ScriptedAskLLM):
    """The parity scripted ask LLM whose generator cites one fixed chunk."""

    def __init__(self, cited_chunk_id: str) -> None:
        super().__init__([_sufficient()])
        self._cited_chunk_id = cited_chunk_id

    async def _invoke(self, schema: type, messages: list[Any]) -> Any:
        if schema is GeneratedAnswer:
            self.generator_prompts.append(str(messages[-1].content))
            return GeneratedAnswer(
                answer="The Employer must pay the monthly charge.",
                citations=[
                    Citation(
                        chunk_id=self._cited_chunk_id,
                        clause_type="payment",
                        claim="an obligation to pay exists",
                    )
                ],
                confidence="high",
            )
        return await super()._invoke(schema, messages)


def _real_policy_predicate_is_pinned() -> None:
    """Mechanism pin: the real SQL enforces the policy the double mirrors."""
    assert "CAST(:require_graphiti_verified AS boolean) IS FALSE" in _FILTER_SQL
    assert "c.graphiti_verified IS TRUE" in _FILTER_SQL


@pytest.mark.usefixtures("_wired")
async def test_ask_policy_excludes_non_graph_indexed_chunks() -> None:
    """Side 1 — /legal/ask (require_graphiti_verified=True): exclusion."""
    _real_policy_predicate_is_pinned()
    repo = _TwoChunkPolicyRepository()
    result = await _service(_CitingAskLLM(_GRAPH_CHUNK_ID), repo).ask(
        user_id="user-1",
        tenant_id=_TENANT_A,
        allowed_doc_ids=[_CANONICAL_DOC, _GRAPH_DOC],
        payload=_payload(),
        require_graphiti_verified=True,
    )

    assert isinstance(result, Success)
    response = result.unwrap()
    # The policy is served and the canonical-only chunk never entered the
    # pipeline: not ranked, not hydrated, not cited, not covered.
    assert response.evidence_policy == "graph_indexed_only"
    assert [citation.chunk_id for citation in response.citations] == [_GRAPH_CHUNK_ID]
    assert _CHUNK_ID not in [citation.chunk_id for citation in response.citations]
    assert all(_CHUNK_ID not in fetched for fetched in repo.fetched_chunk_ids)
    assert repo.seen_filter_params
    assert repo.seen_filter_params[0]["require_graphiti_verified"] is True
    assert response.coverage.graph_indexed_documents == 1
    assert response.coverage.graph_unavailable_documents == 0


@pytest.mark.usefixtures("_wired")
async def test_canonical_path_serves_non_graph_indexed_chunks_with_disclosure() -> None:
    """Side 2 — Saul's canonical-chunk path (require=False): serve + disclose."""
    _real_policy_predicate_is_pinned()
    repo = _TwoChunkPolicyRepository()
    # The entry mapping Saul's qna engine constructs: identical to the ask
    # state except `require_graphiti_verified=False` (agent_saul graph/nodes).
    result = await _run_graph(
        _CitingAskLLM(_CHUNK_ID),
        repo,
        state_overrides={
            "allowed_doc_ids": [_CANONICAL_DOC, _GRAPH_DOC],
            "document_versions": [(_CANONICAL_DOC, 1, None), (_GRAPH_DOC, 1, None)],
            "document_contracts": [
                (_CANONICAL_DOC, 1, (0,), False),
                (_GRAPH_DOC, 1, (0,), False),
            ],
            "require_graphiti_verified": False,
        },
    )

    response = _graph_to_ask_response(result)
    assert response.evidence_policy == "mechanically_verified"
    # The canonical-only chunk IS served evidence under this path.
    served = [citation.chunk_id for citation in response.citations]
    assert served == [_CHUNK_ID]
    assert _CHUNK_ID in [chunk for fetched in repo.fetched_chunk_ids for chunk in fetched]
    # ... and the missing graph verification is disclosed as coverage, not
    # hidden: the graph-indexed document counts as indexed, the canonical-only
    # document as graph-unavailable, in the same truthful envelope.
    assert response.coverage.graph_indexed_documents == 1
    assert response.coverage.graph_unavailable_documents == 1
    # The same corpus was eligible: both documents were requested.
    assert response.coverage.documents_requested == 2


@pytest.mark.usefixtures("_wired")
async def test_policy_difference_is_side_by_side_on_identical_corpus() -> None:
    """Both sides over ONE corpus in one test: the difference is deliberate."""
    _real_policy_predicate_is_pinned()

    async def _run(require_graphiti_verified: bool, cited_chunk_id: str) -> Any:
        repo = _TwoChunkPolicyRepository()
        result = await _run_graph(
            _CitingAskLLM(cited_chunk_id),
            repo,
            state_overrides={
                "allowed_doc_ids": [_CANONICAL_DOC, _GRAPH_DOC],
                "document_versions": [(_CANONICAL_DOC, 1, None), (_GRAPH_DOC, 1, None)],
                "document_contracts": [
                    (_CANONICAL_DOC, 1, (0,), False),
                    (_GRAPH_DOC, 1, (0,), False),
                ],
                "require_graphiti_verified": require_graphiti_verified,
            },
        )
        response = _graph_to_ask_response(result)
        return response, repo

    ask_response, ask_repo = await _run(True, _GRAPH_CHUNK_ID)
    canonical_response, canonical_repo = await _run(False, _CHUNK_ID)

    # Identical corpus, identical query, opposite served-evidence outcomes:
    assert ask_response.evidence_policy == "graph_indexed_only"
    assert canonical_response.evidence_policy == "mechanically_verified"
    assert ask_response.citations[0].chunk_id == _GRAPH_CHUNK_ID
    assert canonical_response.citations[0].chunk_id == _CHUNK_ID
    assert all(_CHUNK_ID not in fetched for fetched in ask_repo.fetched_chunk_ids)
    assert any(_CHUNK_ID in fetched for fetched in canonical_repo.fetched_chunk_ids)
    # Graphiti is never legal truth on either side: the canonical-only chunk's
    # absence of graph verification is disclosed, never asserted away.
    assert canonical_response.coverage.graph_unavailable_documents == 1
    assert ask_response.coverage.graph_unavailable_documents == 0
