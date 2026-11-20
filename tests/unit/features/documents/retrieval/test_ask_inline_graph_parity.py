"""The ask parity suite (mount-retrieval-graph task 2.1) — the cutover gate.

The same queries run through the served inline contract
(``DocumentQueryService.ask`` / ``_ask``) and the compiled retrieval graph
must produce identical response shapes, citation provenance, fallback
behaviour, and cache-key derivation. Every inline grade-loop / edge-expansion
test case that can be replayed is replayed here; the suite stays as a
permanent regression gate that fails on behavioural drift on either side.

Each assertion names the behaviour it pins, so the deliberate-drift drill
(task 2.2) fails naming the divergent behaviour.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest
from returns.result import Failure, Success

import app.features.documents.application.queries.documents as documents_module
from app.ai.generation import GeneratedStructured
from app.features.documents.api.http.dto import (
    EvidenceCoverage,
    LegalCitationResponse,
    QualityWarningDTO,
    UnifiedAskRequest,
    UnifiedAskResponse,
)
from app.features.documents.application.queries.documents import (
    _FALLBACK_ANSWER,
    DocumentQueryService,
    _build_answer_cache_key,
)
from app.features.documents.retrieval.graph import (
    RetrievalEngine,
    build_retrieval_graph,
)
from app.features.documents.retrieval.graph.nodes import (
    _answer_cache_key as _graph_answer_cache_key,
)
from app.features.documents.retrieval.graph.state import (
    Citation,
    ContextGrade,
    GeneratedAnswer,
    QueryPlan,
    RetrievalGraphContext,
    SourceCriteria,
)
from app.retrieval.graphiti import client as graphiti_client_module

if TYPE_CHECKING:
    from typing import Any

    from app.features.documents.retrieval.graph.state import GroundedCitation

_CHUNK_ID = "chunk-1"
_DOC_ID = "doc-a"
_TENANT_A = "tenant-a"
_TENANT_B = "tenant-b"
_ORIGINAL_QUERY = "payment obligations"
_REWRITE = "rewritten payment clause"

_EDGE_BODY = (
    "Acme Corp obliges GlobalTech Ltd. Source clause: 2.1. "
    "Citation: clause 2.1. Confidence: 0.90. Grounding: chars 1042-1110."
)


# --------------------------------------------------------------------------
# Shared hermetic doubles (recorded inline-path fakes)
# --------------------------------------------------------------------------


class _ScriptedAskLLM:
    """Structured-output double scripted per seam, recording every prompt.

    Shared by both paths: the inline ask loop and the compiled graph consume
    the same double, so a divergence cannot come from the fake.
    """

    def __init__(self, grades: list[ContextGrade]) -> None:
        self._grades = list(grades)
        self.analyzer_prompts: list[str] = []
        self.grader_prompts: list[str] = []
        self.generator_prompts: list[str] = []

    def with_structured_output(self, schema: type) -> Any:
        return _ScriptedHandler(schema, self)

    async def _invoke(self, schema: type, messages: list[Any]) -> Any:
        prompt = str(messages[-1].content)
        if schema is QueryPlan:
            self.analyzer_prompts.append(prompt)
            return QueryPlan(rewritten_query=_REWRITE if _REWRITE in prompt else _ORIGINAL_QUERY)
        if schema is SourceCriteria:
            return SourceCriteria()
        if schema is ContextGrade:
            self.grader_prompts.append(prompt)
            return self._grades[min(len(self.grader_prompts) - 1, len(self._grades) - 1)]
        if schema is GeneratedAnswer:
            self.generator_prompts.append(prompt)
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
        raise AssertionError(f"unexpected structured schema {schema}")  # noqa: EM102


class _ScriptedHandler:
    def __init__(self, schema: type, llm: _ScriptedAskLLM) -> None:
        self._schema = schema
        self._llm = llm

    async def ainvoke(self, messages: list[Any]) -> Any:
        return await self._llm._invoke(self._schema, messages)


class _RecordingRepository:
    """Branch + hydration doubles returning exactly one hydrated chunk.

    Records every branch ``filter_params`` it is called with so the parity
    suite can compare the retrieval filter derivations directly.
    """

    def __init__(
        self,
        *,
        span: tuple[int, int] = (1000, 1200),
        extra_row: dict[str, Any] | None = None,
        quality_warnings: list[Any] | None = None,
        hydrated_chunk_ids: list[str] | None = None,
        resolve_contracts: bool = True,
        contract_mismatch: bool = False,
    ) -> None:
        self.span = span
        self.extra_row = extra_row
        self.quality_warnings = quality_warnings or []
        self.hydrated_chunk_ids = hydrated_chunk_ids
        self.resolve_contracts = resolve_contracts
        self.contract_mismatch = contract_mismatch
        self.seen_filter_params: list[dict[str, Any]] = []

    async def bm25_search(self, **kwargs: Any) -> Success[list[dict[str, Any]]]:
        self.seen_filter_params.append(dict(kwargs.get("filter_params", {})))
        results = [{"chunk_id": _CHUNK_ID, "score": 2.0}]
        if self.extra_row is not None:
            results.append({"chunk_id": str(self.extra_row["chunk_id"]), "score": 1.5})
        return Success(results)

    async def vector_search(self, **_kwargs: object) -> Success[list[dict[str, Any]]]:
        return Success([])

    async def trigram_search(self, **_kwargs: object) -> Success[list[dict[str, Any]]]:
        return Success([])

    def _rows(self, chunk_ids: list[str]) -> dict[str, dict[str, Any]]:
        rows: dict[str, dict[str, Any]] = {
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
                "chunk_metadata": {
                    "source_start_char": self.span[0],
                    "source_end_char": self.span[1],
                },
                "quality_warnings": self.quality_warnings,
                "graphiti_verified": False,
                "graph_indexed_at": None,
            }
        }
        if self.extra_row is not None:
            extra = dict(self.extra_row)
            rows[str(extra["chunk_id"])] = {
                "document_id": extra.get("document_id", _DOC_ID),
                "title": "Services Agreement",
                "content": str(extra.get("content", "")),
                "preamble": "",
                "document_version": 1,
                "contract_index": 0,
                "pipeline_version": "canonical",
                "chunk_index": 1,
                "chunk_kind": "legal_contract",
                "clause_type": str(extra.get("clause_type", "other")),
                "chunk_metadata": dict(extra.get("metadata", {})),
                "quality_warnings": extra.get("quality_warnings", []),
                "graphiti_verified": bool(extra.get("graphiti_verified")),
                "graph_indexed_at": None,
            }
        return {key: row for key, row in rows.items() if key in chunk_ids}

    async def fetch_chunks_by_ids(
        self, chunk_ids: list[str], **_kwargs: object
    ) -> Success[dict[str, dict[str, Any]]]:
        if self.hydrated_chunk_ids is not None:
            chunk_ids = [cid for cid in chunk_ids if cid in self.hydrated_chunk_ids]
        return Success(self._rows(chunk_ids))

    async def resolve_scope_versions(
        self, *, tenant_id: str, document_ids: tuple[str, ...]
    ) -> tuple[tuple[str, int, str | None], ...]:
        if tenant_id != _TENANT_A or _DOC_ID not in document_ids:
            return ()
        return ((_DOC_ID, 1, None),)

    async def resolve_scope_contracts(
        self, *, tenant_id: str, document_ids: tuple[str, ...]
    ) -> tuple[tuple[str, int, tuple[int | None, ...], bool], ...]:
        if not self.resolve_contracts or tenant_id != _TENANT_A or _DOC_ID not in document_ids:
            return ()
        if self.contract_mismatch:
            return ((_DOC_ID, 2, (0,), False),)
        return ((_DOC_ID, 1, (0,), False),)


class _PassThroughReranker:
    async def rerank(self, _query: str, chunks: list[Any], *, limit: int) -> list[Any]:
        return chunks[:limit]


class _FakeRedis:
    """Minimal async redis double that records every key it touches."""

    def __init__(self) -> None:
        self.values: dict[str, str] = {}
        self.get_keys: list[str] = []
        self.setex_keys: list[str] = []

    async def get(self, key: str) -> str | None:
        self.get_keys.append(key)
        return self.values.get(key)

    async def setex(self, key: str, _time: int, value: str) -> None:
        self.setex_keys.append(key)
        self.values[key] = value


async def _fake_embed(*_args: object, **_kwargs: object) -> list[float]:
    return [0.1, 0.2, 0.3]


def _tenant_group(tenant_id: str, doc_id: str) -> str:
    from app.retrieval.graphiti.grouping import build_graphiti_group_id

    return build_graphiti_group_id(
        tenant_id=tenant_id, document_id=doc_id, document_version=1, contract_index=0
    )


def _edge_source(doc_id: str) -> str:
    return json.dumps(
        {"type": "relationship_edge", "doc_id": doc_id, "clause_id": "2.1", "confidence": 0.9}
    )


def _episode(episode_id: str, *, group_id: str, content: str, source_description: str) -> Any:
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


def _install_episode_source(monkeypatch: pytest.MonkeyPatch, episodes: list[Any]) -> Any:
    """Patch the episode read surface both edge readers use; record calls."""
    from types import SimpleNamespace

    graphiti = SimpleNamespace(calls=[], driver=object())

    async def fake_search(*, query: str, group_ids: list[str], num_results: int) -> list[Any]:
        graphiti.calls.append(
            {"surface": "search", "group_ids": list(group_ids), "limit": num_results}
        )
        return []

    graphiti.search = fake_search

    async def fake_read(
        _driver: object, *, group_ids: list[str], limit: int, uuid_cursor: str | None = None
    ) -> list[Any]:
        graphiti.calls.append({"surface": "episodes", "group_ids": list(group_ids), "limit": limit})
        return [episode for episode in episodes if episode.group_id in group_ids][:limit]

    monkeypatch.setattr(graphiti_client_module, "_read_episodes_by_group", fake_read)
    return graphiti


# --------------------------------------------------------------------------
# Path runners: the inline contract and the compiled graph
# --------------------------------------------------------------------------


def _engine(
    llm: Any,
    *,
    graphiti: Any = None,
    redis: Any = None,
    grader: Any = None,
) -> RetrievalEngine:
    """The compile-once engine the cutover wires into the service (task 4.1)."""
    return RetrievalEngine(
        session_factory=None,  # ty: ignore[invalid-argument-type] — the service path never opens a session
        redis=redis,
        graphiti=graphiti,
        llm_factory=lambda: llm,
        grader=grader,
    )


def _service(
    llm: Any,
    repo: Any,
    *,
    graphiti: Any = None,
    redis: Any = None,
    grader: Any = None,
) -> DocumentQueryService:
    return DocumentQueryService(
        repo,  # ty: ignore[invalid-argument-type]
        lambda: llm,
        redis,
        graphiti,
        grader=grader,
        retrieval_engine=_engine(llm, graphiti=graphiti, redis=redis, grader=grader),
    )


def _payload(bypass_cache: bool = True) -> UnifiedAskRequest:
    return UnifiedAskRequest(query=_ORIGINAL_QUERY, bypass_cache=bypass_cache)


def _initial_state(**overrides: Any) -> dict[str, Any]:
    """The graph's initial state, as the served wrapper constructs it."""
    state: dict[str, Any] = {
        "user_id": "user-1",
        "tenant_id": _TENANT_A,
        "allowed_doc_ids": [_DOC_ID],
        "document_versions": [(_DOC_ID, 1, None)],
        "document_contracts": [(_DOC_ID, 1, (0,), False)],
        "scope_unavailable_documents": [],
        "scope_bound": True,
        "query": _ORIGINAL_QUERY,
        "doc_ids_filter": [],
        "contract_index": None,
        "jurisdiction": None,
        "contract_type": None,
        "clause_type": None,
        "require_graphiti_verified": False,
        "messages": [],
        "iteration_count": 0,
    }
    state.update(overrides)
    return state


async def _run_graph(
    llm: Any,
    repo: Any,
    *,
    graphiti: Any = None,
    redis: Any = None,
    grader: Any = None,
    context: Any = None,
    state_overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    graph = build_retrieval_graph()
    if context is None:
        context = RetrievalGraphContext(
            llm=llm,
            repo=repo,  # ty: ignore[invalid-argument-type]
            redis=redis,
            graphiti=graphiti,
            reranker=_PassThroughReranker(),
            grader=grader,
        )
    return await graph.ainvoke(_initial_state(**(state_overrides or {})), context=context)


def _grounded_citation_to_response(citation: GroundedCitation) -> LegalCitationResponse:
    return LegalCitationResponse(
        chunk_id=citation.chunk_id,
        clause_type=citation.clause_type,
        claim=citation.claim,
        document_id=citation.document_id,
        document_version=citation.document_version,
        contract_index=citation.contract_index,
        pipeline_version=citation.pipeline_version,
        page_no=citation.page_no,
    )


def _graph_to_ask_response(result: dict[str, Any]) -> UnifiedAskResponse:
    """The service-boundary mapping the cutover must perform (phase 4).

    Maps the graph's result state onto ``UnifiedAskResponse`` exactly as the
    inline wrapper does: citations through the hydrated-row provenance
    discipline, the truthful coverage envelope, the merged warnings, and the
    derived evidence policy.
    """
    answer: GeneratedAnswer = result["generated_answer"]
    coverage = result.get("coverage")
    return UnifiedAskResponse(
        answer=answer.answer,
        citations=[
            _grounded_citation_to_response(citation) for citation in result["resolved_citations"]
        ],
        confidence=answer.confidence,
        evidence_policy=result["evidence_policy"],
        coverage=EvidenceCoverage(
            documents_requested=coverage.documents_requested,
            documents_ready=coverage.documents_ready,
            graph_indexed_documents=coverage.graph_indexed_documents,
            graph_unavailable_documents=coverage.graph_unavailable_documents,
        )
        if coverage is not None
        else EvidenceCoverage(),
        warnings=[
            QualityWarningDTO(
                stage=warning.stage,
                code=warning.code,
                message=warning.message,
                severity=warning.severity,
            )
            for warning in result.get("quality_warnings", [])
        ],
        cache_hit=bool(result.get("cache_hit")),
    )


def _assert_identical_response_shape(
    inline: UnifiedAskResponse, graph: UnifiedAskResponse, behaviour: str
) -> None:
    """Assert the two ask responses are byte-identical, naming the behaviour."""
    assert inline.answer == graph.answer, f"divergent behaviour: {behaviour} (answer text)"
    assert inline.confidence == graph.confidence, f"divergent behaviour: {behaviour} (confidence)"
    assert inline.evidence_policy == graph.evidence_policy, (
        f"divergent behaviour: {behaviour} (evidence policy)"
    )
    assert inline.cache_hit == graph.cache_hit, f"divergent behaviour: {behaviour} (cache_hit)"
    assert [c.model_dump() for c in inline.citations] == [
        c.model_dump() for c in graph.citations
    ], f"divergent behaviour: {behaviour} (citation provenance)"
    assert inline.coverage.model_dump() == graph.coverage.model_dump(), (
        f"divergent behaviour: {behaviour} (coverage envelope)"
    )
    assert [w.model_dump() for w in inline.warnings] == [w.model_dump() for w in graph.warnings], (
        f"divergent behaviour: {behaviour} (warning envelope)"
    )


def _sufficient() -> ContextGrade:
    return ContextGrade(sufficient=True, missing_aspects=[])


def _insufficient(rewrite: str | None = None) -> ContextGrade:
    return ContextGrade(
        sufficient=False,
        missing_aspects=["payment amounts"],
        rewrite_suggestion=rewrite,
    )


@pytest.fixture
def _wired(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(documents_module, "embed_text", _fake_embed)
    import app.features.documents.retrieval.graph.nodes as graph_nodes

    monkeypatch.setattr(graph_nodes, "embed_text", _fake_embed)
    # The graph resolves the reranker per invocation from the process config;
    # the service leg's context carries no reranker, so patch the resolver.
    monkeypatch.setattr(graph_nodes, "get_configured_reranker", _PassThroughReranker)


# --------------------------------------------------------------------------
# Parity cases (task 2.1's fixed query set)
# --------------------------------------------------------------------------


@pytest.mark.usefixtures("_wired")
async def test_sufficient_grade_produces_identical_responses() -> None:
    llm = _ScriptedAskLLM([_sufficient()])
    repo_inline = _RecordingRepository()
    repo_graph = _RecordingRepository()
    inline_result = await _service(llm, repo_inline).ask(
        user_id="user-1",
        tenant_id=_TENANT_A,
        allowed_doc_ids=[_DOC_ID],
        payload=_payload(),
        require_graphiti_verified=False,
    )
    graph_result = await _run_graph(llm, repo_graph)

    assert isinstance(inline_result, Success)
    inline_response = inline_result.unwrap()
    graph_response = _graph_to_ask_response(graph_result)
    _assert_identical_response_shape(inline_response, graph_response, "sufficient first grade")
    # Coverage is truthful: the one hydrated document counts as ready and, not
    # being graph-indexed, as graph-unavailable in the same response.
    assert graph_response.coverage.documents_requested == 1
    assert graph_response.coverage.documents_ready == 1
    assert graph_response.coverage.graph_indexed_documents == 0
    assert graph_response.coverage.graph_unavailable_documents == 1


@pytest.mark.usefixtures("_wired")
async def test_insufficient_grade_rewrites_then_matches() -> None:
    llm_inline = _ScriptedAskLLM([_insufficient(_REWRITE), _sufficient()])
    llm_graph = _ScriptedAskLLM([_insufficient(_REWRITE), _sufficient()])
    repo_inline = _RecordingRepository()
    repo_graph = _RecordingRepository()
    inline_result = await _service(llm_inline, repo_inline).ask(
        user_id="user-1",
        tenant_id=_TENANT_A,
        allowed_doc_ids=[_DOC_ID],
        payload=_payload(),
        require_graphiti_verified=False,
    )
    graph_result = await _run_graph(llm_graph, repo_graph)

    assert isinstance(inline_result, Success)
    _assert_identical_response_shape(
        inline_result.unwrap(), _graph_to_ask_response(graph_result), "insufficient-then-rewrite"
    )
    # The rewrite arrived as the second iteration's retrieval query on both
    # paths, and the loop stopped at two grader attempts on both.
    for llm in (llm_inline, llm_graph):
        assert len(llm.analyzer_prompts) == 2
        assert _REWRITE in llm.analyzer_prompts[1]
        assert len(llm.grader_prompts) == 2
    assert graph_result["iteration_count"] == 2


@pytest.mark.usefixtures("_wired")
async def test_cap_exhausted_fallback_is_identical() -> None:
    llm_inline = _ScriptedAskLLM([_insufficient(_REWRITE), _insufficient(_REWRITE)])
    llm_graph = _ScriptedAskLLM([_insufficient(_REWRITE), _insufficient(_REWRITE)])
    repo_inline = _RecordingRepository()
    repo_graph = _RecordingRepository()
    inline_result = await _service(llm_inline, repo_inline).ask(
        user_id="user-1",
        tenant_id=_TENANT_A,
        allowed_doc_ids=[_DOC_ID],
        payload=_payload(),
        require_graphiti_verified=False,
    )
    graph_result = await _run_graph(llm_graph, repo_graph)

    assert isinstance(inline_result, Success)
    inline_response = inline_result.unwrap()
    graph_response = _graph_to_ask_response(graph_result)
    _assert_identical_response_shape(inline_response, graph_response, "cap-exhausted fallback")
    # Byte-identical fallback: the recorded inline fallback text, no citations,
    # no generation on either path.
    assert inline_response.answer == _FALLBACK_ANSWER
    assert inline_response.citations == []
    assert inline_response.confidence == "uncertain"
    assert len(llm_inline.generator_prompts) == 0
    assert len(llm_graph.generator_prompts) == 0


@pytest.mark.usefixtures("_wired")
async def test_grader_failure_degrade_is_identical() -> None:
    """The seam-absent degrade (unavailable grade proceeds to generation)."""

    class _FailingGraderLLM(_ScriptedAskLLM):
        async def _invoke(self, schema: type, messages: list[Any]) -> Any:
            if schema is ContextGrade:
                self.grader_prompts.append("failed")
                message = "grader unavailable"
                raise ValueError(message)
            return await super()._invoke(schema, messages)

        def __init__(self) -> None:
            super().__init__([_sufficient()])

    llm = _FailingGraderLLM()
    repo_inline = _RecordingRepository()
    repo_graph = _RecordingRepository()
    inline_result = await _service(llm, repo_inline).ask(
        user_id="user-1",
        tenant_id=_TENANT_A,
        allowed_doc_ids=[_DOC_ID],
        payload=_payload(),
        require_graphiti_verified=False,
    )
    graph_result = await _run_graph(llm, repo_graph)

    assert isinstance(inline_result, Success)
    _assert_identical_response_shape(
        inline_result.unwrap(), _graph_to_ask_response(graph_result), "grader-failure degrade"
    )


@pytest.mark.usefixtures("_wired")
async def test_empty_rerank_fallback_loop_is_identical() -> None:
    """No hydrated chunks: the insufficient feedback loop twice, then fallback."""
    llm_inline = _ScriptedAskLLM([_insufficient(_ORIGINAL_QUERY), _insufficient(_ORIGINAL_QUERY)])
    llm_graph = _ScriptedAskLLM([_insufficient(_ORIGINAL_QUERY), _insufficient(_ORIGINAL_QUERY)])
    repo_inline = _RecordingRepository(hydrated_chunk_ids=[])
    repo_graph = _RecordingRepository(hydrated_chunk_ids=[])
    inline_result = await _service(llm_inline, repo_inline).ask(
        user_id="user-1",
        tenant_id=_TENANT_A,
        allowed_doc_ids=[_DOC_ID],
        payload=_payload(),
        require_graphiti_verified=False,
    )
    graph_result = await _run_graph(llm_graph, repo_graph)

    assert isinstance(inline_result, Success)
    inline_response = inline_result.unwrap()
    graph_response = _graph_to_ask_response(graph_result)
    _assert_identical_response_shape(inline_response, graph_response, "empty-rerank fallback loop")
    assert inline_response.answer == _FALLBACK_ANSWER
    # The no-chunks short-circuit grades without the LLM on both paths; the
    # loop still ran its two bounded iterations.
    assert len(llm_inline.grader_prompts) == 0
    assert len(llm_graph.grader_prompts) == 0
    assert len(llm_inline.analyzer_prompts) == 2
    assert len(llm_graph.analyzer_prompts) == 2
    assert graph_result["iteration_count"] == 2


@pytest.mark.usefixtures("_wired")
@pytest.mark.parametrize("require_graphiti_verified", [False, True])
async def test_empty_scope_returns_the_policy_carrying_fallback_envelope(
    require_graphiti_verified: bool,
) -> None:
    """An explicitly empty authorized scope returns the grounded fallback with
    the request's evidence policy, and no unscoped retrieval runs (task 1.3)."""
    llm = _ScriptedAskLLM([])
    repo_inline = _RecordingRepository()
    repo_graph = _RecordingRepository()
    inline_result = await _service(llm, repo_inline).ask(
        user_id="user-1",
        tenant_id=_TENANT_A,
        allowed_doc_ids=[],
        payload=_payload(),
        require_graphiti_verified=require_graphiti_verified,
    )
    graph_result = await _run_graph(
        llm,
        repo_graph,
        state_overrides={
            "allowed_doc_ids": [],
            "require_graphiti_verified": require_graphiti_verified,
        },
    )

    assert isinstance(inline_result, Success)
    inline_response = inline_result.unwrap()
    graph_response = _graph_to_ask_response(graph_result)
    _assert_identical_response_shape(
        inline_response,
        graph_response,
        f"empty-scope fallback envelope (policy={require_graphiti_verified})",
    )
    expected_policy = "graph_indexed_only" if require_graphiti_verified else "mechanically_verified"
    assert inline_response.evidence_policy == expected_policy
    assert inline_response.answer == _FALLBACK_ANSWER
    assert inline_response.citations == []
    # No search ran on either path: the fallback envelope is not an unscoped read.
    assert repo_inline.seen_filter_params == []
    assert repo_graph.seen_filter_params == []


@pytest.mark.usefixtures("_wired")
async def test_filter_params_are_derived_identically(monkeypatch: pytest.MonkeyPatch) -> None:
    """The hybrid node's filter params match the inline loop's, key for key."""
    llm = _ScriptedAskLLM([_sufficient()])
    repo_inline = _RecordingRepository()
    repo_graph = _RecordingRepository()
    await _service(llm, repo_inline).ask(
        user_id="user-1",
        tenant_id=_TENANT_A,
        allowed_doc_ids=[_DOC_ID],
        payload=_payload(),
        require_graphiti_verified=False,
    )
    await _run_graph(llm, repo_graph)

    assert repo_inline.seen_filter_params
    assert repo_graph.seen_filter_params
    assert repo_inline.seen_filter_params[0] == repo_graph.seen_filter_params[0]


async def test_graphiti_verified_filter_threads_into_hybrid_params() -> None:
    """/legal/ask-shaped requests filter on graph-indexed evidence (task 1.3)."""
    monkeypatch = pytest.MonkeyPatch()
    try:
        monkeypatch.setattr(documents_module, "embed_text", _fake_embed)
        import app.features.documents.retrieval.graph.nodes as graph_nodes

        monkeypatch.setattr(graph_nodes, "embed_text", _fake_embed)
        llm = _ScriptedAskLLM([_sufficient()])
        repo = _RecordingRepository()
        await _run_graph(
            llm,
            repo,
            state_overrides={"require_graphiti_verified": True},
        )
        assert repo.seen_filter_params
        assert repo.seen_filter_params[0]["require_graphiti_verified"] is True
    finally:
        monkeypatch.undo()


@pytest.mark.usefixtures("_wired")
async def test_edge_grounded_citations_carry_identical_provenance(
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
    llm_inline = _ScriptedAskLLM([_sufficient()])
    llm_graph = _ScriptedAskLLM([_sufficient()])
    repo_inline = _RecordingRepository()
    repo_graph = _RecordingRepository()
    inline_result = await _service(llm_inline, repo_inline, graphiti=graphiti).ask(
        user_id="user-1",
        tenant_id=_TENANT_A,
        allowed_doc_ids=[_DOC_ID],
        payload=_payload(),
        require_graphiti_verified=False,
    )
    graph_result = await _run_graph(llm_graph, repo_graph, graphiti=graphiti)

    assert isinstance(inline_result, Success)
    inline_response = inline_result.unwrap()
    graph_response = _graph_to_ask_response(graph_result)
    _assert_identical_response_shape(
        inline_response, graph_response, "edge-grounded citation provenance"
    )
    # The edge citation points at the passage the grounded span falls in, with
    # provenance from the hydrated row, on both paths.
    inline_edge = [c for c in inline_response.citations if "Grounded by" in c.claim]
    assert len(inline_edge) == 1
    assert inline_edge[0].chunk_id == _CHUNK_ID
    assert inline_edge[0].document_id == _DOC_ID
    assert inline_edge[0].document_version == 1
    assert inline_edge[0].contract_index == 0
    assert "OBLIGES edge 'Acme Corp' -> 'GlobalTech Ltd'" in inline_edge[0].claim
    assert "chars 1042-1110" in inline_edge[0].claim
    # The expansion reached the generator on both paths.
    assert "grounding_edges" in llm_inline.generator_prompts[0]
    assert "chars 1042-1110" in llm_inline.generator_prompts[0]
    assert "grounding_edges" in llm_graph.generator_prompts[0]
    assert "chars 1042-1110" in llm_graph.generator_prompts[0]


@pytest.mark.usefixtures("_wired")
async def test_edge_free_query_keeps_the_pre_edge_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A graphiti holding only non-edge episodes behaves exactly like none."""
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
    llm = _ScriptedAskLLM([_sufficient()])
    repo_none = _RecordingRepository()
    repo_bare = _RecordingRepository()
    none_result = await _run_graph(llm, repo_none)
    bare_result = await _run_graph(llm, repo_bare, graphiti=bare_graphiti)

    none_response = _graph_to_ask_response(none_result)
    bare_response = _graph_to_ask_response(bare_result)
    _assert_identical_response_shape(
        none_response, bare_response, "edge-free query keeps the pre-edge contract"
    )
    assert all("Grounded by" not in citation.claim for citation in bare_response.citations)
    assert "grounding_edges" not in llm.generator_prompts[0]


@pytest.mark.usefixtures("_wired")
async def test_edge_expansion_cannot_cross_tenant_boundaries(
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
    llm = _ScriptedAskLLM([_sufficient()])
    repo = _RecordingRepository()
    graph_result = await _run_graph(llm, repo, graphiti=graphiti)

    graph_response = _graph_to_ask_response(graph_result)
    assert all("BetaCorp" not in citation.claim for citation in graph_response.citations)
    assert all("Gamma" not in citation.claim for citation in graph_response.citations)
    # The graph never even asked for tenant B's graph partition.
    assert graphiti.calls
    for call in graphiti.calls:
        assert tenant_b_group not in call["group_ids"]


@pytest.mark.usefixtures("_wired")
async def test_edge_span_resolves_to_the_span_owning_passage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A shared entity name must not re-point one contract's edge at another
    passage; and when the span-owning chunk IS hydrated, the citation points
    there (both replayed from the inline edge-expansion cases)."""
    late_edge_source = json.dumps(
        {
            "type": "relationship_edge",
            "doc_id": _DOC_ID,
            "clause_id": "9.1",
            "confidence": 0.9,
            "from_entity": "Acme Corp",
            "relationship": "obliges",
            "to_entity": "GlobalTech Ltd",
            "contract_index": 0,
            "document_version": 1,
        }
    )
    late_edge_body = (
        "Acme Corp obliges GlobalTech Ltd. Source clause: 9.1. "
        "Citation: clause 9.1. Confidence: 0.90. Grounding: chars 2000-2050."
    )

    # Case 1: no hydrated row covers chars 2000-2050 — no edge citation.
    graphiti = _install_episode_source(
        monkeypatch,
        [
            _episode(
                "edge-late",
                group_id=_tenant_group(_TENANT_A, _DOC_ID),
                content=late_edge_body,
                source_description=late_edge_source,
            )
        ],
    )
    llm = _ScriptedAskLLM([_sufficient()])
    repo = _RecordingRepository(span=(0, 500))
    graph_result = await _run_graph(llm, repo, graphiti=graphiti)
    graph_response = _graph_to_ask_response(graph_result)
    assert all("Grounded by" not in citation.claim for citation in graph_response.citations)

    # Case 2: the span-owning passage IS hydrated — the citation points there.
    graphiti2 = _install_episode_source(
        monkeypatch,
        [
            _episode(
                "edge-late",
                group_id=_tenant_group(_TENANT_A, _DOC_ID),
                content=late_edge_body,
                source_description=late_edge_source,
            )
        ],
    )
    llm2 = _ScriptedAskLLM([_sufficient()])
    repo2 = _RecordingRepository(
        span=(0, 500),
        extra_row={
            "chunk_id": "chunk-late",
            "document_id": _DOC_ID,
            "content": "Late clause: Acme Corp obliges GlobalTech Ltd.",
            "clause_type": "obligation",
            "metadata": {
                "source_start_char": 1900,
                "source_end_char": 2100,
                "contract_index": 0,
                "document_version": 1,
            },
        },
    )
    graph_result2 = await _run_graph(llm2, repo2, graphiti=graphiti2)
    graph_response2 = _graph_to_ask_response(graph_result2)
    late = [c for c in graph_response2.citations if "Grounded by" in c.claim]
    assert len(late) == 1
    assert late[0].chunk_id == "chunk-late"


@pytest.mark.usefixtures("_wired")
async def test_warnings_merge_and_dedupe_identically() -> None:
    warning_one = {
        "stage": "ingestion",
        "code": "LOW_OCR_CONFIDENCE",
        "message": "OCR confidence below threshold",
        "severity": "warning",
    }
    warning_two = {
        "stage": "enrichment",
        "code": "METADATA_PARTIAL",
        "message": "metadata extraction incomplete",
        "severity": "info",
    }
    duplicate_one = dict(warning_one)
    llm = _ScriptedAskLLM([_sufficient()])
    repo_inline = _RecordingRepository(
        quality_warnings=[warning_one, warning_two],
        extra_row={
            "chunk_id": "chunk-2",
            "document_id": _DOC_ID,
            "content": "The Employer shall also pay interest.",
            "clause_type": "payment",
            "metadata": {"source_start_char": 0, "source_end_char": 900},
            "quality_warnings": [duplicate_one],
        },
    )
    repo_graph = _RecordingRepository(
        quality_warnings=[warning_one, warning_two],
        extra_row={
            "chunk_id": "chunk-2",
            "document_id": _DOC_ID,
            "content": "The Employer shall also pay interest.",
            "clause_type": "payment",
            "metadata": {"source_start_char": 0, "source_end_char": 900},
            "quality_warnings": [duplicate_one],
        },
    )
    inline_result = await _service(llm, repo_inline).ask(
        user_id="user-1",
        tenant_id=_TENANT_A,
        allowed_doc_ids=[_DOC_ID],
        payload=_payload(),
        require_graphiti_verified=False,
    )
    graph_result = await _run_graph(llm, repo_graph)

    assert isinstance(inline_result, Success)
    inline_response = inline_result.unwrap()
    graph_response = _graph_to_ask_response(graph_result)
    _assert_identical_response_shape(inline_response, graph_response, "warning merge/dedupe")
    assert [w.model_dump() for w in inline_response.warnings] == [
        w.model_dump() for w in graph_response.warnings
    ]
    assert len(inline_response.warnings) == 2  # the duplicate is dropped


@pytest.mark.usefixtures("_wired")
async def test_unavailable_scope_documents_warn_like_the_inline_contract() -> None:
    """The scope-unavailable warning matches the inline contract verbatim
    (the `ask_via_retrieval_graph` wrapper's message), and the coverage
    envelope stays rows-derived (the served inline `_ask` derivation)."""
    llm = _ScriptedAskLLM([_sufficient()])
    repo = _RecordingRepository()
    graph_result = await _run_graph(
        llm,
        repo,
        state_overrides={
            # Version resolves but the contract scope does not, so the served
            # wrapper marks the document scope-unavailable.
            "document_contracts": [],
            "scope_unavailable_documents": [_DOC_ID],
        },
    )
    graph_response = _graph_to_ask_response(graph_result)
    scope_warning = [
        warning for warning in graph_response.warnings if warning.code == "GRAPH_SCOPE_UNAVAILABLE"
    ]
    assert len(scope_warning) == 1
    assert scope_warning[0].stage == "graph_scope"
    assert scope_warning[0].severity == "warning"
    assert scope_warning[0].message == (
        "Authoritative document version/contract scope was unavailable "
        "for 1 authorized document(s)."
    )
    # The rest of the envelope is unchanged: the answer still generated.
    assert graph_response.answer == "Acme Corp must pay the monthly charge."


# --------------------------------------------------------------------------
# Cache-key derivation parity (task 1.2 / the cache-key spec scenarios)
# --------------------------------------------------------------------------


def _cache_key_state(**overrides: Any) -> dict[str, Any]:
    state: dict[str, Any] = {
        "user_id": "user-1",
        "tenant_id": _TENANT_A,
        "query": _ORIGINAL_QUERY,
        "doc_ids_filter": [],
        "allowed_doc_ids": [_DOC_ID],
        "document_versions": [(_DOC_ID, 1, None)],
        "jurisdiction": None,
        "contract_type": None,
        "contract_index": None,
        "clause_type": None,
        "require_graphiti_verified": False,
    }
    state.update(overrides)
    return state


def _inline_cache_key_kwargs(state: dict[str, Any]) -> dict[str, Any]:
    """The inline `_build_answer_cache_key` kwargs for the same inputs."""
    return {
        "user_id": state["user_id"],
        "tenant_id": state["tenant_id"],
        "query": state["query"],
        "doc_ids_filter": state["doc_ids_filter"],
        "allowed_doc_ids": tuple(state["allowed_doc_ids"]),
        "document_versions": tuple(state["document_versions"]),
        "jurisdiction": state["jurisdiction"],
        "contract_type": state["contract_type"],
        "contract_index": state["contract_index"],
        "clause_type": state["clause_type"],
        "require_graphiti_verified": state["require_graphiti_verified"],
        "evidence_policy": (
            "graph_indexed_only" if state["require_graphiti_verified"] else "mechanically_verified"
        ),
    }


def test_cache_key_derivation_matches_the_inline_contract_exactly() -> None:
    for overrides in (
        {},
        {"require_graphiti_verified": True},
        {"document_versions": [(_DOC_ID, 2, "documents-v1")], "doc_ids_filter": ["doc-b"]},
        {"jurisdiction": "India", "contract_index": 3, "clause_type": "payment"},
    ):
        state = _cache_key_state(**overrides)
        inline_key = _build_answer_cache_key(**_inline_cache_key_kwargs(state))
        graph_key = _graph_answer_cache_key(state)
        assert inline_key == graph_key, (
            f"divergent behaviour: version-aware cache-key derivation (inputs={overrides})"
        )


def test_document_version_change_produces_a_different_key() -> None:
    first = _graph_answer_cache_key(_cache_key_state(document_versions=[(_DOC_ID, 1, None)]))
    second = _graph_answer_cache_key(_cache_key_state(document_versions=[(_DOC_ID, 2, None)]))
    assert first != second, (
        "divergent behaviour: a re-ingested document must not serve stale answers"
    )


def test_pipeline_version_change_produces_a_different_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from app.features.documents.domain import constants as documents_constants

    first = _graph_answer_cache_key(_cache_key_state())
    monkeypatch.setattr(documents_constants, "CANONICAL_PIPELINE_VERSION", "documents-v2")
    monkeypatch.setattr(documents_module, "CANONICAL_PIPELINE_VERSION", "documents-v2")
    second_graph = _graph_answer_cache_key(_cache_key_state())
    second_inline = _build_answer_cache_key(**_inline_cache_key_kwargs(_cache_key_state()))
    assert first != second_graph, (
        "divergent behaviour: a pipeline bump must invalidate cached answers"
    )
    assert second_graph == second_inline, (
        "divergent behaviour: pipeline-bumped key derivation matches the inline contract"
    )


@pytest.mark.usefixtures("_wired")
async def test_cache_hit_serves_the_cached_answer_under_the_same_key() -> None:
    llm = _ScriptedAskLLM([_sufficient()])
    # Inline: the second ask is a cache hit carrying the recorded envelope.
    inline_redis = _FakeRedis()
    repo_inline = _RecordingRepository()
    service = _service(llm, repo_inline, redis=inline_redis)
    first_inline = await service.ask(
        user_id="user-1",
        tenant_id=_TENANT_A,
        allowed_doc_ids=[_DOC_ID],
        payload=_payload(bypass_cache=False),
        require_graphiti_verified=False,
    )
    second_inline = await service.ask(
        user_id="user-1",
        tenant_id=_TENANT_A,
        allowed_doc_ids=[_DOC_ID],
        payload=_payload(bypass_cache=False),
        require_graphiti_verified=False,
    )
    assert isinstance(first_inline, Success)
    assert isinstance(second_inline, Success)
    assert second_inline.unwrap().cache_hit is True
    assert first_inline.unwrap().model_dump() == {
        **second_inline.unwrap().model_dump(),
        "cache_hit": False,
    }

    # Graph: the second invocation is a cache hit under the identical key.
    graph_redis = _FakeRedis()
    repo_graph = _RecordingRepository()
    llm_graph = _ScriptedAskLLM([_sufficient()])
    first_graph = await _run_graph(llm_graph, repo_graph, redis=graph_redis)
    second_graph = await _run_graph(llm_graph, repo_graph, redis=graph_redis)

    assert first_graph["cache_hit"] is False
    assert second_graph["cache_hit"] is True
    assert first_graph["generated_answer"].answer == second_graph["generated_answer"].answer, (
        "divergent behaviour: cache hit serves the cached answer"
    )
    # The graph derived exactly the key the inline derivation produces.
    assert graph_redis.setex_keys == [
        _build_answer_cache_key(**_inline_cache_key_kwargs(_cache_key_state()))
    ], "divergent behaviour: the graph writes under the inline cache key"
    assert graph_redis.get_keys[0] == graph_redis.setex_keys[0]
    assert inline_redis.setex_keys[0] == graph_redis.setex_keys[0], (
        "divergent behaviour: cache-key derivation identical across paths"
    )


# --------------------------------------------------------------------------
# Injected grader seam parity (task 1.5)
# --------------------------------------------------------------------------


class _SeamGrader:
    """Text-generation seam double returning scripted ContextGrade outputs."""

    def __init__(self, results: list[Any]) -> None:
        self._results = list(results)
        self.requests: list[Any] = []

    async def generate_structured(self, request: Any) -> Any:
        self.requests.append(request)
        outcome = self._results[min(len(self.requests) - 1, len(self._results) - 1)]
        if isinstance(outcome, Exception):
            raise outcome
        if isinstance(outcome, Failure):
            return outcome
        return Success(GeneratedStructured(output=outcome))


@pytest.mark.usefixtures("_wired")
async def test_seam_present_grade_matches_the_inline_seam_contract() -> None:
    grade = ContextGrade(sufficient=True, missing_aspects=[])
    inline_grader = _SeamGrader([grade])
    graph_grader = _SeamGrader([grade])
    repo_inline = _RecordingRepository()
    repo_graph = _RecordingRepository()

    inline_result = await _service(_ScriptedAskLLM([]), repo_inline, grader=inline_grader).ask(
        user_id="user-1",
        tenant_id=_TENANT_A,
        allowed_doc_ids=[_DOC_ID],
        payload=_payload(),
        require_graphiti_verified=False,
    )
    graph_result = await _run_graph(_ScriptedAskLLM([]), repo_graph, grader=graph_grader)

    assert isinstance(inline_result, Success)
    _assert_identical_response_shape(
        inline_result.unwrap(), _graph_to_ask_response(graph_result), "seam-present grade"
    )
    # The seam request is identical on both paths: same prompt, same system
    # message, temperature 0.1 (never the provider default).
    assert inline_grader.requests
    assert graph_grader.requests
    assert inline_grader.requests[0].prompt == graph_grader.requests[0].prompt
    assert inline_grader.requests[0].system == graph_grader.requests[0].system
    assert graph_grader.requests[0].temperature == pytest.approx(0.1)


@pytest.mark.usefixtures("_wired")
async def test_seam_failure_degrades_to_sufficient_identically() -> None:
    from app.ai.generation import TextGenerationUnavailableError

    inline_grader = _SeamGrader([Failure(TextGenerationUnavailableError())])
    graph_grader = _SeamGrader([Failure(TextGenerationUnavailableError())])
    repo_inline = _RecordingRepository()
    repo_graph = _RecordingRepository()

    inline_result = await _service(_ScriptedAskLLM([]), repo_inline, grader=inline_grader).ask(
        user_id="user-1",
        tenant_id=_TENANT_A,
        allowed_doc_ids=[_DOC_ID],
        payload=_payload(),
        require_graphiti_verified=False,
    )
    graph_result = await _run_graph(_ScriptedAskLLM([]), repo_graph, grader=graph_grader)

    assert isinstance(inline_result, Success)
    # The seam-absent/failed degrade: an unavailable grade proceeds to
    # generation rather than failing the ask.
    _assert_identical_response_shape(
        inline_result.unwrap(), _graph_to_ask_response(graph_result), "seam-failure degrade"
    )


@pytest.mark.usefixtures("_wired")
async def test_seam_absent_grade_replays_the_legacy_llm_path() -> None:
    """No context seam: grading rides the legacy structured LLM path."""
    llm_inline = _ScriptedAskLLM([_sufficient()])
    llm_graph = _ScriptedAskLLM([_sufficient()])
    repo_inline = _RecordingRepository()
    repo_graph = _RecordingRepository()
    inline_result = await _service(llm_inline, repo_inline).ask(
        user_id="user-1",
        tenant_id=_TENANT_A,
        allowed_doc_ids=[_DOC_ID],
        payload=_payload(),
        require_graphiti_verified=False,
    )
    graph_result = await _run_graph(llm_graph, repo_graph)

    assert isinstance(inline_result, Success)
    _assert_identical_response_shape(
        inline_result.unwrap(), _graph_to_ask_response(graph_result), "seam-absent grade"
    )
    assert len(llm_inline.grader_prompts) == len(llm_graph.grader_prompts) == 1
