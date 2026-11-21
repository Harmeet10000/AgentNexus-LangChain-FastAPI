"""Section-level hydration (G5 task 4.3): gated, additive, citation-precise.

The hydration walker resolves the parent section of a matched chunk from the
stored structural tree (fail-closed: no matching heading hydrates nothing),
the generator payload gains the section text only when the flag is on, and the
citations resolve to the matched clauses identically with and without
hydration — precision under hydration is pinned, not asserted.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from app.config.settings import get_settings
from app.features.documents.application.retrieval.graph.hydration import (
    hydrate_sections_for_chunks,
    section_text_for_heading_path,
)

if TYPE_CHECKING:
    from typing import Any


def _count_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def _tree() -> dict[str, object]:
    """A serialized Docling-shaped tree: two sections, the first nested."""
    return {
        "name": "Master Services Agreement",
        "body": {
            "children": ["#/texts/0", "#/texts/1", "#/groups/0", "#/texts/4", "#/texts/5"],
        },
        "groups": [
            {
                "self_ref": "#/groups/0",
                "label": "section",
                "name": "sub",
                "children": ["#/texts/2", "#/texts/3"],
            }
        ],
        "texts": [
            {"self_ref": "#/texts/0", "label": "section_header", "text": "Term and Termination"},
            {
                "self_ref": "#/texts/1",
                "label": "text",
                "text": "Either party may terminate for material breach on thirty days notice.",
            },
            {
                "self_ref": "#/texts/2",
                "label": "section_header",
                "text": "Consequences of Termination",
                "level": 2,
            },
            {
                "self_ref": "#/texts/3",
                "label": "text",
                "text": "Accrued payment obligations survive termination of this agreement.",
            },
            {"self_ref": "#/texts/4", "label": "section_header", "text": "Governing Law"},
            {
                "self_ref": "#/texts/5",
                "label": "text",
                "text": "Governing law of Ontario applies.",
            },
        ],
    }


def test_parent_section_hydrates_with_nested_subsection_text() -> None:
    section = section_text_for_heading_path(
        _tree(),
        ["Term and Termination"],
        count_tokens=_count_tokens,
    )
    assert section is not None
    # The parent section plus its nested subsection's text — not the next
    # top-level section ("Governing law..." sits outside the target path).
    assert "material breach" in section
    assert "Accrued payment obligations" in section
    assert "Governing law of Ontario" not in section


def test_unmatched_heading_path_yields_no_fabricated_text() -> None:
    assert (
        section_text_for_heading_path(_tree(), ["Nonexistent Section"], count_tokens=_count_tokens)
        is None
    )
    assert section_text_for_heading_path(_tree(), [], count_tokens=_count_tokens) is None


def test_hydrate_sections_for_chunks_maps_by_chunk_id() -> None:
    from app.features.documents.application.retrieval.graph.state import RetrievedChunk as Chunk

    chunks = [
        Chunk(
            chunk_id="chunk-1",
            chunk_text="Either party may terminate for material breach.",
            preamble="",
            clause_type="termination",
            parent_doc_id="doc-1",
            chunk_index=0,
            metadata_={"heading_path": ["Term and Termination"]},
            custom_metadata={},
            score=1.0,
        ),
        Chunk(
            chunk_id="chunk-2",
            chunk_text="orphan",
            preamble="",
            clause_type="other",
            parent_doc_id="doc-1",
            chunk_index=1,
            metadata_={"heading_path": ["Missing Section"]},
            custom_metadata={},
            score=0.9,
        ),
        Chunk(
            chunk_id="chunk-3",
            chunk_text="no path",
            preamble="",
            clause_type="other",
            parent_doc_id="doc-1",
            chunk_index=2,
            metadata_={},
            custom_metadata={},
            score=0.8,
        ),
    ]
    hydrated = hydrate_sections_for_chunks(
        trees={"doc-1": _tree()}, chunks=chunks, count_tokens=_count_tokens
    )
    assert set(hydrated) == {"chunk-1"}
    assert "material breach" in hydrated["chunk-1"]


def _ref_tree() -> dict[str, object]:
    """The stored Docling shape: children are ``{"$ref": ...}`` objects."""
    return {
        "name": "Master Services Agreement",
        "body": {
            "children": [
                {"$ref": "#/texts/0"},
                {"$ref": "#/texts/1"},
                {"cref": "#/groups/0"},
            ]
        },
        "groups": [
            {
                "self_ref": "#/groups/0",
                "label": "section",
                "name": "sub",
                "children": [{"$ref": "#/texts/2"}, {"$ref": "#/texts/3"}],
            }
        ],
        "texts": [
            {"self_ref": "#/texts/0", "label": "section_header", "text": "Term and Termination"},
            {
                "self_ref": "#/texts/1",
                "label": "text",
                "text": "Either party may terminate for material breach on thirty days notice.",
            },
            {
                "self_ref": "#/texts/2",
                "label": "section_header",
                "text": "Consequences of Termination",
                "level": 2,
            },
            {
                "self_ref": "#/texts/3",
                "label": "text",
                "text": "Accrued payment obligations survive termination of this agreement.",
            },
        ],
    }


def test_ref_shaped_children_hydrate_the_parent_section() -> None:
    """Stored trees serialize children as reference objects: the walker
    resolves them, so section hydration works for real stored documents."""
    section = section_text_for_heading_path(
        _ref_tree(),
        ["Term and Termination"],
        count_tokens=_count_tokens,
    )
    assert section is not None
    assert "material breach" in section
    # The nested subsection (a `cref` reference) hydrates too; the next
    # top-level section does not exist here to leak in.
    assert "Accrued payment obligations" in section


def test_unresolvable_ref_skips_the_child_with_a_warning() -> None:
    from loguru import logger as loguru_logger

    tree = {
        "body": {
            "children": [
                {"$ref": "#/texts/0"},
                {"$ref": "#/texts/99"},
                {"$ref": "#/texts/1"},
            ]
        },
        "texts": [
            {"self_ref": "#/texts/0", "label": "section_header", "text": "Term and Termination"},
            {
                "self_ref": "#/texts/1",
                "label": "text",
                "text": "Either party may terminate for material breach.",
            },
        ],
    }
    events: list[dict[str, object]] = []

    def _sink(message: Any) -> None:
        record = message.record
        if record["message"] == "section_hydration_ref_unresolved":
            events.append({"reference": record["extra"].get("reference")})

    sink_id = loguru_logger.add(_sink, level="DEBUG", format="{message}")
    try:
        section = section_text_for_heading_path(
            tree, ["Term and Termination"], count_tokens=_count_tokens
        )
    finally:
        loguru_logger.remove(sink_id)
    # The unresolvable reference is skipped with its typed warning — never a
    # raise — and the resolvable siblings still hydrate the section.
    assert section == "Either party may terminate for material breach."
    assert events == [{"reference": "#/texts/99"}]


@pytest.fixture
def _hydration_on(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DOCUMENTS_SECTION_HYDRATION_ENABLED", "true")
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def test_hydration_flag_defaults_off() -> None:
    from app.config.settings import Settings

    assert Settings().DOCUMENTS_SECTION_HYDRATION_ENABLED is False


class _HydrationRepo:
    """Retrieval rows plus the stored structural tree for the parent document."""

    async def bm25_search(self, **_kwargs: object) -> Any:
        from returns.result import Success

        return Success([{"chunk_id": "chunk-1", "score": 2.0}])

    async def vector_search(self, **_kwargs: object) -> Any:
        from returns.result import Success

        return Success([])

    async def trigram_search(self, **_kwargs: object) -> Any:
        from returns.result import Success

        return Success([])

    async def fetch_chunks_by_ids(self, chunk_ids: list[str], **_kwargs: object) -> Any:
        from returns.result import Success

        return Success(
            {
                chunk_id: {
                    "document_id": "doc-1",
                    "title": "MSA",
                    "content": "Either party may terminate for material breach on notice.",
                    "preamble": "",
                    "document_version": 1,
                    "contract_index": 0,
                    "pipeline_version": "canonical",
                    "chunk_index": 0,
                    "chunk_kind": "legal_contract",
                    "clause_type": "termination",
                    "chunk_metadata": {"heading_path": ["Term and Termination"]},
                    "quality_warnings": [],
                    "graphiti_verified": False,
                    "graph_indexed_at": None,
                }
                for chunk_id in chunk_ids
            }
        )

    async def fetch_structural_trees(
        self, *, user_id: str, document_ids: list[str], tenant_id: str | None = None
    ) -> Any:
        from returns.result import Success

        assert document_ids == ["doc-1"]
        return Success([{"document_id": "doc-1", "structural_tree": _tree()}])


class _CaptureLLM:
    """Records the generator payload; grades sufficient, answers grounded."""

    def __init__(self) -> None:
        self.payloads: list[str] = []

    def with_structured_output(self, schema: type) -> Any:
        return _CaptureHandler(schema, self)


class _CaptureHandler:
    def __init__(self, schema: type, llm: _CaptureLLM) -> None:
        self._schema = schema
        self._llm = llm

    async def ainvoke(self, messages: list[Any]) -> Any:
        from app.features.documents.application.retrieval.graph.state import (
            ContextGrade,
            GeneratedAnswer,
            QueryPlan,
            SourceCriteria,
        )

        if self._schema is QueryPlan:
            return QueryPlan(rewritten_query="termination rights")
        if self._schema is SourceCriteria:
            return SourceCriteria()
        if self._schema is ContextGrade:
            return ContextGrade(sufficient=True, missing_aspects=[])
        if self._schema is GeneratedAnswer:
            self._llm.payloads.append(str(messages[-1].content))
            return GeneratedAnswer(
                answer="terminates on notice",
                citations=[{"chunk_id": "chunk-1", "clause_type": "termination", "claim": "c"}],
                confidence="high",
            )
        raise AssertionError(f"unexpected schema {self._schema}")  # noqa: EM102


class _PassThroughReranker:
    async def rerank(self, _query: str, chunks: list[Any], *, limit: int) -> list[Any]:
        return chunks[:limit]


async def _fake_embed(*_args: object, **_kwargs: object) -> list[float]:
    return [0.1, 0.2, 0.3]


def _input() -> dict[str, Any]:
    return {
        "user_id": "user-1",
        "tenant_id": "tenant-a",
        "allowed_doc_ids": ["doc-1"],
        "document_versions": [("doc-1", 1, None)],
        "document_contracts": [("doc-1", 1, (0,), False)],
        "scope_unavailable_documents": [],
        "scope_bound": True,
        "query": "termination rights",
        "doc_ids_filter": [],
        "contract_index": None,
        "jurisdiction": None,
        "contract_type": None,
        "clause_type": None,
        "require_graphiti_verified": False,
        "messages": [],
        "iteration_count": 0,
    }


async def _run_graph(llm: Any) -> dict[str, Any]:
    import app.features.documents.application.retrieval.graph.nodes as graph_nodes
    from app.features.documents.application.retrieval.graph import (
        RetrievalGraphContext,
        build_retrieval_graph,
    )

    original = graph_nodes.embed_text
    graph_nodes.embed_text = _fake_embed
    try:
        graph = build_retrieval_graph()
        return await graph.ainvoke(
            _input(),
            context=RetrievalGraphContext(
                llm=llm, repo=_HydrationRepo(), reranker=_PassThroughReranker()
            ),
        )
    finally:
        graph_nodes.embed_text = original


@pytest.mark.usefixtures("_hydration_on")
async def test_hydration_reaches_the_generator_and_citations_stay_clause_precise() -> None:
    llm = _CaptureLLM()
    result = await _run_graph(llm)

    assert llm.payloads, "the generator must have run"
    assert "section_context" in llm.payloads[0]
    assert "material breach" in llm.payloads[0]
    # The citation resolves to the matched clause identically — the hydrated
    # section text never becomes a citation target.
    citations = result["resolved_citations"]
    assert [(c.chunk_id, c.clause_type) for c in citations] == [("chunk-1", "termination")]
    assert all(c.document_id == "doc-1" for c in citations)


async def test_hydration_off_keeps_the_payload_unchanged(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DOCUMENTS_SECTION_HYDRATION_ENABLED", "false")
    get_settings.cache_clear()
    llm = _CaptureLLM()
    await _run_graph(llm)
    assert llm.payloads
    assert "section_context" not in llm.payloads[0]
