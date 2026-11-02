"""Unit tests for the portfolio traversal tools (tasks 2.1–2.4).

Covers the seeded-edges happy paths per tool, scope discipline (server-owned
allowlist, no model-supplied scope path), the no-context refusal, the
honest empty states (``no_results`` / ``budget_exhausted`` / ``refused``
never collapsing into an empty success), the envelope bounds at high
match counts, cap-truncation disclosure, and real ToolNode invocations
(the runtime injection seam the specialists actually call through).
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest
from langchain.tools import ToolRuntime
from langchain_core.messages import AIMessage, ToolMessage
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode

from app.features.documents.application.tools.portfolio import (
    _TRAVERSAL_READ_CAP,
    MAX_CROSS_DOCUMENT_OBLIGATION_MATCHES,
    MAX_PARTY_OBLIGATION_MAP_MATCHES,
    TRAVERSAL_GATE_NAME,
    make_cross_document_obligations_tool,
    make_find_conflicting_clauses_tool,
    make_party_obligation_map_tool,
)
from app.retrieval.edge_expansion import LegalEdgeRecord

if TYPE_CHECKING:
    from collections.abc import Sequence
    from typing import Any

pytestmark = pytest.mark.unit

ALLOWED_DOCS = ("doc-a", "doc-b")
OUT_OF_SCOPE_DOC = "doc-outside-scope"


def _edge(
    edge_id: str,
    *,
    document_id: str = "doc-a",
    from_entity: str = "Acme Corp",
    relationship: str = "obliges",
    to_entity: str = "GlobalTech Ltd",
    clause_id: str = "2.1",
) -> LegalEdgeRecord:
    return LegalEdgeRecord(
        edge_id=edge_id,
        document_id=document_id,
        from_entity=from_entity,
        relationship=relationship,
        to_entity=to_entity,
        clause_id=clause_id,
        char_start=1042,
        char_end=1110,
        document_version=2,
        contract_index=0,
    )


def _seeded_edges() -> tuple[LegalEdgeRecord, ...]:
    """A small two-document portfolio, plus one out-of-scope document."""
    return (
        _edge("edge-1", document_id="doc-a", clause_id="2.1"),
        _edge(
            "edge-2",
            document_id="doc-a",
            from_entity="GlobalTech Ltd",
            to_entity="Acme Corp",
            clause_id="5.4",
        ),
        _edge(
            "edge-3",
            document_id="doc-b",
            relationship="indemnifies",
            to_entity="VentureCo",
            clause_id="indemn-7.2",
        ),
        _edge(
            "edge-4",
            document_id="doc-b",
            relationship="references",
            to_entity="VentureCo",
            clause_id="7.3",
        ),
        _edge(
            "edge-5",
            document_id="doc-b",
            from_entity="GlobalTech Ltd",
            relationship="overridden_by",
            to_entity="Acme Corp",
            clause_id="2.9",
        ),
        # Outside the authorized allowlist: must never reach any envelope.
        _edge(
            "edge-6",
            document_id=OUT_OF_SCOPE_DOC,
            to_entity="VentureCo",
            clause_id="1.1",
        ),
    )


class _StubGraphitiService:
    """Test double for the read-only relationship-edge fetch.

    Mirrors the real bound read's strict contract: a seeded exception only
    surfaces when the caller asks for strict mode (the binding-time probe);
    leniently the read degrades to an empty result.
    """

    def __init__(self, edges: tuple[LegalEdgeRecord, ...] | Exception) -> None:
        self.edges = edges
        self.calls: list[dict[str, Any]] = []

    async def fetch_relationship_edges(
        self,
        query: str,
        *,
        allowed_doc_ids: tuple[str, ...] = (),
        tenant_id: str | None = None,
        contract_index: int | None = None,
        document_versions: Sequence[tuple[str, int, str | None]] = (),
        document_contracts: Sequence[tuple[str, int, Sequence[int | None], bool]] = (),
        num_results: int = 32,
        strict: bool = False,
    ) -> tuple[LegalEdgeRecord, ...]:
        self.calls.append(
            {
                "query": query,
                "allowed_doc_ids": allowed_doc_ids,
                "tenant_id": tenant_id,
                "num_results": num_results,
                "strict": strict,
            }
        )
        if isinstance(self.edges, Exception):
            if strict:
                raise self.edges
            return ()
        # Return everything it holds — the tool must do its own scoping.
        return self.edges


def _runtime(**context_overrides: Any) -> ToolRuntime:
    context: dict[str, Any] = {
        "user_id": "user-1",
        "tenant_id": "tenant-1",
        "allowed_doc_ids": ALLOWED_DOCS,
        # Production principals carry document.read from the auth layer; the
        # traversal tools enforce it like the other scoped document tools.
        "allowed_operations": ("document.read", "evidence.search"),
        "thread_id": "thread-1",
        "run_id": "run-1",
    }
    context.update(context_overrides)
    return ToolRuntime(
        state={},
        context=context,
        config={},
        stream_writer=None,
        tool_call_id="test-call",
        store=None,
    )


def _tools(edges: tuple[LegalEdgeRecord, ...]) -> tuple[Any, Any, Any, _StubGraphitiService]:
    service = _StubGraphitiService(edges)
    return (
        make_cross_document_obligations_tool(service),
        make_find_conflicting_clauses_tool(service),
        make_party_obligation_map_tool(service),
        service,
    )


# ---------------------------------------------------------------------------
# Task 2.1 — cross_document_obligations
# ---------------------------------------------------------------------------


async def test_cross_document_obligations_returns_handle_bound_matches() -> None:
    obligations, _, _, service = _tools(_seeded_edges())

    envelope = await obligations.coroutine(  # type: ignore[call-arg]
        party="acme corp",  # case-insensitive exact party match
        runtime=_runtime(),
    )

    assert envelope["status"] == "completed"
    matches = envelope["matches"]
    # obligation-class edges touching Acme in scope; the `references` edge and
    # the out-of-scope document's edge are both excluded.
    assert [m["edge_id"] for m in matches] == ["edge-1", "edge-2", "edge-3"]
    first = matches[0]
    assert first["from_party"] == "Acme Corp"
    assert first["to_party"] == "GlobalTech Ltd"
    assert first["relationship"] == "obliges"
    assert first["document_id"] == "doc-a"
    assert first["clause_id"] == "2.1"
    assert first["document_version"] == 2
    assert first["contract_index"] == 0
    # Handles are deterministic clause locators, never clause text — and
    # never presented as EvidenceService handle tokens (finding D).
    assert first["evidence_locator"] == "portfolio:doc-a:2.1:edge-1"
    assert not any(
        key in {"text", "content", "snippet", "clause_text"} for match in matches for key in match
    )
    coverage = envelope["coverage"]
    assert coverage == {
        "documents_in_scope": 2,
        "edges_scanned": 6,
        "matches_found": 3,
        "matches_returned": 3,
        "truncated": False,
    }
    # The traversal ran inside the server-owned allowlist only.
    assert service.calls[-1]["allowed_doc_ids"] == ALLOWED_DOCS
    assert service.calls[-1]["tenant_id"] == "tenant-1"


async def test_cross_document_obligations_clause_class_filter() -> None:
    obligations, _, _, _ = _tools(_seeded_edges())

    envelope = await obligations.coroutine(  # type: ignore[call-arg]
        party="Acme Corp",
        clause_class="indemn",
        runtime=_runtime(),
    )

    assert envelope["status"] == "completed"
    assert [m["clause_id"] for m in envelope["matches"]] == ["indemn-7.2"]
    assert envelope["matches"][0]["relationship"] == "indemnifies"


def test_traversal_input_schemas_expose_query_arguments_only() -> None:
    obligations, conflicts, party_map, _ = _tools(_seeded_edges())
    scope_like = {"allowed_doc_ids", "tenant_id", "user_id", "run_id", "doc_ids", "scope"}
    assert set(obligations.args_schema.model_json_schema()["properties"]) == {
        "party",
        "clause_class",
    }
    assert set(conflicts.args_schema.model_json_schema()["properties"]) == {"conflict_notion"}
    assert set(party_map.args_schema.model_json_schema()["properties"]) == {"parties"}
    for schema, required in (
        (obligations.args_schema, {"party": "Acme Corp"}),
        (conflicts.args_schema, {"conflict_notion": "overridden_by"}),
        (party_map.args_schema, {"parties": ["Acme Corp"]}),
    ):
        assert not set(schema.model_json_schema()["properties"]) & scope_like
        # ToolNode injects the runtime kwarg before validation, so the schemas
        # must tolerate extras (review finding A, WebSearchInput precedent) —
        # a scope-like extra key is IGNORED, never honored as an argument.
        validated = schema.model_validate(
            {
                **required,
                "allowed_doc_ids": [OUT_OF_SCOPE_DOC],
                "tenant_id": "other-tenant",
            }
        )
        assert not set(validated.model_dump()) & scope_like


async def test_out_of_scope_edges_never_ship_even_when_the_read_returns_them() -> None:
    """Defense in depth: a compromised read cannot widen the traversal scope."""
    obligations, _, _, _ = _tools(_seeded_edges())

    envelope = await obligations.coroutine(  # type: ignore[call-arg]
        party="Acme Corp",
        runtime=_runtime(),
    )

    assert all(match["document_id"] in ALLOWED_DOCS for match in envelope["matches"])
    assert all(match["edge_id"] != "edge-6" for match in envelope["matches"])


# ---------------------------------------------------------------------------
# Task 2.2 — find_conflicting_clauses
# ---------------------------------------------------------------------------


async def test_find_conflicting_clauses_returns_both_sides_with_provenance() -> None:
    _, conflicts, _, _ = _tools(_seeded_edges())

    envelope = await conflicts.coroutine(  # type: ignore[call-arg]
        conflict_notion="overridden_by",
        runtime=_runtime(),
    )

    assert envelope["status"] == "completed"
    assert len(envelope["matches"]) == 1
    match = envelope["matches"][0]
    assert match["from_party"] == "GlobalTech Ltd"
    assert match["to_party"] == "Acme Corp"
    assert match["relationship"] == "overridden_by"
    assert match["document_id"] == "doc-b"
    assert match["evidence_locator"] == "portfolio:doc-b:2.9:edge-5"


async def test_find_conflicting_clauses_covers_amendment_chain_notion() -> None:
    edges = (_edge("edge-7", document_id="doc-b", relationship="supersedes", clause_id="3.0"),)
    _, conflicts, _, _ = _tools(edges)

    envelope = await conflicts.coroutine(  # type: ignore[call-arg]
        conflict_notion="Supersedes",
        runtime=_runtime(),
    )

    assert envelope["status"] == "completed"
    assert [m["edge_id"] for m in envelope["matches"]] == ["edge-7"]


async def test_find_conflicting_clauses_unknown_notion_is_a_refusal() -> None:
    _, conflicts, _, _ = _tools(_seeded_edges())

    envelope = await conflicts.coroutine(  # type: ignore[call-arg]
        conflict_notion="contradicts",
        runtime=_runtime(),
    )

    # An argument the graph cannot interpret is a typed refusal naming the
    # closed vocabulary — never a traversal that pretends to have run.
    assert envelope["status"] == "refused"
    assert "contradicts" in envelope["message"]
    assert "overridden_by" in envelope["message"]
    assert envelope["matches"] == []


# ---------------------------------------------------------------------------
# Task 2.3 — party_obligation_map
# ---------------------------------------------------------------------------


async def test_party_obligation_map_returns_the_full_web() -> None:
    _, _, party_map, _ = _tools(_seeded_edges())

    envelope = await party_map.coroutine(  # type: ignore[call-arg]
        parties=("GlobalTech Ltd", "VentureCo"),
        runtime=_runtime(),
    )

    assert envelope["status"] == "completed"
    # Every typed edge touching either named party in scope (all
    # relationships, unlike the obligation-class sweep), in the
    # deterministic (document, clause, edge) order.
    assert [m["edge_id"] for m in envelope["matches"]] == [
        "edge-1",
        "edge-2",
        "edge-5",
        "edge-4",
        "edge-3",
    ]


# ---------------------------------------------------------------------------
# Task 2.4 — honest empty states and bounds
# ---------------------------------------------------------------------------


async def test_no_result_query_is_a_distinct_status_never_an_empty_success() -> None:
    obligations, _, _, _ = _tools(_seeded_edges())

    envelope = await obligations.coroutine(  # type: ignore[call-arg]
        party="Nobody LLC",
        runtime=_runtime(),
    )

    assert envelope["status"] == "no_results"
    assert envelope["matches"] == []
    assert envelope["message"] is not None
    assert "no-result state" in envelope["message"]
    # The traversal genuinely ran over the scope.
    assert envelope["coverage"]["documents_in_scope"] == 2
    assert envelope["coverage"]["edges_scanned"] == 6


async def test_cross_document_budget_exhaustion_returns_partial_matches() -> None:
    edges = tuple(
        _edge(f"edge-{index}", clause_id=f"C-{index:03d}")
        for index in range(MAX_CROSS_DOCUMENT_OBLIGATION_MATCHES + 2)
    )
    obligations, _, _, _ = _tools(edges)

    envelope = await obligations.coroutine(  # type: ignore[call-arg]
        party="Acme Corp",
        runtime=_runtime(),
    )

    assert envelope["status"] == "budget_exhausted"
    assert len(envelope["matches"]) == MAX_CROSS_DOCUMENT_OBLIGATION_MATCHES
    coverage = envelope["coverage"]
    assert coverage["matches_found"] == MAX_CROSS_DOCUMENT_OBLIGATION_MATCHES + 2
    assert coverage["matches_returned"] == MAX_CROSS_DOCUMENT_OBLIGATION_MATCHES
    assert "budget exhausted" in envelope["message"]


async def test_party_map_budget_exhaustion_respects_its_own_bound() -> None:
    edges = tuple(
        _edge(
            f"edge-{index}",
            from_entity="GlobalTech Ltd",
            to_entity=f"Counterparty {index}",
            clause_id=f"C-{index:03d}",
        )
        for index in range(MAX_PARTY_OBLIGATION_MAP_MATCHES + 1)
    )
    _, _, party_map, _ = _tools(edges)

    envelope = await party_map.coroutine(  # type: ignore[call-arg]
        parties=("GlobalTech Ltd",),
        runtime=_runtime(),
    )

    assert envelope["status"] == "budget_exhausted"
    assert len(envelope["matches"]) == MAX_PARTY_OBLIGATION_MAP_MATCHES
    assert envelope["coverage"]["matches_found"] == MAX_PARTY_OBLIGATION_MAP_MATCHES + 1


# ---------------------------------------------------------------------------
# Refusals: no context, no scope, no backend, closed gate
# ---------------------------------------------------------------------------


async def test_no_runtime_context_refuses_per_the_executor_contract() -> None:
    obligations, conflicts, party_map, _ = _tools(_seeded_edges())
    for traversal_tool in (obligations, conflicts, party_map):
        envelope = await traversal_tool.coroutine(  # type: ignore[call-arg]
            **(
                {"party": "Acme Corp"}
                if traversal_tool is obligations
                else {"conflict_notion": "overridden_by"}
                if traversal_tool is conflicts
                else {"parties": ("Acme Corp",)}
            ),
            runtime=None,
        )
        assert envelope["status"] == "refused"
        assert "authorization context is required" in envelope["message"]


async def test_unauthorized_operation_refuses() -> None:
    obligations, _, _, _ = _tools(_seeded_edges())

    envelope = await obligations.coroutine(  # type: ignore[call-arg]
        party="Acme Corp",
        runtime=_runtime(allowed_operations=("evidence.search",)),
    )

    assert envelope["status"] == "refused"
    assert "operation is outside the authorized capability" in envelope["message"]


async def test_empty_document_scope_refuses() -> None:
    obligations, _, _, service = _tools(_seeded_edges())

    envelope = await obligations.coroutine(  # type: ignore[call-arg]
        party="Acme Corp",
        runtime=_runtime(allowed_doc_ids=()),
    )

    assert envelope["status"] == "refused"
    assert "no authorized document scope" in envelope["message"]
    assert service.calls == []  # fail-closed: nothing was read


async def test_backendless_tool_refuses_rather_than_crashing() -> None:
    obligations = make_cross_document_obligations_tool(None)

    envelope = await obligations.coroutine(  # type: ignore[call-arg]
        party="Acme Corp",
        runtime=_runtime(),
    )

    assert envelope["status"] == "refused"
    assert "no graph backend" in envelope["message"] or "graph backend" in envelope["message"]


async def test_gate_closed_at_runtime_refuses_naming_the_gate() -> None:
    # A corpus that lost its typed edges after binding: refused, never an
    # empty success (runtime probe behind the binding-time gate).
    obligations, _, _, _ = _tools(())

    envelope = await obligations.coroutine(  # type: ignore[call-arg]
        party="Acme Corp",
        runtime=_runtime(),
    )

    assert envelope["status"] == "refused"
    assert f"gate '{TRAVERSAL_GATE_NAME}' is closed" in envelope["message"]


# ---------------------------------------------------------------------------
# Real ToolNode invocations (review finding A regression)
# ---------------------------------------------------------------------------


def _tool_call(name: str, args: dict[str, Any]) -> dict[str, Any]:
    return {"name": name, "args": args, "id": f"call-{name}", "type": "tool_call"}


async def test_all_three_tools_run_through_a_real_tool_node() -> None:
    """The specialists' actual invocation seam: a compiled graph's ToolNode.

    ToolNode injects the ``InjectedToolArg``-marked ``runtime`` kwarg into
    the call args before validating them against the args schema, so schemas
    with ``extra="forbid"`` rejected EVERY ToolNode invocation (the
    ``WebSearchInput`` precedent, review finding A). This pins that each
    portfolio tool executes end to end through a real ToolNode in a compiled
    graph, with the granted runtime context flowing through the graph
    context into the injected runtime.
    """
    obligations, conflicts, party_map, service = _tools(_seeded_edges())
    builder = StateGraph(MessagesState)
    builder.add_node("tools", ToolNode([obligations, conflicts, party_map]))
    builder.add_edge(START, "tools")
    builder.add_edge("tools", END)
    graph = builder.compile()

    ai_message = AIMessage(
        content="",
        tool_calls=[
            _tool_call("cross_document_obligations", {"party": "Acme Corp"}),
            _tool_call("find_conflicting_clauses", {"conflict_notion": "overridden_by"}),
            _tool_call("party_obligation_map", {"parties": ["GlobalTech Ltd", "VentureCo"]}),
        ],
    )
    result = await graph.ainvoke(
        {"messages": [ai_message]},
        context={
            "user_id": "user-1",
            "tenant_id": "tenant-1",
            "allowed_doc_ids": ALLOWED_DOCS,
            "allowed_operations": ("document.read",),
        },
    )

    tool_messages = [message for message in result["messages"] if isinstance(message, ToolMessage)]
    assert len(tool_messages) == 3
    envelopes = {message.tool_call_id: json.loads(message.content) for message in tool_messages}
    assert all(message.status == "success" for message in tool_messages)
    assert envelopes["call-cross_document_obligations"]["status"] == "completed"
    assert [m["edge_id"] for m in envelopes["call-cross_document_obligations"]["matches"]] == [
        "edge-1",
        "edge-2",
        "edge-3",
    ]
    assert envelopes["call-find_conflicting_clauses"]["status"] == "completed"
    assert envelopes["call-find_conflicting_clauses"]["matches"][0]["edge_id"] == "edge-5"
    assert envelopes["call-party_obligation_map"]["status"] == "completed"
    assert len(envelopes["call-party_obligation_map"]["matches"]) == 5
    # Every traversal ran inside the server-owned allowlist, and the runtime
    # reached the tool bodies through the injection seam (not via .coroutine).
    assert {call["allowed_doc_ids"] for call in service.calls} == {ALLOWED_DOCS}
    assert all(call["strict"] is False for call in service.calls)


async def test_tool_node_invocation_without_the_document_read_grant_refuses() -> None:
    """Through a real ToolNode, a missing grant still refuses (fail closed)."""
    obligations, _, _, _ = _tools(_seeded_edges())
    builder = StateGraph(MessagesState)
    builder.add_node("tools", ToolNode([obligations]))
    builder.add_edge(START, "tools")
    builder.add_edge("tools", END)
    graph = builder.compile()

    result = await graph.ainvoke(
        {
            "messages": [
                AIMessage(
                    content="",
                    tool_calls=[_tool_call("cross_document_obligations", {"party": "Acme Corp"})],
                )
            ]
        },
        context={
            "user_id": "user-1",
            "tenant_id": "tenant-1",
            "allowed_doc_ids": ALLOWED_DOCS,
            "allowed_operations": ("evidence.search",),
        },
    )

    tool_message = result["messages"][-1]
    assert isinstance(tool_message, ToolMessage)
    envelope = json.loads(tool_message.content)
    assert envelope["status"] == "refused"


# ---------------------------------------------------------------------------
# Cap-truncation disclosure (review finding C)
# ---------------------------------------------------------------------------


async def test_capped_read_with_matches_discloses_truncation() -> None:
    """A read that reached the cap never presents a prefix as a whole answer."""
    edges = (
        _edge("edge-match-1"),
        _edge("edge-match-2"),
        _edge("edge-match-3", clause_id="5.5"),
        *(
            _edge(
                f"edge-{index}",
                from_entity="Other Corp",
                to_entity="Also Other Corp",
                clause_id=f"C-{index:03d}",
            )
            for index in range(_TRAVERSAL_READ_CAP - 3)
        ),
    )
    obligations, _, _, _ = _tools(edges)

    envelope = await obligations.coroutine(  # type: ignore[call-arg]
        party="Acme Corp",
        runtime=_runtime(),
    )

    assert envelope["status"] == "completed"
    assert [m["edge_id"] for m in envelope["matches"]] == [
        "edge-match-1",
        "edge-match-2",
        "edge-match-3",
    ]
    assert envelope["coverage"]["edges_scanned"] == _TRAVERSAL_READ_CAP
    assert envelope["coverage"]["matches_found"] == 3
    assert envelope["coverage"]["truncated"] is True
    assert envelope["message"] is not None
    assert "reached its cap" in envelope["message"]
    assert "later matches may exist" in envelope["message"]


async def test_capped_read_over_the_bound_reports_budget_exhausted_and_truncated() -> None:
    edges = tuple(
        _edge(f"edge-{index}", clause_id=f"C-{index:03d}") for index in range(_TRAVERSAL_READ_CAP)
    )
    _, _, party_map, _ = _tools(edges)

    envelope = await party_map.coroutine(  # type: ignore[call-arg]
        parties=("Acme Corp",),
        runtime=_runtime(),
    )

    assert envelope["status"] == "budget_exhausted"
    assert envelope["coverage"]["truncated"] is True
    assert envelope["coverage"]["matches_found"] == _TRAVERSAL_READ_CAP
    assert envelope["coverage"]["matches_returned"] == MAX_PARTY_OBLIGATION_MAP_MATCHES


async def test_capped_read_with_no_matches_is_never_no_results() -> None:
    """A capped empty read is unevaluable, not an honest no-result state."""
    edges = tuple(
        _edge(
            f"edge-{index}",
            from_entity="Other Corp",
            to_entity="Also Other Corp",
            clause_id=f"C-{index:03d}",
        )
        for index in range(_TRAVERSAL_READ_CAP)
    )
    obligations, _, _, _ = _tools(edges)

    envelope = await obligations.coroutine(  # type: ignore[call-arg]
        party="Acme Corp",
        runtime=_runtime(),
    )

    assert envelope["status"] != "no_results"
    assert envelope["status"] == "budget_exhausted"
    assert envelope["matches"] == []
    assert envelope["coverage"]["truncated"] is True
    assert envelope["message"] is not None
    assert "not fully searched" in envelope["message"]


async def test_uncapped_empty_read_still_reports_no_results() -> None:
    """Below the cap an empty read is a genuine no-result state (unchanged)."""
    edges = (
        _edge("edge-1", from_entity="Other Corp", to_entity="Also Other Corp"),
        _edge("edge-2", from_entity="Third Corp", to_entity="Fourth Corp"),
    )
    obligations, _, _, _ = _tools(edges)

    envelope = await obligations.coroutine(  # type: ignore[call-arg]
        party="Acme Corp",
        runtime=_runtime(),
    )

    assert envelope["status"] == "no_results"
    assert envelope["coverage"]["truncated"] is False


# ---------------------------------------------------------------------------
# Evidence locators vs EvidenceService handles (review finding D)
# ---------------------------------------------------------------------------


async def test_locators_are_scope_clean_through_the_evidence_read_seam() -> None:
    """A locator's document/clause resolves within the run's scope — and the
    read seam answers ``invalid_handle`` (never ``unauthorized``), pinning
    that the locator is not mistaken for an EvidenceService handle token
    that could resolve outside the run's authorization."""
    from app.features.documents.application.evidence.service import EvidenceService

    obligations, _, _, _ = _tools(_seeded_edges())

    envelope = await obligations.coroutine(  # type: ignore[call-arg]
        party="Acme Corp",
        runtime=_runtime(),
    )

    evidence_service = EvidenceService(repo=object())
    for match in envelope["matches"]:
        # The cited clause's document is inside the run's authorized scope.
        assert match["document_id"] in ALLOWED_DOCS
        assert match["evidence_locator"].startswith("portfolio:")
        response = await evidence_service.read_evidence_by_id(
            evidence_id=match["evidence_locator"],
            principal_id="user-1",
            tenant_id="tenant-1",
            allowed_doc_ids=ALLOWED_DOCS,
            thread_id="thread-1",
            run_id="run-1",
        )
        # The locator is honestly not a handle: it resolves to nothing. It
        # must never come back as an out-of-scope ("unauthorized") read.
        assert response.status == "invalid_handle"
