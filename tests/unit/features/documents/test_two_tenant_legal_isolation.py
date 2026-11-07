"""Permanent two-tenant isolation suite over every legal tool (task 2.6).

Fixture: two tenants — personal:alice and personal:bob — each ingested the
SAME identical statute chunk (same text, same section identity), as would
happen with any widely-cited statute. The suite drives every legal tool in
the registry twice, once under each tenant's server-derived scope, and
fails CI the moment any tool's outbound scope (Graphiti group selection,
SQL binds, or evidence-service scope) references the other tenant, the
other tenant's documents, or issues a corpus read without an allowlist.

An unauthorized document id must be denied before any corpus read, and a
missing allowlist must fail closed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from langchain.tools import ToolRuntime
from sqlalchemy import TextClause
from sqlalchemy.dialects import postgresql

from app.features.documents.api.http.dto import (
    EvidenceReadResponse,
    EvidenceSearchResponse,
    IngestionStateResponse,
)
from app.features.documents.application.tools.evidence import make_evidence_tools
from app.features.documents.application.tools.get_obligation_chain import (
    make_get_obligation_chain_tool,
)
from app.features.documents.application.tools.query_knowledge_graph import (
    make_query_knowledge_graph_tool,
)
from app.features.documents.application.tools.retrieve_statute_section import (
    make_retrieve_statute_section_tool,
)
from app.features.documents.application.tools.search_legal_precedents import (
    make_search_legal_precedents_tool,
)
from app.resilience.idempotency import IdempotencyGuard
from app.retrieval.graphiti.schemas import GraphitiSearchOutcome

if TYPE_CHECKING:
    from langchain_core.tools.base import BaseTool

if TYPE_CHECKING:
    from typing import Any

_TENANT_A = "personal:alice"
_TENANT_B = "personal:bob"
_DOCS_A = ("aaaaaaaa-1111-1111-1111-111111111111",)
_DOCS_B = ("bbbbbbbb-2222-2222-2222-222222222222",)
_FOREIGN_DOC = "bbbbbbbb-2222-2222-2222-222222222222"
_IDENTICAL_STATUTE = (
    "Section 73: When a contract has been broken, the party who suffers from "
    "the breach is entitled to receive compensation for the loss."
)


class _CaptureGraphiti:
    """Graphiti service double recording every scoped search call.

    Mirrors the real service boundary: an empty allowlist fails closed
    without issuing a search, so a recorded call always carries a scope.
    """

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def _record(self, method: str, kwargs: dict[str, Any]) -> GraphitiSearchOutcome:
        if not tuple(kwargs.get("allowed_doc_ids") or ()):
            return GraphitiSearchOutcome()
        self.calls.append((method, kwargs))
        return GraphitiSearchOutcome()

    async def search_for_precedent_chains(self, **kwargs: Any) -> GraphitiSearchOutcome:
        return self._record("search_for_precedent_chains", kwargs)

    async def search_for_risk_context(self, **kwargs: Any) -> GraphitiSearchOutcome:
        return self._record("search_for_risk_context", kwargs)

    async def get_obligation_chain(self, **kwargs: Any) -> GraphitiSearchOutcome:
        return self._record("get_obligation_chain", kwargs)


class _CaptureConnection:
    def __init__(self, calls: list[tuple[Any, Any]]) -> None:
        self._calls = calls

    async def execute(self, statement: Any, params: Any = None) -> Any:
        self._calls.append((statement, params))

        class _Rows:
            def __init__(self, rows: list[tuple[Any, ...]]) -> None:
                self._rows = rows

            def fetchall(self) -> list[tuple[Any, ...]]:
                return self._rows

            def fetchone(self) -> tuple[Any, ...] | None:
                return self._rows[0] if self._rows else None

        if isinstance(statement, TextClause):
            return _Rows([])
        return _Rows(
            [
                (
                    "chunk-a",
                    "Indian Contract Act 1872",
                    "73",
                    _IDENTICAL_STATUTE,
                    _DOCS_A[0],
                    1872,
                )
            ]
        )

    async def __aenter__(self) -> _CaptureConnection:
        return self

    async def __aexit__(self, *_exc: object) -> bool:
        return False


class _CaptureEngine:
    def __init__(self) -> None:
        self.calls: list[tuple[Any, Any]] = []

    def connect(self) -> _CaptureConnection:
        return _CaptureConnection(self.calls)


class _CaptureEvidenceService:
    """EvidenceService double recording the scope each call carries.

    Mirrors the real fail-closed service: an empty allowlist is
    `unauthorized` before any scope read or issuance happens.
    """

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def _fail_closed(self, kwargs: dict[str, Any]) -> bool:
        return not tuple(kwargs.get("allowed_doc_ids") or ())

    async def search_evidence(self, **kwargs: Any) -> EvidenceSearchResponse:
        if self._fail_closed(kwargs):
            return EvidenceSearchResponse(
                status="unauthorized",
                evidence_policy="mechanically_verified",
                message="No authorized documents were supplied for this analysis.",
            )
        self.calls.append(kwargs)
        return EvidenceSearchResponse(status="no_evidence", evidence_policy="mechanically_verified")

    async def read_evidence_by_id(self, **kwargs: Any) -> EvidenceReadResponse:
        if self._fail_closed(kwargs):
            return EvidenceReadResponse(
                status="unauthorized",
                evidence_policy="mechanically_verified",
                message="No authorized documents were supplied for this analysis.",
            )
        self.calls.append(kwargs)
        return EvidenceReadResponse(status="no_evidence", evidence_policy="mechanically_verified")

    async def get_ingestion_state(self, **kwargs: Any) -> IngestionStateResponse:
        if self._fail_closed(kwargs):
            return IngestionStateResponse(
                status="unauthorized",
                evidence_policy="mechanically_verified",
                message="No authorized documents were supplied for this analysis.",
            )
        self.calls.append(kwargs)
        return IngestionStateResponse(status="ok", evidence_policy="mechanically_verified")


def _null_idempotency() -> IdempotencyGuard:
    from unittest.mock import MagicMock

    redis = MagicMock()

    async def _none(*_a: Any, **_k: Any) -> None:
        return None

    redis.get = _none
    redis.set = _none

    class _Unreachable:
        def connect(self) -> None:
            message = "no database in unit test"
            raise RuntimeError(message)

    return IdempotencyGuard(redis=redis, db_engine=_Unreachable())


def _runtime(tenant_id: str, allowed_doc_ids: tuple[str, ...]) -> ToolRuntime:
    return ToolRuntime(
        state={},
        context={
            "user_id": tenant_id.removeprefix("personal:"),
            "tenant_id": tenant_id,
            "allowed_doc_ids": allowed_doc_ids,
            "allowed_operations": ("evidence.search", "evidence.read", "document.read"),
            "thread_id": "thread-1",
            "current_step": "1",
            "run_id": "run-1",
            "checkpoint_id": "cp-1",
            "document_versions": (),
            "document_contracts": (),
            "capabilities": (),
        },
        config={},
        stream_writer=None,
        tool_call_id="test-call",
        store=None,
    )


def _all_legal_tools() -> dict[str, tuple[BaseTool, Any]]:
    """Every registered legal tool wired to capture doubles."""
    graphiti = _CaptureGraphiti()
    engine = _CaptureEngine()
    evidence_service = _CaptureEvidenceService()
    idempotency = _null_idempotency()
    evidence_tools = make_evidence_tools(evidence_service)
    return {
        "search_legal_precedents": (
            make_search_legal_precedents_tool(graphiti, engine, idempotency),
            (graphiti, engine),
        ),
        "retrieve_statute_section": (
            make_retrieve_statute_section_tool(engine, idempotency),
            (engine,),
        ),
        "query_knowledge_graph": (
            make_query_knowledge_graph_tool(graphiti, idempotency),
            (graphiti,),
        ),
        "get_obligation_chain": (
            make_get_obligation_chain_tool(graphiti, idempotency),
            (graphiti,),
        ),
        "search_evidence": (evidence_tools[0], (evidence_service,)),
        "read_evidence_by_id": (evidence_tools[1], (evidence_service,)),
    }


_TOOL_ARGS: dict[str, dict[str, Any]] = {
    "search_legal_precedents": {
        "query": "compensation for breach",
        "clause_id": "c-1",
        "jurisdiction": "India",
    },
    "retrieve_statute_section": {
        "act_name": "Indian Contract Act 1872",
        "section_ref": "73",
        "jurisdiction": "India",
    },
    "query_knowledge_graph": {"query": "indemnity obligations"},
    "get_obligation_chain": {"entity_name": "Party A"},
    "search_evidence": {"query": "compensation for breach"},
    "read_evidence_by_id": {"evidence_id": "evh-1"},
}


def _graphiti_scope_violations(
    graphiti: _CaptureGraphiti, *, tenant: str, docs: tuple[str, ...]
) -> list[str]:
    problems: list[str] = []
    for method, kwargs in graphiti.calls:
        if kwargs.get("tenant_id") != tenant:
            problems.append(f"{method} tenant_id={kwargs.get('tenant_id')!r}")
        if tuple(kwargs.get("allowed_doc_ids") or ()) != docs:
            problems.append(f"{method} allowed_doc_ids={kwargs.get('allowed_doc_ids')!r}")
    return problems


def _engine_scope_violations(
    engine: _CaptureEngine, *, tenant: str, docs: tuple[str, ...]
) -> list[str]:
    problems: list[str] = []
    for statement, params in engine.calls:
        if isinstance(statement, TextClause):
            binds = dict(params or {})
        else:
            binds = dict(statement.compile(dialect=postgresql.dialect()).params)
        values = list(binds.values())
        if tenant not in values:
            problems.append("statement without the caller's tenant bind")
        if docs not in [tuple(v) for v in values if isinstance(v, (list, tuple))]:
            problems.append("statement without the caller's allowlist bind")
        other = _DOCS_B if tenant == _TENANT_A else _DOCS_A
        if tuple(other) in [tuple(v) for v in values if isinstance(v, (list, tuple))]:
            problems.append("statement binds the other tenant's documents")
    return problems


def _evidence_scope_violations(
    service: _CaptureEvidenceService, *, tenant: str, docs: tuple[str, ...]
) -> list[str]:
    problems: list[str] = []
    for kwargs in service.calls:
        if kwargs.get("tenant_id") != tenant:
            problems.append(f"evidence scope tenant_id={kwargs.get('tenant_id')!r}")
        if tuple(kwargs.get("allowed_doc_ids") or ()) != docs:
            problems.append(f"evidence scope allowed_doc_ids={kwargs.get('allowed_doc_ids')!r}")
    return problems


@pytest.mark.parametrize("tool_name", sorted(_TOOL_ARGS))
@pytest.mark.asyncio
async def test_two_tenants_with_identical_statute_chunks_cannot_read_each_other(
    tool_name: str,
) -> None:
    tools = _all_legal_tools()
    tool, infra = tools[tool_name]

    for tenant, docs, other_tenant, other_docs in (
        (_TENANT_A, _DOCS_A, _TENANT_B, _DOCS_B),
        (_TENANT_B, _DOCS_B, _TENANT_A, _DOCS_A),
    ):
        # Reset captures per caller.
        for component in infra:
            component.calls.clear()
        result = await tool.ainvoke({**_TOOL_ARGS[tool_name], "runtime": _runtime(tenant, docs)})
        assert not isinstance(result, str), f"{tool_name} rendered its envelope into prose"
        assert result.get("unauthorized") is not True, f"{tool_name} denied an authorized caller"

        tenant_problems: list[str] = []
        for component in infra:
            if isinstance(component, _CaptureGraphiti):
                tenant_problems.extend(
                    _graphiti_scope_violations(component, tenant=tenant, docs=docs)
                )
            elif isinstance(component, _CaptureEngine):
                tenant_problems.extend(
                    _engine_scope_violations(component, tenant=tenant, docs=docs)
                )
            elif isinstance(component, _CaptureEvidenceService):
                tenant_problems.extend(
                    _evidence_scope_violations(component, tenant=tenant, docs=docs)
                )
        assert not tenant_problems, (
            f"{tool_name} leaked cross-tenant scope for {tenant}: {tenant_problems}"
        )

        # A chunk with identical text exists in the other tenant; nothing in
        # the result may attribute it to this caller's scope.
        blob = repr(result)
        assert other_tenant not in blob, f"{tool_name} echoed the other tenant id"
        assert other_docs[0] not in blob, f"{tool_name} echoed the other tenant's document id"


@pytest.mark.parametrize("tool_name", sorted(_TOOL_ARGS))
@pytest.mark.asyncio
async def test_legal_tool_without_allowlist_fails_closed(tool_name: str) -> None:
    """An empty server-issued allowlist is a deny, never a tenant-wide read."""
    tools = _all_legal_tools()
    tool, infra = tools[tool_name]
    result = await tool.ainvoke({**_TOOL_ARGS[tool_name], "runtime": _runtime(_TENANT_A, ())})
    assert not isinstance(result, str)
    blob = repr(result)
    # No authority content may come back, and the other tenant's identity is
    # never echoed.
    assert _IDENTICAL_STATUTE not in blob, f"{tool_name} returned corpus text with no allowlist"
    assert _TENANT_B not in blob, f"{tool_name} echoed the other tenant id"
    assert _DOCS_B[0] not in blob, f"{tool_name} echoed the other tenant's document id"
    # The evidence contract fails closed with an explicit status; the
    # ToolResult tools must not report a source they could not have read.
    if tool_name in {"search_evidence", "read_evidence_by_id"}:
        assert result["status"] == "unauthorized"
    for component in infra:
        if isinstance(component, _CaptureGraphiti):
            assert component.calls == [], f"{tool_name} issued a graphiti read with no allowlist"
        if isinstance(component, _CaptureEngine):
            assert component.calls == [], f"{tool_name} issued a corpus read with no allowlist"
        if isinstance(component, _CaptureEvidenceService):
            assert component.calls == [], f"{tool_name} issued an evidence read with no allowlist"


@pytest.mark.parametrize("tool_name", ["search_legal_precedents", "retrieve_statute_section"])
@pytest.mark.asyncio
async def test_precedent_sql_legs_only_run_under_a_live_allowlist(tool_name: str) -> None:
    """The Postgres-backed legs execute their statements only with both the
    tenant predicate and the allowlist predicate bound."""
    tools = _all_legal_tools()
    tool, infra = tools[tool_name]
    engine = next(c for c in infra if isinstance(c, _CaptureEngine))
    await tool.ainvoke({**_TOOL_ARGS[tool_name], "runtime": _runtime(_TENANT_A, _DOCS_A)})
    assert engine.calls, "the fixture must reach the Postgres leg"
    problems = _engine_scope_violations(engine, tenant=_TENANT_A, docs=_DOCS_A)
    assert not problems, f"{tool_name} Postgres leg scope violations: {problems}"


def _runtime_model_cannot_widen_scope() -> ToolRuntime:
    """The model-side state tries to grant tenant B; context denies it."""
    runtime = _runtime(_TENANT_A, _DOCS_A)
    runtime.state["tenant_id"] = _TENANT_B  # type: ignore[index]
    runtime.state["allowed_doc_ids"] = _DOCS_B  # type: ignore[index]
    return runtime


@pytest.mark.asyncio
async def test_model_state_cannot_widen_the_tenant_boundary() -> None:
    """trusted_tool_context takes identity/capability authority from
    ToolRuntime.context only; specialist state is model-controlled and is
    never a fallback for grants."""
    from app.features.documents.application.tools.authorization import trusted_tool_context

    authz = trusted_tool_context(_runtime_model_cannot_widen_scope())
    assert authz.tenant_id == _TENANT_A
    assert authz.allowed_doc_ids == _DOCS_A
    with pytest.raises(PermissionError):
        authz.require_document(_FOREIGN_DOC)
