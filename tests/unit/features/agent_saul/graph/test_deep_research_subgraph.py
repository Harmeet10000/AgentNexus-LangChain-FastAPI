"""The bounded deep-research subgraph and its single tool wrapper (D5).

Covers the openspec ``deep-research-subgraph`` acceptance matrix: server-owned
context only (V6), typed budget exhaustion, run-scoped handles, no interrupts
inside the subgraph, and the deep_research role's two-tool projection.
"""

from __future__ import annotations

import json
from typing import (  # noqa: TC003 — TypedDict is runtime, Any appears in annotations only
    TYPE_CHECKING,
    Any,
    TypedDict,
)

import pytest
from langchain.tools import ToolRuntime
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from pydantic import ValidationError
from returns.result import Success

from app.ai.tools.registry import (
    deep_research_role_projection,
    get_tool_registry,
    register_deep_research_tool,
    register_default_tools,
    register_portfolio_traversal_tools,
)
from app.features.documents.application.tools.portfolio import (
    make_portfolio_traversal_tools,
)
from app.features.agent_saul.domain.state import (
    OrchestratorAction,
    OrchestratorActionType,
    PlanActionType,
    PlanStep,
    WorkflowStatus,
)
from app.features.agent_saul.graph.deep_research import (
    DeepResearchInput,
    compile_deep_research_subgraph,
    make_deep_research_tool,
)
from app.features.agent_saul.graph.nodes import make_deep_research_node, make_orchestrator_node
from app.features.documents.api.http.dto import (
    EvidenceCoverage,
    EvidenceReadItem,
    EvidenceReadResponse,
    EvidenceRef,
    EvidenceSearchItem,
    EvidenceSearchResponse,
)
from app.features.documents.application.evidence import service as evidence_module
from app.features.documents.application.evidence.service import (
    EvidenceService,
    InMemoryEvidenceHandleStore,
)

if TYPE_CHECKING:
    from langchain_core.tools import BaseTool
    from langgraph.graph.state import CompiledStateGraph

pytestmark = pytest.mark.unit

DOC = "doc-1"
RETRIEVAL_METHOD = "hybrid_rrf"
REAL_DOC = "00000000-0000-0000-0000-000000000001"


# ---------------------------------------------------------------------------
# Fixtures and doubles
# ---------------------------------------------------------------------------


def _search_response(
    *,
    status: str = "ok",
    evidence_id: str = "evh-1",
    run_id: str | None = None,
    coverage: EvidenceCoverage | None = None,
    message: str | None = None,
) -> EvidenceSearchResponse:
    items: list[EvidenceSearchItem] = []
    if status == "ok":
        ref = EvidenceRef(
            evidence_id=evidence_id,
            chunk_id="chunk-1",
            document_id=DOC,
            document_version=1,
            page_no=3,
            graph_indexed=False,
            retrieval_method=RETRIEVAL_METHOD,
            score=0.9,
            rank=1,
            run_id=run_id,
        )
        items.append(
            EvidenceSearchItem(
                evidence=ref,
                document_title="Uploaded authority",
                snippet="search snippet that must stay out of graph state",
                chunk_index=0,
                chunk_kind="generic",
            )
        )
    return EvidenceSearchResponse(
        status=status,  # type: ignore[arg-type]
        evidence_policy="mechanically_verified",
        items=items,
        coverage=coverage or EvidenceCoverage(documents_requested=1, documents_ready=1),
        message=message,
    )


class _RecordingSearchService:
    """Search-only double that records the exact scope it was called with."""

    def __init__(self, response: EvidenceSearchResponse) -> None:
        self.response = response
        self.calls: list[dict[str, Any]] = []

    async def search_evidence(self, **kwargs: Any) -> EvidenceSearchResponse:
        self.calls.append(kwargs)
        return self.response


class _ReadingService:
    """Search issues a handle; read resolves it with bounded content."""

    async def search_evidence(self, **kwargs: Any) -> EvidenceSearchResponse:
        return _search_response()

    async def read_evidence_by_id(self, **kwargs: Any) -> EvidenceReadResponse:
        return EvidenceReadResponse(
            status="ok",
            evidence_policy="mechanically_verified",
            items=[
                EvidenceReadItem(
                    evidence=EvidenceRef(
                        evidence_id="evh-1",
                        chunk_id="chunk-1",
                        document_id=DOC,
                        document_version=1,
                        page_no=3,
                        graph_indexed=False,
                        retrieval_method="handle_read",
                        score=0.0,
                        rank=0,
                    ),
                    document_title="Uploaded authority",
                    content="Termination requires thirty days notice.",
                )
            ],
        )


class _RefusingReadService:
    """Search issues a handle; read refuses it (the run-scoping path)."""

    def __init__(self) -> None:
        self.read_calls: list[dict[str, Any]] = []

    async def search_evidence(self, **kwargs: Any) -> EvidenceSearchResponse:
        return _search_response()

    async def read_evidence_by_id(self, **kwargs: Any) -> EvidenceReadResponse:
        self.read_calls.append(kwargs)
        return EvidenceReadResponse(
            status="unauthorized",
            evidence_policy="mechanically_verified",
            message="Evidence handle is outside the current authorization scope.",
        )


def _runtime(
    *,
    run_id: str = "run-1",
    thread_id: str = "thread-1",
    state: dict[str, Any] | None = None,
    **context_overrides: Any,
) -> ToolRuntime:
    context: dict[str, Any] = {
        "user_id": "user-1",
        "tenant_id": "tenant-1",
        "allowed_doc_ids": (DOC,),
        "thread_id": thread_id,
        "run_id": run_id,
    }
    context.update(context_overrides)
    return ToolRuntime(
        state=state or {},
        context=context,
        config={},
        stream_writer=None,
        tool_call_id="test-call",
        store=None,
    )


def _scoped_tool(
    service: Any,
    *,
    max_steps: int | None = None,
) -> BaseTool:
    kwargs: dict[str, Any] = {}
    if max_steps is not None:
        kwargs["max_steps"] = max_steps
    return make_deep_research_tool(service, **kwargs)


# ---------------------------------------------------------------------------
# Requirement: entered only through one tool with server-owned context
# ---------------------------------------------------------------------------


def test_wrapper_schema_exposes_exactly_objective_and_focus() -> None:
    tool = _scoped_tool(_RecordingSearchService(_search_response()))

    assert set(DeepResearchInput.model_fields) == {"objective", "focus"}
    schema = tool.args_schema.model_json_schema()
    properties = set(schema.get("properties", {}))
    assert properties == {"objective", "focus"}
    # No scope-like argument is model-visible.
    assert not properties & {"tenant_id", "allowed_doc_ids", "run_id", "user_id"}


def test_wrapper_rejects_model_supplied_scope_arguments() -> None:
    with pytest.raises(ValidationError):
        DeepResearchInput.model_validate(
            {
                "objective": "termination risk",
                "tenant_id": "tenant-evil",
                "allowed_doc_ids": ["doc-evil"],
                "run_id": "run-evil",
            }
        )


async def test_wrapper_denies_without_runtime_context() -> None:
    service = _RecordingSearchService(_search_response())
    tool = _scoped_tool(service)

    envelope = await tool.coroutine(objective="termination obligations")  # type: ignore[call-arg]

    assert envelope["status"] == "refused"
    assert "authorization" in envelope["message"]
    assert envelope["evidence"] == []
    assert service.calls == []  # the subgraph never started


async def test_wrapper_denies_without_tenant_scope() -> None:
    service = _RecordingSearchService(_search_response())
    tool = _scoped_tool(service)

    envelope = await tool.coroutine(  # type: ignore[call-arg]
        objective="termination obligations",
        runtime=_runtime(tenant_id=None),
    )

    assert envelope["status"] == "refused"
    assert "tenant" in envelope["message"]
    assert service.calls == []


async def test_model_supplied_scope_never_reaches_subgraph() -> None:
    """V6: hostile state and model args cannot define scope; context does."""
    service = _RecordingSearchService(_search_response())
    tool = _scoped_tool(service)

    envelope = await tool.coroutine(  # type: ignore[call-arg]
        objective="termination obligations",
        runtime=_runtime(
            # Graph state that a model or a stale checkpoint could influence:
            state={
                "user_id": "attacker",
                "tenant_id": "tenant-evil",
                "allowed_doc_ids": ["doc-evil"],
                "run_id": "run-evil",
            },
        ),
    )

    assert envelope["status"] == "completed"
    assert len(service.calls) == 1
    called = service.calls[0]
    # Only the server-owned runtime context reached the subgraph.
    assert called["principal_id"] == "user-1"
    assert called["tenant_id"] == "tenant-1"
    assert called["allowed_doc_ids"] == (DOC,)
    assert called["run_id"] == "run-1"
    # And the model-visible schema never offered scope as an argument.
    assert set(DeepResearchInput.model_fields) == {"objective", "focus"}


async def test_run_id_is_server_owned_not_state_owned() -> None:
    """Decision 4: the handle checkpoint marker comes from runtime context."""
    service = _RecordingSearchService(_search_response())
    tool = _scoped_tool(service)

    await tool.coroutine(  # type: ignore[call-arg]
        objective="termination obligations",
        runtime=_runtime(run_id="run-42", state={"evidence_checkpoint_id": "forged"}),
    )

    called = service.calls[0]
    assert called["run_id"] == "run-42"
    assert called["checkpoint_id"] == "run:run-42"


# ---------------------------------------------------------------------------
# Requirement: hard-bounded loop with typed terminal statuses
# ---------------------------------------------------------------------------


async def test_full_run_completes_with_one_terminal_envelope() -> None:
    service = _RecordingSearchService(_search_response())
    tool = _scoped_tool(service)

    envelope = await tool.coroutine(  # type: ignore[call-arg]
        objective="termination obligations",
        runtime=_runtime(),
    )

    assert envelope["status"] == "completed"
    assert isinstance(envelope["summary"], str)
    assert envelope["summary"]
    assert [ref["evidence_id"] for ref in envelope["evidence"]] == ["evh-1"]
    # Snippets never travel with the envelope.
    assert all("snippet" not in json.dumps(ref) for ref in envelope["evidence"])
    assert envelope["coverage"]["documents_requested"] == 1


async def test_subgraph_has_no_internal_checkpointer_and_no_interrupts() -> None:
    subgraph: CompiledStateGraph = compile_deep_research_subgraph(
        _RecordingSearchService(_search_response())
    )
    assert subgraph.checkpointer is None

    final = await subgraph.ainvoke(
        {
            "objective": "termination obligations",
            "focus": None,
            "user_id": "user-1",
            "tenant_id": "tenant-1",
            "allowed_doc_ids": (DOC,),
            "thread_id": "thread-1",
            "run_id": "run-1",
            "evidence_checkpoint_id": "run:run-1",
            "queries": [],
            "pending_handles": [],
            "evidence": [],
            "read_notes": [],
            "refusals": [],
            "coverage": {},
            "steps_used": 0,
            "max_steps": 4,
            "max_reads": 4,
            "search_limit": 5,
            "status": "running",
            "summary": None,
            "message": None,
        }
    )
    # A full run emits zero interrupts and lands on exactly one terminal status.
    assert "__interrupt__" not in final
    assert final["status"] == "completed"


async def test_budget_exhaustion_returns_partial_handles_without_answer() -> None:
    service = _ReadingService()
    tool = _scoped_tool(service, max_steps=2)

    envelope = await tool.coroutine(  # type: ignore[call-arg]
        objective="termination obligations",
        runtime=_runtime(),
    )

    assert envelope["status"] == "budget_exhausted"
    # Partial handles are preserved…
    assert [ref["evidence_id"] for ref in envelope["evidence"]] == ["evh-1"]
    # …but no synthesis is presented as a complete answer.
    assert envelope["summary"] is None
    assert "budget" in envelope["message"]


async def test_no_evidence_is_a_status_not_an_empty_success() -> None:
    service = _RecordingSearchService(
        _search_response(
            status="no_evidence",
            message="No uploaded authority matched.",
            coverage=EvidenceCoverage(documents_requested=1, documents_ready=1),
        )
    )
    tool = _scoped_tool(service)

    envelope = await tool.coroutine(  # type: ignore[call-arg]
        objective="termination obligations",
        runtime=_runtime(),
    )

    assert envelope["status"] == "no_evidence"
    assert envelope["evidence"] == []
    assert envelope["summary"] is None


async def test_unservable_scope_is_ingestion_incomplete() -> None:
    service = _RecordingSearchService(
        _search_response(
            status="index_not_ready",
            message="One or more authorized documents are not ready for evidence search.",
            coverage=EvidenceCoverage(
                documents_requested=1,
                documents_ready=0,
                documents_not_ready=1,
            ),
        )
    )
    tool = _scoped_tool(service)

    envelope = await tool.coroutine(  # type: ignore[call-arg]
        objective="termination obligations",
        runtime=_runtime(),
    )

    assert envelope["status"] == "ingestion_incomplete"
    assert envelope["evidence"] == []
    assert envelope["summary"] is None


# ---------------------------------------------------------------------------
# Requirement: run-scoped handles
# ---------------------------------------------------------------------------


async def test_refused_handle_read_names_the_run_scoping_rule() -> None:
    service = _RefusingReadService()
    tool = _scoped_tool(service)

    envelope = await tool.coroutine(  # type: ignore[call-arg]
        objective="termination obligations",
        runtime=_runtime(),
    )

    assert envelope["status"] == "ingestion_incomplete"
    assert envelope["summary"] is None
    assert service.read_calls[0]["run_id"] == "run-1"
    # The refusal names the run-scoping rule instead of vanishing.
    assert "run" in (envelope["message"] or "")
    assert "valid only within the run" in (envelope["message"] or "")


class _Repo:
    def __init__(self) -> None:
        self.row: dict[str, object] = {
            "chunk_id": "00000000-0000-0000-0000-000000000011",
            "document_id": REAL_DOC,
            "title": "contract.pdf",
            "content": "Termination requires thirty days notice.",
            "preamble": "",
            "document_version": 1,
            "chunk_index": 0,
            "chunk_kind": "legal_contract",
            "clause_type": "termination",
            "page_no": 4,
            "chunk_metadata": {"pipeline_version": "p1"},
            "document_metadata": {},
            "quality_warnings": [],
            "graphiti_verified": True,
        }
        self.status: dict[str, object] = {
            "document_id": REAL_DOC,
            "status": "completed",
            "chunk_count": 1,
            "verified_chunk_count": 1,
            "document_version": 1,
            "document_metadata": {"pipeline_version": "p1"},
            "warnings": [],
            "extraction_incomplete": False,
        }

    async def fetch_chunks_by_ids(self, chunk_ids: list[str], **_: object) -> Any:
        if self.row["chunk_id"] not in chunk_ids:
            return Success({})
        return Success({str(self.row["chunk_id"]): dict(self.row)})

    async def fetch_status(self, **_: object) -> Any:
        return Success(dict(self.status))


@pytest.fixture
def real_service(monkeypatch: pytest.MonkeyPatch) -> EvidenceService:
    repo = _Repo()

    async def _embed(*_: object, **__: object) -> list[float]:
        return [0.0]

    async def _retrieve(**_: object) -> Any:
        return Success(
            (
                [
                    _ranked(
                        chunk_id=str(repo.row["chunk_id"]),
                    )
                ],
                {str(repo.row["chunk_id"]): dict(repo.row)},
            )
        )

    monkeypatch.setattr(evidence_module, "embed_text", _embed)
    monkeypatch.setattr(evidence_module, "retrieve_fused", _retrieve)
    return EvidenceService(repo=repo, handle_store=InMemoryEvidenceHandleStore())


def _ranked(*, chunk_id: str) -> Any:
    from app.retrieval.fusion import RankedChunk

    return RankedChunk(chunk_id=chunk_id, score=0.9, rank=1)


class _RunScopedReadService:
    """Run-2 service: search replays a run-1 handle, read uses the real store."""

    def __init__(self, real: EvidenceService, stale_handle: str) -> None:
        self._real = real
        self._stale_handle = stale_handle
        self.read_calls: list[dict[str, Any]] = []

    async def search_evidence(self, **kwargs: Any) -> EvidenceSearchResponse:
        return _search_response(evidence_id=self._stale_handle)

    async def read_evidence_by_id(self, **kwargs: Any) -> Any:
        self.read_calls.append(kwargs)
        return await self._real.read_evidence_by_id(**kwargs)


async def test_handle_from_run_n_fails_membership_in_run_n_plus_1(
    real_service: EvidenceService,
) -> None:
    run_one = _scoped_tool(real_service)
    first = await run_one.coroutine(  # type: ignore[call-arg]
        objective="termination obligations",
        runtime=_runtime(run_id="run-1", allowed_doc_ids=(REAL_DOC,)),
    )
    assert first["status"] == "completed"
    stale_handle = first["evidence"][0]["evidence_id"]

    # The same service, asked directly in a later run on the same thread:
    # handle-membership fails because handles are run-scoped.
    direct = await real_service.read_evidence_by_id(
        evidence_id=stale_handle,
        principal_id="user-1",
        tenant_id="tenant-1",
        allowed_doc_ids=(REAL_DOC,),
        thread_id="thread-1",
        run_id="run-2",
    )
    assert direct.status == "unauthorized"

    # And through the subgraph's own read stage in run 2:
    run_two = _scoped_tool(_RunScopedReadService(real_service, stale_handle))
    second = await run_two.coroutine(  # type: ignore[call-arg]
        objective="termination obligations",
        runtime=_runtime(run_id="run-2", allowed_doc_ids=(REAL_DOC,)),
    )
    assert second["status"] == "ingestion_incomplete"
    assert "valid only within the run" in (second["message"] or "")


# ---------------------------------------------------------------------------
# Requirement: the projection stays at one visible tool
# ---------------------------------------------------------------------------


def _register_startup_catalog() -> None:
    """Mirror graph-factory startup: defaults plus portfolio seats, then D5."""

    register_default_tools()
    register_portfolio_traversal_tools(
        make_portfolio_traversal_tools(None),
        validate=False,
    )


def test_registry_registration_carries_read_only_authority_tags() -> None:
    _register_startup_catalog()
    tool = _scoped_tool(_RecordingSearchService(_search_response()))
    register_deep_research_tool(tool)
    registry = get_tool_registry()

    assert registry.get("deep_research") is tool
    matches = {t.name for t in registry.by_tags("capability:deep-research", "side-effect:read")}
    assert "deep_research" in matches
    assert "capability:deep-research" in registry.tags()


def test_deep_research_role_projection_is_exactly_transfer_plus_research() -> None:
    _register_startup_catalog()
    register_deep_research_tool(_scoped_tool(_RecordingSearchService(_search_response())))

    projection = deep_research_role_projection()
    assert [tool.name for tool in projection] == [
        "transfer_to_deep_research",
        "deep_research",
    ]


# ---------------------------------------------------------------------------
# Node mapping (envelope → graph state)
# ---------------------------------------------------------------------------


async def test_node_maps_budget_exhaustion_to_typed_failure_with_partial_handles() -> None:
    tool = _scoped_tool(_ReadingService(), max_steps=2)
    node = make_deep_research_node(
        deep_research_tool=tool,
    )
    state = _node_state()

    result = await node(state)  # type: ignore[arg-type]

    assert result["status"] is WorkflowStatus.FAILED
    codes = [error.code for error in result["errors"]]
    assert "BUDGET_EXHAUSTED" in codes
    # Partial handles survive in the registry for downstream verification.
    assert result["evidence_registry"]["evh-1"].document_id == DOC


def _node_state(*, run_id: str = "run-1") -> dict[str, Any]:
    return {
        "user_id": "user-1",
        "tenant_id": "tenant-1",
        "thread_id": "thread-1",
        "correlation_id": "correlation-1",
        "run_id": run_id,
        "doc_id": DOC,
        "allowed_doc_ids": [DOC],
        "plan": [
            PlanStep(
                step_id="research-1",
                action=PlanActionType.SEARCH_PRECEDENTS,
                description="termination obligations",
            )
        ],
        "current_step": 0,
        "messages": [],
        "evidence_registry": {},
        "coverage": {},
        "errors": [],
        "status": WorkflowStatus.PLAN_APPROVED,
    }


# ---------------------------------------------------------------------------
# Plan-level execution: orchestrator → deep_research → bounded envelope
# ---------------------------------------------------------------------------


class _RunState(TypedDict, total=False):
    user_id: str
    tenant_id: str
    thread_id: str
    correlation_id: str
    run_id: str
    doc_id: str
    allowed_doc_ids: list[str]
    plan: list[PlanStep]
    current_step: int
    plan_approved: bool
    messages: list[Any]
    working_memory: dict[str, Any]
    ingestion_state: list[Any]
    risk_analysis: Any
    compliance_result: Any
    grounding: Any
    human_review: Any
    orchestrator_action: Any
    evidence_registry: dict[str, Any]
    coverage: dict[str, int]
    errors: list[Any]
    deep_research_results: str | None
    budget_ledger: Any
    status: WorkflowStatus


class _ScriptedOrchestrator:
    """Delegates to deep_research on the first call, then synthesizes."""

    def __init__(self) -> None:
        self.calls = 0

    async def ainvoke(self, _messages: list[Any]) -> OrchestratorAction:
        self.calls += 1
        if self.calls == 1:
            return OrchestratorAction(
                action_type=OrchestratorActionType.CONTINUE,
                target_node="deep_research",
                reflection="Research the termination obligations first.",
            )
        return OrchestratorAction(
            action_type=OrchestratorActionType.SYNTHESIZE,
            target_node="finalization",
            reflection="Research complete.",
        )


def _route_after_orchestrator(state: dict[str, Any]) -> str:
    action = state.get("orchestrator_action")
    if (
        action is not None
        and action.action_type == OrchestratorActionType.CONTINUE
        and action.target_node == "deep_research"
    ):
        return "deep_research"
    return END


async def test_plan_execution_reaches_deep_research_and_lands_envelope() -> None:
    _register_startup_catalog()
    register_deep_research_tool(_scoped_tool(_RecordingSearchService(_search_response())))
    # Toolset identity holds at plan level: exactly transfer + deep_research.
    assert [tool.name for tool in deep_research_role_projection()] == [
        "transfer_to_deep_research",
        "deep_research",
    ]

    tool = _scoped_tool(_RecordingSearchService(_search_response()))
    orchestrator = make_orchestrator_node(_ScriptedOrchestrator())  # type: ignore[arg-type]
    research = make_deep_research_node(deep_research_tool=tool)

    # No state annotations here: LangGraph must not merge LegalAgentState's
    # channels into this test graph's schema.
    async def orchestrator_node(state: Any) -> dict[str, Any]:
        return await orchestrator(state)

    async def deep_research_node(state: Any) -> dict[str, Any]:
        return await research(state)

    graph: StateGraph = StateGraph(_RunState)
    graph.add_node("orchestrator", orchestrator_node)
    graph.add_node("deep_research", deep_research_node)
    graph.add_edge(START, "orchestrator")
    graph.add_conditional_edges("orchestrator", _route_after_orchestrator)
    graph.add_edge("deep_research", END)

    compiled = graph.compile(checkpointer=InMemorySaver())
    final = await compiled.ainvoke(
        config={"configurable": {"thread_id": "thread-1"}},
        input={
            "user_id": "user-1",
            "tenant_id": "tenant-1",
            "thread_id": "thread-1",
            "correlation_id": "correlation-1",
            "run_id": "run-1",
            "doc_id": DOC,
            "allowed_doc_ids": [DOC],
            # The orchestrator advances current_step before delegating, so the
            # research step sits at index 1 exactly as the planner emits it.
            "plan": [
                PlanStep(
                    step_id="frame-1",
                    action=PlanActionType.SUMMARIZE,
                    description="frame the research objective",
                ),
                PlanStep(
                    step_id="research-1",
                    action=PlanActionType.SEARCH_PRECEDENTS,
                    description="termination obligations",
                ),
            ],
            "current_step": 0,
            "plan_approved": True,
            "messages": [],
            "risk_analysis": None,
            "compliance_result": None,
            "grounding": None,
            "human_review": None,
            "status": WorkflowStatus.PLAN_APPROVED,
        },
    )

    assert final["status"] is WorkflowStatus.PLAN_APPROVED
    assert "evh-1" in final["deep_research_results"]
    assert final["evidence_registry"]["evh-1"].document_id == DOC
    # The bounded envelope landed in the transcript: status, handles,
    # coverage — no snippets, no read content.
    envelope_message = json.loads(final["messages"][-1].content)
    assert envelope_message["tool"] == "deep_research"
    assert envelope_message["status"] == "completed"
    assert envelope_message["evidence"] == ["evh-1"]
    assert "snippet" not in final["messages"][-1].content
