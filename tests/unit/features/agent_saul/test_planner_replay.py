"""Planner approval is a durable proposal boundary, not an LLM replay."""

from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any, TypedDict

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, StateSnapshot, interrupt
from returns.result import Success

from app.features.agent_saul.api.dto import WSResumeMessage
from app.features.agent_saul.application.service import (
    AgentSaulService,
    _legacy_planner_proposal,
)
from app.features.agent_saul.application.transcript.projection import snapshot_to_view
from app.features.agent_saul.domain.state import (
    STATE_SCHEMA_VERSION,
    LegalAgentState,
    OrchestratorAction,
    OrchestratorActionType,
    PlanActionType,
    PlanStep,
    WorkflowStatus,
)
from app.features.agent_saul.graph.nodes import (
    PlannerOutput,
    make_plan_approval_node,
    make_plan_generation_node,
    route_deep_research,
    route_from_orchestrator,
)
from app.observability.logging import logger


class _PlannerState(TypedDict, total=False):
    user_id: str
    thread_id: str
    messages: list[Any]
    plan: list[PlanStep]
    current_step: int
    plan_approved: bool
    plan_rejection_feedback: str | None
    status: WorkflowStatus


class _LegacyResumeState(TypedDict, total=False):
    """Small graph schema covering the channels a legacy resume hydrates."""

    schema_version: int
    user_id: str
    thread_id: str
    correlation_id: str
    doc_id: str
    document_text: str | None
    messages: list[Any]
    user_query: str
    qna_confidence: float
    plan: list[PlanStep]
    current_step: int
    plan_approved: bool
    plan_rejection_feedback: str | None
    working_memory: dict[str, Any]
    status: WorkflowStatus


def _legal_state(
    *,
    orchestrator_action: OrchestratorAction | None = None,
    plan: list[PlanStep] | None = None,
    plan_approved: bool = False,
    status: WorkflowStatus = WorkflowStatus.INITIALIZED,
) -> LegalAgentState:
    """Build the complete graph state required by typed routing nodes."""

    return {
        "user_id": "user-1",
        "thread_id": "thread-1",
        "correlation_id": "correlation-1",
        "schema_version": STATE_SCHEMA_VERSION,
        "doc_id": "doc-1",
        "document_text": None,
        "messages": [],
        "user_query": "query",
        "qna_confidence": 0.0,
        "plan": [] if plan is None else plan,
        "current_step": 0,
        "plan_approved": plan_approved,
        "plan_rejection_feedback": None,
        "orchestrator_action": orchestrator_action,
        "normalized_document": None,
        "segments": [],
        "extracted_entities": [],
        "relationships": [],
        "risk_analysis": None,
        "compliance_result": None,
        "grounding": None,
        "human_review": None,
        "final_report": None,
        "long_term_refs": [],
        "working_memory": {},
        "deep_research_results": None,
        "status": status,
        "errors": [],
        "retry_count": 0,
        "permissions": {},
    }


class _DeterministicPlanner:
    def __init__(self) -> None:
        self.calls = 0
        self.inputs: list[list[Any]] = []

    async def ainvoke(self, _messages: list[Any]) -> PlannerOutput:
        self.inputs.append(_messages)
        self.calls += 1
        return PlannerOutput(
            steps=[
                PlanStep(
                    step_id=f"S-{self.calls}",
                    action=PlanActionType.SUMMARIZE,
                    description=f"P{self.calls}",
                )
            ],
            rationale="deterministic test proposal",
        )


def _build_graph(planner: _DeterministicPlanner) -> Any:
    generate = make_plan_generation_node(planner)  # type: ignore[arg-type]
    approve = make_plan_approval_node()

    async def generate_node(state: _PlannerState) -> dict[str, Any]:
        return await generate(state)  # type: ignore[arg-type]

    async def approve_node(state: _PlannerState) -> dict[str, Any]:
        return await approve(state)  # type: ignore[arg-type]

    def after_approval(state: _PlannerState) -> str:
        return "generate" if state.get("status") == WorkflowStatus.PLAN_REJECTED else END

    builder = StateGraph(_PlannerState)
    builder.add_node("generate", generate_node)
    builder.add_node("approve", approve_node)
    builder.add_edge(START, "generate")
    builder.add_edge("generate", "approve")
    builder.add_conditional_edges("approve", after_approval, {"generate": "generate", END: END})
    return builder.compile(checkpointer=InMemorySaver())


def _initial_state() -> _PlannerState:
    return {
        "user_id": "user-1",
        "thread_id": "thread-1",
        "messages": [],
        "plan": [],
        "current_step": 0,
        "plan_approved": False,
        "plan_rejection_feedback": None,
        "status": WorkflowStatus.INITIALIZED,
    }


async def test_approval_replay_executes_checkpointed_p1_without_rerunning_planner() -> None:
    planner = _DeterministicPlanner()
    graph = _build_graph(planner)
    config = {"configurable": {"thread_id": "thread-1"}}

    paused = await graph.ainvoke(_initial_state(), config)
    assert planner.calls == 1
    assert paused["__interrupt__"][0].value["plan"][0]["description"] == "P1"

    resumed = await graph.ainvoke(Command(resume={"action": "approve"}), config)

    assert planner.calls == 1
    assert resumed["plan"][0].description == "P1"
    assert resumed["plan_approved"] is True


async def test_reject_replans_and_modify_uses_validated_user_plan() -> None:
    planner = _DeterministicPlanner()
    graph = _build_graph(planner)
    config = {"configurable": {"thread_id": "thread-1"}}

    await graph.ainvoke(_initial_state(), config)
    replanned = await graph.ainvoke(
        Command(resume={"action": "reject", "feedback": "add evidence"}),
        config,
    )

    assert planner.calls == 2
    assert replanned["__interrupt__"][0].value["plan"][0]["description"] == "P2"
    assert any("add evidence" in str(message.content) for message in planner.inputs[1])

    modified = {
        "step_id": "S-custom",
        "action": PlanActionType.COMPLIANCE_CHECK.value,
        "description": "user-approved modification",
    }
    finished = await graph.ainvoke(
        Command(resume={"action": "modify", "modified_plan": [modified]}),
        config,
    )

    assert planner.calls == 2
    assert finished["plan"][0].step_id == "S-custom"
    assert finished["plan"][0].action == PlanActionType.COMPLIANCE_CHECK
    assert finished["plan_rejection_feedback"] is None


def test_planner_routes_are_explicit_and_rejection_cannot_skip_replan() -> None:
    assert route_from_orchestrator(_legal_state()) == "planner_generate"
    assert (
        route_from_orchestrator(_legal_state(status=WorkflowStatus.PLAN_REJECTED))
        == "planner_generate"
    )
    assert (
        route_deep_research(_legal_state(status=WorkflowStatus.PLAN_REJECTED)) == "planner_generate"
    )
    start = OrchestratorAction(
        action_type=OrchestratorActionType.START_PIPELINE,
        reflection="start",
    )
    assert (
        route_from_orchestrator(
            _legal_state(orchestrator_action=start, status=WorkflowStatus.COMPLETED)
        )
        == "__end__"
    )
    assert (
        route_from_orchestrator(
            _legal_state(orchestrator_action=start, status=WorkflowStatus.FINALIZING)
        )
        == "finalization"
    )
    plan = PlanStep(
        step_id="S-1",
        action=PlanActionType.SUMMARIZE,
        description="ready",
    )
    assert route_from_orchestrator(_legal_state(orchestrator_action=start)) == "planner_generate"
    assert (
        route_from_orchestrator(_legal_state(orchestrator_action=start, plan=[plan]))
        == "planner_generate"
    )
    assert (
        route_from_orchestrator(
            _legal_state(orchestrator_action=start, plan=[plan], plan_approved=True)
        )
        == "ingestion"
    )
    assert (
        route_from_orchestrator(
            _legal_state(
                orchestrator_action=OrchestratorAction(
                    action_type=OrchestratorActionType.DONE,
                    reflection="done",
                )
            )
        )
        == "__end__"
    )


def _legacy_snapshot(*, proposal: list[dict[str, Any]]) -> StateSnapshot:
    pending = SimpleNamespace(
        id="legacy-interrupt-id",
        value={
            "type": "plan_approval",
            "plan": proposal,
            "message": "Please review and approve the execution plan",
        },
    )
    return StateSnapshot(
        values={"schema_version": 1, "plan": []},
        next=("planner",),
        config={"configurable": {"thread_id": "thread-legacy", "checkpoint_id": "cp-legacy"}},
        metadata={},
        created_at=datetime.now(tz=UTC).isoformat(),
        parent_config=None,
        tasks=(SimpleNamespace(interrupts=(pending,)),),
        interrupts=(),
    )


class _ResumeRecorder:
    def __init__(self) -> None:
        self.evidence: list[Any] = []

    async def prepare_resume(self, *, plan_resume: Any = None) -> Any:
        self.evidence.append(plan_resume)
        return Success(None)

    async def reconcile(self) -> Any:
        return Success(None)


class _ResumeGraph:
    def __init__(self, snapshot: StateSnapshot) -> None:
        self.snapshot = snapshot

    async def aget_state(self, _config: Any) -> StateSnapshot:
        return self.snapshot


async def test_legacy_paused_planner_checkpoint_resumes_the_shown_plan() -> None:
    proposal = [
        {
            "step_id": "S-legacy",
            "action": "summarize",
            "description": "P1 shown by the old planner",
        }
    ]
    snapshot = _legacy_snapshot(proposal=proposal)
    recovered, error = _legacy_planner_proposal(snapshot)

    assert error is None
    assert recovered is not None
    assert recovered[0].description == proposal[0]["description"]

    service = AgentSaulService.__new__(AgentSaulService)
    service._graph = _ResumeGraph(snapshot)
    service._log = logger.bind(test="legacy-planner")
    recorder = _ResumeRecorder()
    command = await service._apply_resume(
        WSResumeMessage(thread_id="thread-legacy", action="approve"),
        ws=SimpleNamespace(),
        thread_id="thread-legacy",
        recorder=recorder,
        config={"configurable": {"thread_id": "thread-legacy"}},
    )

    assert command is not None
    assert command.update["plan"][0].description == proposal[0]["description"]
    assert command.update["schema_version"] == STATE_SCHEMA_VERSION
    assert command.update["plan_approved"] is False
    assert command.update["plan_rejection_feedback"] is None
    assert recorder.evidence[0] is not None


async def test_reject_does_not_record_approval_evidence() -> None:
    snapshot = _legacy_snapshot(
        proposal=[
            {
                "step_id": "S-legacy",
                "action": "summarize",
                "description": "P1",
            }
        ]
    )
    service = AgentSaulService.__new__(AgentSaulService)
    service._graph = _ResumeGraph(snapshot)
    service._log = logger.bind(test="legacy-planner-reject")
    recorder = _ResumeRecorder()

    command = await service._apply_resume(
        WSResumeMessage(thread_id="thread-legacy", action="reject"),
        ws=SimpleNamespace(),
        thread_id="thread-legacy",
        recorder=recorder,
        config={"configurable": {"thread_id": "thread-legacy"}},
    )

    assert command is not None
    assert recorder.evidence == [None]


async def test_legacy_checkpoint_resume_reuses_displayed_plan_end_to_end() -> None:
    """A direct legacy-node resume must hydrate before the new approval node."""

    saver = InMemorySaver()
    legacy_llm_calls = 0
    proposal = PlanStep(
        step_id="S-legacy",
        action=PlanActionType.SUMMARIZE,
        description="P1 shown by the old planner",
    )

    async def old_planner(_state: _LegacyResumeState) -> dict[str, Any]:
        nonlocal legacy_llm_calls
        legacy_llm_calls += 1
        answer = interrupt(
            {
                "type": "plan_approval",
                "plan": [proposal.model_dump()],
                "message": "Please review and approve the execution plan",
            }
        )
        assert answer["action"] == "approve"
        return {
            "plan": [proposal],
            "plan_approved": True,
            "status": WorkflowStatus.PLAN_APPROVED,
        }

    old_builder = StateGraph(_LegacyResumeState)
    old_builder.add_node("planner", old_planner)
    old_builder.add_edge(START, "planner")
    old_graph = old_builder.compile(checkpointer=saver)
    config = {"configurable": {"thread_id": "legacy-e2e"}}
    paused = await old_graph.ainvoke({"plan": [], "schema_version": 1}, config)
    assert paused["__interrupt__"][0].value["plan"] == [proposal.model_dump()]
    assert legacy_llm_calls == 1
    legacy_snapshot = await old_graph.aget_state(config)

    service = AgentSaulService.__new__(AgentSaulService)
    service._graph = _ResumeGraph(legacy_snapshot)
    service._log = logger.bind(test="legacy-e2e")
    recorder = _ResumeRecorder()
    command = await service._apply_resume(
        WSResumeMessage(thread_id="legacy-e2e", action="approve"),
        ws=SimpleNamespace(),
        thread_id="legacy-e2e",
        recorder=recorder,
        config=config,
    )
    assert command is not None
    assert command.update["schema_version"] == STATE_SCHEMA_VERSION
    assert command.update["plan"][0] == proposal

    approve = make_plan_approval_node()

    async def approval_node(state: _LegacyResumeState) -> dict[str, Any]:
        return await approve(state)  # type: ignore[arg-type]

    def later_pause(_state: _LegacyResumeState) -> dict[str, Any]:
        interrupt({"type": "clarification_needed", "message": "later pause"})
        return {}

    resumed_builder = StateGraph(_LegacyResumeState)
    resumed_builder.add_node("planner", approval_node)
    resumed_builder.add_node("later_pause", later_pause)
    resumed_builder.add_edge(START, "planner")
    resumed_builder.add_edge("planner", "later_pause")
    resumed_graph = resumed_builder.compile(checkpointer=saver)

    paused_again = await resumed_graph.ainvoke(command, config)
    assert legacy_llm_calls == 1
    assert paused_again["plan"] == [proposal]
    assert paused_again["plan_approved"] is True

    final_snapshot = await resumed_graph.aget_state(config)
    assert final_snapshot.values["schema_version"] == STATE_SCHEMA_VERSION
    assert final_snapshot.values["plan"] == [proposal]
    view = snapshot_to_view(final_snapshot)
    assert view.schema_version == STATE_SCHEMA_VERSION
    assert final_snapshot.next == ("later_pause",)
