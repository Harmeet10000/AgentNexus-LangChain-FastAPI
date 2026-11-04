"""Bounded deep-research subgraph exposed as exactly one tool (D5).

The 2026-10-25 review decision rebuilds deep research as a compiled
``plan → search_evidence → read_evidence_by_id → synthesize`` subgraph with a
hard step bound, reached only through a single ``@tool`` wrapper. Design
decisions (openspec ``deep-research-bounded-subgraph``):

1. Compiled once at graph-build time, invoked per call, **no internal
   checkpointer** — the run is bounded and short-lived, so per-call state dies
   with the call and a resume is a re-invocation, not a checkpoint replay.
2. The wrapper is the only door, and it *constructs* the subgraph's initial
   state from server-owned ``ToolRuntime[ToolRuntimeContext]``. There is no
   ambient path for model-supplied scope to travel (closes check V6 by
   construction).
3. The step bound is a local, per-invocation counter with the same typed
   exhaustion semantics as Saul's ledger. Deviation note: the ledger itself is
   not yet wireable through ``ToolRuntimeContext`` (it is a state channel, not
   a context field), so the wrapper cannot receive it; the node-level
   ``DEEP_RESEARCH_FANOUT``/``RETRIEVAL_CALLS`` ledger gates still apply
   outside the tool, and no fresh allowance is minted beyond the fixed bound.
4. Handles stay run-scoped because the context already carries ``run_id``; the
   evidence service refuses any handle outside the issuing run.
5. No interrupts and no memory writes inside the subgraph: interrupts do not
   cross a tool boundary (reference 09), and deep research is read-only (P3).

Terminal envelope statuses are exactly ``completed | budget_exhausted |
no_evidence | ingestion_incomplete`` (plus ``refused`` for the pre-subgraph
authorization denial). Budget exhaustion carries partial handles and never a
synthesis rendered as a complete answer.
"""

from __future__ import annotations

from enum import StrEnum
from typing import (  # noqa: TC003 — langchain resolves tool annotations at registration time
    TYPE_CHECKING,
    Annotated,
    Any,
    TypedDict,
    cast,
)

from langchain.tools import ToolRuntime
from langchain_core.tools import InjectedToolArg, tool
from langgraph.errors import GraphRecursionError
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, ConfigDict, Field

from app.features.documents.api.http.dto import EvidenceSearchRequest, EvidenceSearchResponse
from app.features.documents.application.tools.authorization import trusted_tool_context

if TYPE_CHECKING:
    from collections.abc import Mapping

    from langchain_core.tools import BaseTool
    from langgraph.graph.state import CompiledStateGraph

    from app.ai.evaluation.deep_research_integration import DeepResearchInstrumentation

#: Stage-granularity step bound: search, read, and synthesize each consume one
#: step; ``plan`` is deterministic and free. The default leaves one spare step.
DEFAULT_MAX_STEPS = 4

DEFAULT_SEARCH_LIMIT = 5

#: Maximum handles dereferenced in one bounded read stage. Remaining handles
#: are still returned; only the content notes for synthesis are capped.
DEFAULT_MAX_READS = 4

_READ_NOTE_CHARS = 400
_RECURSION_HEADROOM = 8

_RUN_SCOPING_RULE = (
    "evidence handles are valid only within the run that issued them "
    "(handle-membership is scoped by run_id)"
)


class DeepResearchStatus(StrEnum):
    """Terminal envelope statuses; ``RUNNING`` is internal and never returned."""

    RUNNING = "running"
    COMPLETED = "completed"
    BUDGET_EXHAUSTED = "budget_exhausted"
    NO_EVIDENCE = "no_evidence"
    INGESTION_INCOMPLETE = "ingestion_incomplete"


_TERMINAL_STATUSES: frozenset[DeepResearchStatus] = frozenset(
    {
        DeepResearchStatus.COMPLETED,
        DeepResearchStatus.BUDGET_EXHAUSTED,
        DeepResearchStatus.NO_EVIDENCE,
        DeepResearchStatus.INGESTION_INCOMPLETE,
    }
)

REFUSED_STATUS = "refused"


class DeepResearchState(TypedDict, total=False):
    """Internal state of the bounded research subgraph.

    Scope fields are written only by the tool wrapper from server-owned
    runtime context; nodes never read identity or document authority from any
    other source. Read content lives here only for the lifetime of the tool
    call (no checkpointer) and never reaches Saul's checkpointed state.
    """

    # Objective (the only model-visible inputs, via the wrapper args schema)
    objective: str
    focus: str | None
    # Server-owned scope — written by the wrapper, read-only for nodes
    user_id: str
    tenant_id: str
    allowed_doc_ids: tuple[str, ...]
    thread_id: str
    run_id: str
    # ``checkpoint_id`` itself is a reserved LangGraph channel name, so the
    # handle-checkpoint marker rides under the evidence-scoped name.
    evidence_checkpoint_id: str
    # Bounded loop bookkeeping
    queries: list[str]
    pending_handles: list[str]
    evidence: list[dict[str, Any]]  # EvidenceRef metadata only, never snippets
    read_notes: list[str]  # bounded content notes for synthesis
    refusals: list[str]  # typed read refusals (run-scoping, stale, expired)
    coverage: dict[str, int]
    steps_used: int
    max_steps: int
    max_reads: int
    search_limit: int
    status: str
    summary: str | None
    message: str | None


class DeepResearchInput(BaseModel):
    """Model-visible arguments for deep research — exactly these two.

    ``ConfigDict(extra="forbid")`` makes any model-supplied scope-like
    argument (tenant, document ids, run ids) a schema violation instead of a
    silently ignored hint. Identity and document scope are injected server
    properties, never model arguments.
    """

    model_config = ConfigDict(extra="forbid")

    objective: str = Field(
        description="What the research must establish, in one sentence.",
    )
    focus: str | None = Field(
        default=None,
        description="Optional narrower question to prioritise inside the objective.",
    )


def _envelope(
    state: Mapping[str, Any],
    *,
    status: str,
    message: str | None = None,
) -> dict[str, Any]:
    """Project the subgraph state onto the bounded tool envelope."""
    return {
        "status": status,
        "summary": state.get("summary"),
        "evidence": list(state.get("evidence", [])),
        "coverage": dict(state.get("coverage", {})),
        "message": message,
    }


def _exhausted_message(state: Mapping[str, Any]) -> str:
    return (
        "Deep-research step budget was exhausted before synthesis completed "
        f"(steps used: {state.get('steps_used', 0)}/{state.get('max_steps', 0)}); "
        "partial evidence handles are returned without a synthesized answer."
    )


def _merge_coverage(
    coverage: dict[str, int],
    response: EvidenceSearchResponse,
) -> dict[str, int]:
    merged = dict(coverage)
    for key, value in response.coverage.model_dump(mode="json").items():
        if isinstance(value, int):
            merged[key] = max(merged.get(key, 0), value)
    return merged


def _normalize_search_response(response: Any) -> EvidenceSearchResponse | None:
    if isinstance(response, EvidenceSearchResponse):
        return response
    try:
        return EvidenceSearchResponse.model_validate(response)
    except Exception:  # noqa: BLE001 — a malformed envelope is a typed refusal below
        return None


def _make_plan_node() -> Any:
    """Deterministic, free plan: bounded query list from objective plus focus."""

    async def plan_node(state: DeepResearchState) -> dict[str, Any]:
        queries = [state["objective"]]
        focus = state.get("focus")
        if focus and focus not in queries:
            queries.append(focus)
        return {
            "queries": queries[:2],
            "status": DeepResearchStatus.RUNNING.value,
        }

    return plan_node


def _instrument_node(node: Any, stage: str, instrumentation: DeepResearchInstrumentation) -> Any:
    async def instrumented_node(state: DeepResearchState) -> dict[str, Any]:
        input_state = dict(state)
        if stage == "plan":
            instrumentation.start(state["objective"], node_limit=8)
        delta = await node(state)
        tool_calls: list[str] = []
        if stage == "search":
            consumed = delta.get("steps_used", state["steps_used"]) - state["steps_used"]
            tool_calls = ["search_evidence"] * consumed
        elif stage == "read":
            read_notes = delta.get("read_notes", state.get("read_notes", []))
            refusals = delta.get("refusals", state.get("refusals", []))
            attempts = (
                len(read_notes)
                + len(refusals)
                - len(state.get("read_notes", []))
                - len(state.get("refusals", []))
            )
            tool_calls = ["read_evidence_by_id"] * attempts
        instrumentation.record_stage(stage, input_state, delta, tool_calls=tool_calls)
        return delta

    return instrumented_node


def _make_search_node(service: Any) -> Any:
    async def search_node(state: DeepResearchState) -> dict[str, Any]:
        if state["steps_used"] >= state["max_steps"]:
            return {
                "status": DeepResearchStatus.BUDGET_EXHAUSTED.value,
                "message": _exhausted_message(state),
            }

        evidence = list(state.get("evidence", []))
        pending: list[str] = []
        coverage = dict(state.get("coverage", {}))
        steps_used = state["steps_used"]
        status = DeepResearchStatus.RUNNING.value
        message: str | None = None

        for query in state.get("queries", []) or [state["objective"]]:
            if steps_used >= state["max_steps"]:
                status = DeepResearchStatus.BUDGET_EXHAUSTED.value
                message = _exhausted_message(state)
                break
            steps_used += 1
            response: EvidenceSearchResponse | None = _normalize_search_response(
                await service.search_evidence(
                    principal_id=state["user_id"],
                    tenant_id=state["tenant_id"],
                    allowed_doc_ids=state["allowed_doc_ids"],
                    thread_id=state["thread_id"],
                    run_id=state["run_id"],
                    checkpoint_id=state["evidence_checkpoint_id"],
                    request=EvidenceSearchRequest(
                        query=query, limit=state.get("search_limit", DEFAULT_SEARCH_LIMIT)
                    ),
                )
            )
            if response is None:
                status = DeepResearchStatus.INGESTION_INCOMPLETE.value
                message = "Evidence search returned a malformed envelope."
                break
            coverage = _merge_coverage(coverage, response)
            if response.status == "ok":
                for item in response.items:
                    ref = item.evidence.model_dump(mode="json")
                    evidence.append(ref)
                    pending.append(item.evidence.evidence_id)
                continue
            if response.status == "no_evidence":
                status = DeepResearchStatus.NO_EVIDENCE.value
                message = response.message
                break
            # Readiness/authorization/transient failures are envelope statuses,
            # never confident negatives: the scope could not be served.
            status = DeepResearchStatus.INGESTION_INCOMPLETE.value
            message = response.message or f"Evidence search did not complete ({response.status})."
            break

        if status == DeepResearchStatus.RUNNING.value and not evidence:
            status = DeepResearchStatus.NO_EVIDENCE.value
            message = "No admissible evidence was found for the research objective."

        return {
            "evidence": evidence,
            "pending_handles": pending,
            "coverage": coverage,
            "steps_used": steps_used,
            "status": status,
            "message": message,
        }

    return search_node


def _make_read_node(service: Any) -> Any:
    async def read_node(state: DeepResearchState) -> dict[str, Any]:
        if state["steps_used"] >= state["max_steps"]:
            return {
                "status": DeepResearchStatus.BUDGET_EXHAUSTED.value,
                "message": _exhausted_message(state),
            }

        read = getattr(service, "read_evidence_by_id", None)
        if read is None:  # narrow test doubles never dereference content
            return {"status": DeepResearchStatus.RUNNING.value}

        steps_used = state["steps_used"] + 1
        read_notes = list(state.get("read_notes", []))
        refusals = list(state.get("refusals", []))

        for evidence_id in state.get("pending_handles", [])[
            : state.get("max_reads", DEFAULT_MAX_READS)
        ]:
            response = await read(
                evidence_id=evidence_id,
                principal_id=state["user_id"],
                tenant_id=state["tenant_id"],
                allowed_doc_ids=state["allowed_doc_ids"],
                thread_id=state["thread_id"],
                run_id=state["run_id"],
                checkpoint_id=state["evidence_checkpoint_id"],
            )
            if getattr(response, "status", None) != "ok":
                refusals.append(
                    f"handle {evidence_id} was refused ({getattr(response, 'status', 'unknown')}"
                    f": {getattr(response, 'message', '')}); {_RUN_SCOPING_RULE}"
                )
                continue
            item = response.items[0]
            ref = item.evidence.model_dump(mode="json")
            read_notes.append(
                f"[{ref['document_id']} p{ref['page_no']}] {item.content[:_READ_NOTE_CHARS]}"
            )

        status = DeepResearchStatus.RUNNING.value
        message: str | None = None
        if steps_used >= state["max_steps"]:
            # Budget consumed before synthesis: report partial handles and
            # never render a confident partial answer.
            status = DeepResearchStatus.BUDGET_EXHAUSTED.value
            message = _exhausted_message(state)
        elif state.get("pending_handles") and not read_notes:
            # Every evidence read returned expired/unauthorized/transient_failure:
            # report scope-unavailable, never a completed run.
            status = DeepResearchStatus.INGESTION_INCOMPLETE.value
            message = f"No evidence passages could be read: {'; '.join(refusals)}"
        return {
            "read_notes": read_notes,
            "refusals": refusals,
            "steps_used": steps_used,
            "status": status,
            "message": message,
        }

    return read_node


def _make_synthesize_node() -> Any:
    async def synthesize_node(state: DeepResearchState) -> dict[str, Any]:
        if state["steps_used"] >= state["max_steps"]:
            return {
                "status": DeepResearchStatus.BUDGET_EXHAUSTED.value,
                "message": _exhausted_message(state),
            }
        evidence = state.get("evidence", [])
        if not evidence:
            return {
                "status": DeepResearchStatus.NO_EVIDENCE.value,
                "summary": None,
                "message": state.get("message")
                or "No admissible evidence was found for the research objective.",
            }

        read_notes = state.get("read_notes", [])
        if state.get("status") == DeepResearchStatus.INGESTION_INCOMPLETE.value:
            return {
                "status": DeepResearchStatus.INGESTION_INCOMPLETE.value,
                "summary": None,
                "message": state.get("message")
                or "Evidence passages could not be read for the research objective.",
            }
        document_ids = sorted({str(ref.get("document_id")) for ref in evidence})
        lines = [
            f"Research objective: {state['objective']}",
            (
                f"Bounded run: {len(evidence)} evidence handle(s) across "
                f"{len(document_ids)} document(s); {len(read_notes)} passage(s) read."
            ),
        ]
        lines.extend(f"- {note}" for note in read_notes[:DEFAULT_MAX_READS])
        refusals = state.get("refusals", [])
        if refusals:
            lines.append(f"- {len(refusals)} handle read(s) were refused.")
        lines.append("Handles are metadata only; verify quotations against the cited documents.")
        return {
            "steps_used": state["steps_used"] + 1,
            "status": DeepResearchStatus.COMPLETED.value,
            "summary": "\n".join(lines),
            "message": "; ".join(refusals) if refusals else None,
        }

    return synthesize_node


def _route_after_search(state: DeepResearchState) -> str:
    status = state.get("status", "")
    if status in {item.value for item in _TERMINAL_STATUSES}:
        return END
    if state.get("pending_handles"):
        return "read"
    return "synthesize"


def _route_after_read(state: DeepResearchState) -> str:
    if state.get("status") == DeepResearchStatus.BUDGET_EXHAUSTED.value:
        return END
    return "synthesize"


def compile_deep_research_subgraph(
    service: Any,
    *,
    instrumentation: DeepResearchInstrumentation | None = None,
) -> CompiledStateGraph[Any]:
    """Compile the bounded plan → search → read → synthesize machine.

    Called once at graph-build time. No checkpointer (design Decision 1):
    per-call state dies with the tool call.
    """
    # The same typed-construction workaround build_saul_graph uses: ty cannot
    # bind a ``total=False`` TypedDict through the StateGraph overload.
    state_graph_factory = cast("Any", StateGraph)
    graph: Any = state_graph_factory(DeepResearchState)
    nodes = {
        "plan": _make_plan_node(),
        "search": _make_search_node(service),
        "read": _make_read_node(service),
        "synthesize": _make_synthesize_node(),
    }
    if instrumentation is not None:
        for stage, node in nodes.items():
            nodes[stage] = _instrument_node(node, stage, instrumentation)
    for stage, node in nodes.items():
        graph.add_node(stage, node)
    graph.add_edge(START, "plan")
    graph.add_edge("plan", "search")
    graph.add_conditional_edges("search", _route_after_search)
    graph.add_conditional_edges("read", _route_after_read)
    graph.add_edge("synthesize", END)
    return cast("CompiledStateGraph[Any]", graph.compile())


def make_deep_research_tool(
    service: Any,
    *,
    max_steps: int = DEFAULT_MAX_STEPS,
    search_limit: int = DEFAULT_SEARCH_LIMIT,
    max_reads: int = DEFAULT_MAX_READS,
    instrumentation: DeepResearchInstrumentation | None = None,
) -> BaseTool:
    """Build the single deep-research tool wrapping the compiled subgraph.

    The wrapper is the only door into the subgraph: it builds the initial
    state from ``ToolRuntime[ToolRuntimeContext]`` and denies by default when
    no valid server-owned runtime context is present. Model-visible arguments
    are exactly ``objective`` and optional ``focus``.
    """
    subgraph = compile_deep_research_subgraph(service, instrumentation=instrumentation)

    @tool(args_schema=DeepResearchInput)
    async def deep_research(
        objective: str,
        focus: str | None = None,
        runtime: Annotated[ToolRuntime | None, InjectedToolArg] = None,
    ) -> dict[str, Any]:
        """Run bounded precedent research over the authorized document scope."""
        try:
            authz = trusted_tool_context(runtime)
        except PermissionError as exc:
            # Deny-by-default: a typed refusal naming the missing
            # authorization; the subgraph never starts.
            return _envelope(
                {},
                status=REFUSED_STATUS,
                message=str(exc),
            )
        if not authz.allowed_doc_ids:
            return _envelope(
                {},
                status=REFUSED_STATUS,
                message="deep research has no authorized document scope in the tool runtime",
            )

        initial: DeepResearchState = {
            "objective": objective,
            "focus": focus,
            "user_id": authz.user_id,
            "tenant_id": authz.tenant_id,
            "allowed_doc_ids": authz.allowed_doc_ids,
            "thread_id": authz.thread_id,
            "run_id": authz.run_id,
            "evidence_checkpoint_id": authz.checkpoint_id or f"run:{authz.run_id}",
            "queries": [],
            "pending_handles": [],
            "evidence": [],
            "read_notes": [],
            "refusals": [],
            "coverage": {},
            "steps_used": 0,
            "max_steps": max_steps,
            "max_reads": max_reads,
            "search_limit": search_limit,
            "status": DeepResearchStatus.RUNNING.value,
            "summary": None,
            "message": None,
        }
        try:
            final = await subgraph.ainvoke(
                initial,
                config={"recursion_limit": max_steps * 2 + _RECURSION_HEADROOM},
            )
        except GraphRecursionError:
            # Belt-and-braces behind the in-state counter: a hard stop is a
            # typed exhaustion, never a truncated answer.
            return _envelope(
                {},
                status=DeepResearchStatus.BUDGET_EXHAUSTED.value,
                message="Deep-research recursion bound was reached before synthesis.",
            )
        status = final.get("status", DeepResearchStatus.INGESTION_INCOMPLETE.value)
        return _envelope(final, status=status, message=final.get("message"))

    return deep_research
