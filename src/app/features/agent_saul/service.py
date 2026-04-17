"""
AgentSaulService: the session orchestration layer.

Responsibilities:
  - Build initial LegalAgentState from WSStartMessage
  - Drive the astream_events v2 loop → WebSocket frames
  - Detect LangGraph interrupt() pauses via aget_state()
  - Emit WSHITLInterruptFrame, await WSResumeMessage, issue Command(resume=...)
  - Loop until graph completes (state.next == ()) or error

The HITL loop pattern:

    while True:
        stream graph events → ws frames
        check state snapshot for pending interrupts
        if interrupt → send HITLFrame → await resume → Command(resume=...)
        if no next nodes → send DoneFrame → break

This service is instantiated ONCE per WebSocket session.  It holds no
class-level mutable state across sessions — thread-safe.
"""

from typing import Any, cast
from uuid import uuid4

from fastapi import WebSocket
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command
from pydantic import ValidationError
from redis.asyncio import Redis

from app.features.agent_saul.dto import (
    WSDoneFrame,
    WSErrorFrame,
    WSHITLInterruptFrame,
    WSNodeEndFrame,
    WSNodeStartFrame,
    WSOutbound,
    WSPingMessage,
    WSPongFrame,
    WSResumeMessage,
    WSStartMessage,
    WSStateUpdateFrame,
    WSTokenFrame,
    ws_inbound_adapter,
)
from app.features.auth.websocket_security import (
    WebSocketIdleTimeout,
    WebSocketSecurityContext,
    WebSocketSecurityService,
    WebSocketSecurityViolation,
)
from app.shared.langgraph_layer.agent_saul.state import (
    GRAPH_NODE_NAMES,
    HITLInterruptType,
    WorkflowStatus,
)
from app.utils import logger

# Status emitted when a node starts — keeps WS client progress bar accurate.
_NODE_STATUS_MAP: dict[str, WorkflowStatus] = {
    "gateway": WorkflowStatus.INITIALIZED,
    "qna": WorkflowStatus.QNA_CLARIFICATION,
    "orchestrator": WorkflowStatus.PLAN_PENDING,
    "planner": WorkflowStatus.PLAN_AWAITING_APPROVAL,
    "ingestion": WorkflowStatus.INGESTING,
    "normalization": WorkflowStatus.NORMALIZING,
    "segmentation": WorkflowStatus.SEGMENTING,
    "entity_extraction": WorkflowStatus.EXTRACTING_ENTITIES,
    "relationship_mapping": WorkflowStatus.MAPPING_RELATIONSHIPS,
    "risk_analysis": WorkflowStatus.ANALYZING_RISKS,
    "compliance": WorkflowStatus.CHECKING_COMPLIANCE,
    "grounding_verification": WorkflowStatus.VERIFYING_GROUNDING,
    "human_review": WorkflowStatus.AWAITING_HUMAN_REVIEW,
    "finalization": WorkflowStatus.FINALIZING,
    "persist_memory": WorkflowStatus.PERSISTING_MEMORY,
}

_TOTAL_PIPELINE_STEPS = len(GRAPH_NODE_NAMES)


class AgentSaulService:
    """Per-session WebSocket orchestration service.

    Constructor receives infra clients; run_session() is the entry point.
    No mutable state accumulates between run_session() calls — instantiate
    once per WebSocket connection.
    """

    def __init__(
        self,
        graph: CompiledStateGraph,
        redis: Redis,
        correlation_id: str,
        ws_security: WebSocketSecurityService,
        ws_context: WebSocketSecurityContext,
    ) -> None:
        self._graph = graph
        self._redis = redis
        self._correlation_id = correlation_id
        self._ws_security = ws_security
        self._ws_context = ws_context
        self._log = logger.bind(correlation_id=correlation_id)

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    async def run_session(
        self,
        ws: WebSocket,
        start_msg: WSStartMessage,
        thread_id: str,
        user_id: str,
    ) -> None:
        """Main session loop.  Runs until graph completes or WS closes."""
        config = self._build_config(thread_id)
        initial_input = self._build_initial_state(start_msg, thread_id, user_id)

        # current_input alternates between:
        #   - dict (LegalAgentState subset) on first run
        #   - Command(resume=...) on each HITL resume
        current_input: Any = initial_input

        step = 0
        while True:
            # --- Stream graph events → WS frames ----------------------------
            async for event in self._graph.astream_events(
                current_input,
                config=config,
                version="v2",
            ):
                frame = self._map_event_to_frame(event=event, current_step=step)
                if frame is not None:
                    await self._send_json(ws, frame.model_dump())

            # --- Inspect post-stream graph state ----------------------------
            state_snapshot = await self._graph.aget_state(config)
            metadata = state_snapshot.metadata
            if isinstance(metadata, dict):
                step = int(metadata.get("step", step))

            # Graph has no pending nodes → pipeline completed
            if not state_snapshot.next:
                final_state = state_snapshot.values
                final_report = final_state.get("final_report")
                summary = final_report.summary if final_report else None
                await self._send_json(ws, WSDoneFrame(thread_id=thread_id, summary=summary).model_dump())
                self._log.info("saul_session_completed", thread_id=thread_id)
                break

            # --- Detect pending interrupts ----------------------------------
            pending_interrupts = [
                task_interrupt
                for task in state_snapshot.tasks
                for task_interrupt in task.interrupts
            ]

            if not pending_interrupts:
                # Graph paused without an interrupt — unexpected topology state
                self._log.error(
                    "saul_graph_paused_no_interrupt",
                    thread_id=thread_id,
                    next_nodes=state_snapshot.next,
                )
                await self._send_json(
                    ws,
                    WSErrorFrame(
                        node=None,
                        code="UNEXPECTED_PAUSE",
                        message="Graph paused without a pending interrupt",
                        retryable=False,
                    ).model_dump(),
                )
                break

            # --- HITL: emit interrupt frame, await resume -------------------
            current_input = await self._handle_hitl(
                ws=ws,
                thread_id=thread_id,
                interrupt_value=pending_interrupts[0].value,
            )
            if current_input is None:
                # Client sent an invalid resume or disconnected
                break

    # ------------------------------------------------------------------
    # Event mapping: astream_events v2 → WS frames
    # ------------------------------------------------------------------

    def _map_event_to_frame(
        self,
        event: object,
        current_step: int,
    ) -> WSOutbound | None:
        if not isinstance(event, dict):
            return None

        event_data = cast("dict[str, Any]", event)
        event_type_obj = event_data.get("event")
        event_type = event_type_obj if isinstance(event_type_obj, str) else ""
        metadata_obj = event_data.get("metadata")
        metadata = metadata_obj if isinstance(metadata_obj, dict) else {}
        event_name_obj = event_data.get("name")
        event_name = event_name_obj if isinstance(event_name_obj, str) else ""
        data_obj = event_data.get("data")
        data = data_obj if isinstance(data_obj, dict) else {}

        # langgraph_node is always the top-level graph node, even when the
        # event originates from a sub-graph (create_react_agent) running inside.
        langgraph_node = metadata.get("langgraph_node")
        node_name = langgraph_node if isinstance(langgraph_node, str) else event_name
        langgraph_step_obj = metadata.get("langgraph_step")
        langgraph_step = langgraph_step_obj if isinstance(langgraph_step_obj, int) else current_step

        # ---- Per-token LLM streaming ----------------------------------------
        if event_type == "on_chat_model_stream":
            chunk = data.get("chunk")
            if chunk is None:
                return None
            content = chunk.content if isinstance(chunk.content, str) else ""
            if not content:
                return None
            return WSTokenFrame(node=node_name, token=content)

        # ---- Top-level node lifecycle events --------------------------------
        if event_type == "on_chain_start" and event_name in GRAPH_NODE_NAMES:
            status = _NODE_STATUS_MAP.get(event_name, WorkflowStatus.INITIALIZED)
            return WSNodeStartFrame(
                node=event_name,
                step=langgraph_step,
                status=status,
            )

        if event_type == "on_chain_end" and event_name in GRAPH_NODE_NAMES:
            output = data.get("output")
            output_keys = (
                [key for key in output if isinstance(key, str)]
                if isinstance(output, dict)
                else []
            )
            return WSNodeEndFrame(
                node=event_name,
                step=langgraph_step,
                output_keys=output_keys,
            )

        # ---- Status transitions (emitted by nodes setting state.status) -----
        if event_type == "on_chain_end" and event_name in GRAPH_NODE_NAMES:
            output = data.get("output")
            if isinstance(output, dict) and "status" in output:
                new_status = output["status"]
                return WSStateUpdateFrame(
                    status=new_status,
                    step=langgraph_step,
                    total_steps=_TOTAL_PIPELINE_STEPS,
                )

        return None

    # ------------------------------------------------------------------
    # HITL handling
    # ------------------------------------------------------------------

    async def _handle_hitl(
        self,
        ws: WebSocket,
        thread_id: str,
        interrupt_value: dict[str, Any],
    ) -> Command | None:
        """Emit HITL frame, wait for WSResumeMessage, return Command(resume=...)."""
        interrupt_type_raw = interrupt_value.get("type", HITLInterruptType.PLAN_APPROVAL)
        try:
            interrupt_type = HITLInterruptType(interrupt_type_raw)
        except ValueError:
            interrupt_type = HITLInterruptType.PLAN_APPROVAL

        await self._send_json(
            ws,
            WSHITLInterruptFrame(
                thread_id=thread_id,
                interrupt_type=interrupt_type,
                payload=interrupt_value,
                message=interrupt_value.get("message", "Human input required"),
            ).model_dump(),
        )
        self._log.info(
            "saul_hitl_interrupt_sent",
            thread_id=thread_id,
            interrupt_type=interrupt_type,
        )

        # Wait for the client to respond — may take arbitrarily long (human reviewing)
        while True:
            try:
                raw = await self._receive_json(ws)
            except (WebSocketIdleTimeout, WebSocketSecurityViolation):
                raise
            except Exception as exc:
                self._log.warning("saul_ws_receive_failed", error=str(exc))
                return None

            try:
                inbound = ws_inbound_adapter.validate_python(raw)
            except ValidationError as exc:
                await self._send_json(
                    ws,
                    WSErrorFrame(
                        node=None,
                        code="INVALID_FRAME",
                        message=f"Frame parse error: {exc.error_count()} validation error(s)",
                        retryable=True,
                    ).model_dump(),
                )
                continue  # Ask client to resend

            if isinstance(inbound, WSPingMessage):
                await self._send_json(ws, WSPongFrame().model_dump())
                continue

            if not isinstance(inbound, WSResumeMessage):
                await self._send_json(
                    ws,
                    WSErrorFrame(
                        node=None,
                        code="EXPECTED_RESUME",
                        message=f"Expected 'resume' message, got '{inbound.type}'",
                        retryable=True,
                    ).model_dump(),
                )
                continue

            if inbound.thread_id != thread_id:
                await self._send_json(
                    ws,
                    WSErrorFrame(
                        node=None,
                        code="THREAD_ID_MISMATCH",
                        message="Resume message thread_id does not match active session",
                        retryable=False,
                    ).model_dump(),
                )
                return None

            self._log.info(
                "saul_hitl_resume_received",
                thread_id=thread_id,
                action=inbound.action,
            )

            # Build the resume payload the interrupted node's interrupt() call returns.
            resume_payload: dict[str, Any] = {
                "action": inbound.action,
                "feedback": inbound.feedback,
                "modified_plan": (
                    [step.model_dump() for step in inbound.modified_plan]
                    if inbound.modified_plan
                    else None
                ),
                "overrides": inbound.overrides,
                "reviewer_role": inbound.reviewer_role,
            }
            return Command(resume=resume_payload)

    async def _receive_json(self, ws: WebSocket) -> object:
        return await self._ws_security.receive_json(ws, self._ws_context)

    async def _send_json(self, ws: WebSocket, payload: object) -> None:
        await self._ws_security.send_json(ws, payload, self._ws_context)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_config(self, thread_id: str) -> RunnableConfig:
        return RunnableConfig(
            configurable={"thread_id": thread_id},
            run_id=uuid4(),
            tags=["agent_saul"],
            metadata={"correlation_id": self._correlation_id},
        )

    def _build_initial_state(
        self,
        start_msg: WSStartMessage,
        thread_id: str,
        user_id: str,
    ) -> dict[str, Any]:
        """Build the minimal initial LegalAgentState dict.

        All Annotated accumulation fields (segments, extracted_entities,
        relationships, errors) start as empty lists — their reducers will
        append on each node return.
        """
        return {
            "user_id": user_id,
            "thread_id": thread_id,
            "correlation_id": self._correlation_id,
            "schema_version": 1,
            "doc_id": start_msg.doc_id,
            "document_text": None,
            "messages": [HumanMessage(content=start_msg.user_query)],
            "user_query": start_msg.user_query,
            "qna_confidence": 0.0,
            "plan": [],
            "current_step": 0,
            "plan_approved": False,
            "orchestrator_action": None,
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
            "status": WorkflowStatus.INITIALIZED,
            "errors": [],
            "retry_count": 0,
            "permissions": start_msg.permissions,
        }
