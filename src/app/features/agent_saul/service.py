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

from datetime import UTC, datetime
from typing import Any, cast
from uuid import uuid4

from fastapi import WebSocket
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command, StateSnapshot
from pydantic import ValidationError
from redis.asyncio import Redis
from redis.exceptions import RedisError
from returns.result import Failure, Success
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.features.agent_saul.dto import (
    CreateSessionResponse,
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
from app.features.agent_saul.graph.state import (
    GRAPH_NODE_NAMES,
    STATE_SCHEMA_VERSION,
    HITLInterruptType,
    StateSchemaVersionError,
    WorkflowStatus,
    hydrate_state,
)
from app.features.agent_saul.thread_lock import ThreadLock
from app.features.agent_saul.transcript.projection import PAUSE_WINDOW
from app.features.agent_saul.transcript.recorder import TurnRecorder
from app.features.agent_saul.transcript.repository import ChatTranscriptRepository
from app.features.auth.websocket_security import (
    WebSocketIdleTimeout,
    WebSocketSecurityContext,
    WebSocketSecurityService,
    WebSocketSecurityViolation,
)
from app.observability.logging import logger, trace_layer

from .exceptions import AgentSaulResult

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
        graph: CompiledStateGraph[Any],
        redis: Redis,
        correlation_id: str,
        ws_security: WebSocketSecurityService,
        ws_context: WebSocketSecurityContext,
        *,
        session_factory: async_sessionmaker[AsyncSession],
    ) -> None:
        self._graph = graph
        self._redis = redis
        self._correlation_id = correlation_id
        self._ws_security = ws_security
        self._ws_context = ws_context
        self._session_factory = session_factory
        self._log = logger.bind(correlation_id=correlation_id)

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    @staticmethod
    def create_session(
        *, websocket_url: str, thread_id: str
    ) -> AgentSaulResult[CreateSessionResponse]:
        """Create the HTTP pre-flight response before entering the WS boundary."""
        return Success(CreateSessionResponse(thread_id=thread_id, ws_url=websocket_url))

    @trace_layer("service")
    async def run_session(
        self,
        ws: WebSocket,
        start_msg: WSStartMessage,
        thread_id: str,
        user_id: str,
    ) -> None:
        """Main session loop.  Runs until graph completes or WS closes.

        Connection-lifetime discipline (doc §12.5): lock acquisition → hydrate
        gate → reconcile → (paused-thread refusal / expiry) → run loop. Every
        boundary update rides the recorder; ``finally`` runs T3 and releases
        the lock — the TTL fence is only for process death.
        """
        config = self._build_config(thread_id)

        lock = ThreadLock(self._redis, thread_id)
        repo = ChatTranscriptRepository(self._session_factory)
        recorder = TurnRecorder(
            repo=repo,
            graph=self._graph,
            config=config,
            thread_id=thread_id,
            user_id=user_id,
            redis=self._redis,
            lock=lock,
        )
        config["callbacks"] = [recorder.usage_handler]

        acquired = await lock.acquire()
        if not acquired:
            await self._refuse_thread(ws, thread_id)
            return

        try:
            gated = await self._gate_and_reconcile(
                ws=ws,
                recorder=recorder,
                config=config,
                thread_id=thread_id,
            )
            if not gated:
                return
            initial_input = self._hydrate_or_refuse(
                ws,
                config,
                self._build_initial_state(start_msg, thread_id, user_id),
                thread_id,
            )
            if initial_input is None:
                return

            # current_input alternates between:
            #   - dict (LegalAgentState subset) on first run
            #   - Command(resume=...) on each HITL resume
            current_input: Any = initial_input

            step = 0
            first_stream_done = False
            while True:
                # --- Stream graph events → WS frames ----------------------------
                await self._stream_round(
                    ws=ws, config=config, current_input=current_input, step=step
                )
                # T0/T1): register the boundary once the first stream round has
                # produced durable checkpoints; no-op on later rounds (§5.2).
                if not first_stream_done:
                    recorder.on_first_checkpoint()
                    first_stream_done = True

                # --- Inspect post-stream graph state ----------------------------
                outcome, state_snapshot = await self._assess_round(
                    ws=ws, recorder=recorder, thread_id=thread_id, config=config
                )
                metadata = state_snapshot.metadata if state_snapshot else None
                if isinstance(metadata, dict):
                    step = int(metadata.get("step", step))
                if outcome != "paused" or state_snapshot is None:
                    break

                # --- HITL: emit interrupt frame, await resume -------------------
                current_input = await self._handle_hitl(
                    ws=ws,
                    thread_id=thread_id,
                    interrupt_value=self._interrupts_of(state_snapshot)[0].value,
                    recorder=recorder,
                    state_snapshot=state_snapshot,
                )
                if current_input is None:
                    # Client sent an invalid resume or disconnected
                    break
        finally:
            # T3 — WS drop or exception: await T0, project fresh, nothing else
            # (§5.2); the recorder is exception-native but the boundary is not.
            try:
                await recorder.on_abort()
            except Exception:  # noqa: BLE001 — cleanup must not mask the session error
                self._log.bind(thread_id=thread_id).warning("saul_t3_abort_failed")
            await lock.release()

    # ------------------------------------------------------------------
    # Event mapping: astream_events v2 → WS frames
    # ------------------------------------------------------------------

    @staticmethod
    def _map_event_to_frame(  # noqa: PLR0914
        event: object,
        current_step: int,
    ) -> WSOutbound | None:
        if not isinstance(event, dict):  # ponytail: LangGraph events are untyped objects
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
                [key for key in output if isinstance(key, str)] if isinstance(output, dict) else []
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

    @staticmethod
    def _interrupts_of(snapshot: StateSnapshot) -> list[Any]:
        """Pending interrupts across tasks (P5a over task objects)."""
        return [interrupt for task in snapshot.tasks for interrupt in task.interrupts]

    async def _stream_round(
        self,
        *,
        ws: WebSocket,
        config: RunnableConfig,
        current_input: Any,
        step: int,
    ) -> None:
        """Stream one graph round's events to the wire as WS frames."""
        async for event in self._graph.astream_events(
            current_input,
            config=config,
            version="v2",
        ):
            frame = self._map_event_to_frame(event=event, current_step=step)
            if frame is not None:
                await self._send_json(ws, frame.model_dump())

    async def _hydrate_or_refuse(
        self,
        ws: WebSocket,
        config: RunnableConfig,
        initial_input: dict[str, Any],
        thread_id: str,
    ) -> dict[str, Any] | None:
        """Hydrate checkpointed state; ``None`` means the session was refused.

        Persisted values may predate schema_version. Hydrate them before the
        first node sees state; shapes this runtime refuses fail the session
        loudly with a typed frame — never by resuming unmigrated state (§9).
        """
        persisted = await self._graph.aget_state(config)
        try:
            return (
                hydrate_state(dict(persisted.values))
                if persisted.values
                else hydrate_state(initial_input)
            )
        except StateSchemaVersionError as exc:
            exc.add_note(f"thread_id={thread_id}")
            self._log.bind(thread_id=thread_id, error=str(exc)).warning(
                "saul_session_state_schema_refused"
            )
            await self._send_json(
                ws,
                WSErrorFrame(
                    node=None,
                    code="STATE_SCHEMA_INCOMPATIBLE",
                    message="Saved session state is incompatible with this runtime; start a new session",
                    retryable=False,
                ).model_dump(),
            )
            return None

    async def _assess_round(
        self,
        *,
        ws: WebSocket,
        recorder: TurnRecorder,
        thread_id: str,
        config: RunnableConfig,
    ) -> tuple[str, StateSnapshot | None]:
        """Post-stream decision point: done / unexpected-pause / paused-HITL.

        Returns ``("paused", snapshot)`` for the HITL open; the other outcomes
        carry ``None`` and have already put the terminal frame on the wire.
        """
        state_snapshot = await self._graph.aget_state(config)

        # Graph has no pending nodes → pipeline completed
        if not state_snapshot.next:
            final_state = state_snapshot.values
            final_report = final_state.get("final_report")
            summary = final_report.summary if final_report else None
            await self._send_json(
                ws, WSDoneFrame(thread_id=thread_id, summary=summary).model_dump()
            )
            # T2 runs AFTER the DoneFrame is on the wire (§5.2).
            await recorder.on_end()
            self._log.info("saul_session_completed", thread_id=thread_id)
            return "done", None

        pause = self._interrupts_of(state_snapshot)
        if not pause:
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
            return "unexpected_pause", None
        return "paused", state_snapshot

    async def _gate_and_reconcile(
        self,
        *,
        ws: WebSocket,
        recorder: TurnRecorder,
        config: RunnableConfig,
        thread_id: str,
    ) -> bool:
        """Connect-time discipline (doc §12.5 / §6): reconcile then expiry.

        Reconcile projects whatever the checkpoint holds *before* anything new
        runs, so a crash mid-turn heals via the lineage keys, not replay. A
        paused turn within the 24 h window refuses the new start frame
        (SESSION_PAUSED_AWAITING_RESUME — resumption reconnects keep the
        thread); past the window the turn is closed abandoned_expired by the
        projector and the new start proceeds (close-and-proceed, D2).
        """
        await recorder.reconcile()
        snapshot = await self._graph.aget_state(config)
        paused = [interrupt for task in snapshot.tasks for interrupt in task.interrupts]
        if not paused:
            return True
        created_at = _coerce_ts(snapshot.created_at)
        if created_at is not None and (datetime.now(tz=UTC) - created_at) > PAUSE_WINDOW:
            self._log.bind(thread_id=thread_id).info("saul_paused_turn_expired")
            return True  # turn_closed(abandoned_expired) already projected above
        self._log.bind(thread_id=thread_id).info("saul_thread_paused_refusing_new_start")
        await self._send_json(
            ws,
            WSErrorFrame(
                node=None,
                code="SESSION_PAUSED_AWAITING_RESUME",
                message="This thread is paused awaiting HITL resume; a resume connection may claim it",
                retryable=True,
            ).model_dump(),
        )
        return False

    async def _refuse_thread(self, ws: WebSocket, thread_id: str) -> None:
        """THREAD_BUSY for a second connection; retryable refusal on outage."""
        code = "THREAD_BUSY"
        message = "Another connection is active on this thread"
        try:
            await self._redis.ping()
        except RedisError:
            code = "TRANSCRIPT_UNAVAILABLE"
            message = "Persistence services are unavailable; retry shortly"
        self._log.bind(thread_id=thread_id, code=code).warning("saul_thread_refused")
        await self._send_json(
            ws,
            WSErrorFrame(node=None, code=code, message=message, retryable=True).model_dump(),
        )

    async def _handle_hitl(
        self,
        ws: WebSocket,
        thread_id: str,
        interrupt_value: dict[str, Any],
        recorder: TurnRecorder,
        state_snapshot: object,
    ) -> Command[Any] | None:
        """Emit HITL frame, wait for WSResumeMessage, return Command(resume=...).

        T1 rides right after the frame is on the wire (never delaying the
        client), and project-before-resume precedes ``Command(resume=...)``
        with its single retry (§5.2). A paused turn past the 24 h window is
        refused with SESSION_PAUSE_EXPIRED (§8) — the projector closed it.
        """
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
        # T1 after the frame (§5.2); the client is reading either way.
        pause_result = await recorder.on_pause()
        if isinstance(pause_result, Failure):
            self._log.bind(thread_id=thread_id).warning("saul_t1_pause_project_failed")

        created_at = _coerce_ts(getattr(state_snapshot, "created_at", None))
        if created_at is not None and (datetime.now(tz=UTC) - created_at) > PAUSE_WINDOW:
            await self._send_json(
                ws,
                WSErrorFrame(
                    node=None,
                    code="SESSION_PAUSE_EXPIRED",
                    message="Paused turn exceeded its 24 h window and was closed",
                    retryable=False,
                ).model_dump(),
            )
            return None

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
            except (ValidationError, ValueError, TypeError) as exc:
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
            # T1' — project-before-resume (§5.2): retry once internally, then
            # proceed fail-open with a reconcile hint; never block the resume
            # on transcript health.
            resume_projection = await recorder.prepare_resume()
            if isinstance(resume_projection, Failure):
                self._log.bind(thread_id=thread_id).warning("saul_t1p_before_resume_project_failed")
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
            "schema_version": STATE_SCHEMA_VERSION,
            "doc_id": start_msg.doc_id,
            "document_text": None,
            # D4: the explicit saul_kind tag is the turn-boundary contract —
            # the projector derives turn existence purely from this tagged
            # HumanMessage surviving checkpoint serialization (verified
            # against the pinned JsonPlusSerializer).
            "messages": [
                HumanMessage(
                    content=start_msg.user_query,
                    additional_kwargs={"saul_kind": "query"},
                    id=str(uuid4()),
                )
            ],
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


def _coerce_ts(raw: object) -> datetime | None:
    """Checkpoint ``created_at`` arrives as an ISO string or None."""
    if raw is None:
        return None
    if isinstance(raw, datetime):
        return raw
    if isinstance(raw, str):
        try:
            parsed = datetime.fromisoformat(raw)
        except ValueError:
            return None
        return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=UTC)
    return None
