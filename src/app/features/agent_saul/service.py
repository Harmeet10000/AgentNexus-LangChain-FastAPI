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

from __future__ import annotations

import asyncio
import contextlib
from datetime import UTC, datetime
from typing import TYPE_CHECKING, cast
from uuid import uuid4

from fastapi import WebSocket
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command, StateSnapshot
from pydantic import ValidationError
from redis.exceptions import RedisError
from returns.result import Failure, Success

from app.features.agent_saul.dto import (
    CreateSessionResponse,
    WSDoneFrame,
    WSErrorFrame,
    WSHITLInterruptFrame,
    WSNodeEndFrame,
    WSNodeStartFrame,
    WSPingMessage,
    WSPongFrame,
    WSResumeMessage,
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
    WebSocketSecurityViolation,
)
from app.observability.logging import logger, trace_layer

if TYPE_CHECKING:
    from typing import Any

    from redis.asyncio import Redis
    from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

    from app.features.agent_saul.dto import (
        WSOutbound,
        WSStartMessage,
    )
    from app.features.agent_saul.exceptions import AgentSaulResult
    from app.features.auth.websocket_security import (
        WebSocketSecurityContext,
        WebSocketSecurityService,
    )

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
        start_msg: WSStartMessage | None,
        thread_id: str,
        user_id: str,
        first_resume: WSResumeMessage | None = None,
    ) -> None:
        """Main session loop.  Runs until graph completes or WS closes.

        Connection-lifetime discipline (doc §12.5): lock acquisition → hydrate
        gate → reconcile → (paused-thread refusal / expiry) → run loop. Every
        boundary update rides the recorder; ``finally`` runs T3 and releases
        the lock — the TTL fence is only for process death.

        ``first_resume`` carries a reconnect-resume (§6): the client's first
        frame is a resume for a paused thread; the gate re-derives the saved
        HITL prompt instead of starting a fresh turn (D2 governs start frames
        only).
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
                user_id=user_id,
                first_resume=first_resume,
            )
            if not gated:
                return
            # Reconnect-resume frames carry their own reply; start frames run
            # the graph from hydrated state (§6 / D2 split).
            setup = await self._session_first_input(
                ws=ws,
                start_msg=start_msg,
                first_resume=first_resume,
                recorder=recorder,
                config=config,
                thread_id=thread_id,
                user_id=user_id,
            )
            if setup is None:
                return
            # current_input alternates between:
            #   - dict (LegalAgentState subset) on first run
            #   - Command(resume=...) on each HITL resume
            current_input, has_initial_stream = setup

            step = 0
            first_stream_done = has_initial_stream
            while True:
                # Lease-loss guard between rounds (§5.5): a lost lock means
                # another connection may own this thread — stop before any
                # further writes. Mid-round loss is handled by the
                # lease_lost race inside each round below.
                # --- Stream graph events → WS frames ----------------------------
                if await self._stream_round_guarded(
                    ws=ws,
                    lock=lock,
                    config=config,
                    current_input=current_input,
                    step=step,
                    thread_id=thread_id,
                ):
                    break
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
                    config=config,
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

    async def _stream_round_guarded(
        self,
        *,
        ws: WebSocket,
        lock: Any,
        config: RunnableConfig,
        current_input: Any,
        step: int,
        thread_id: str,
    ) -> bool:
        """One stream round under the lease, True when the run must stop.

        The round runs as a task racing ``lock.lease_lost`` so a lost fence
        stops an ACTIVE graph execution, not just the gaps between rounds
        (§5.5): once the fence is gone a successor may resume this thread
        and produce checkpoints this run would trample. A round that dies
        normally propagates its exception (§9) — never masked as success.
        """
        stream_task = asyncio.create_task(
            self._stream_round(ws=ws, config=config, current_input=current_input, step=step),
            name=f"saul-stream-round-{thread_id}",
        )
        lost = asyncio.create_task(lock.lease_lost.wait())
        try:
            await asyncio.wait({stream_task, lost}, return_when=asyncio.FIRST_COMPLETED)
        finally:
            if not lost.done():
                lost.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await lost
        if lost.done() or not lock.is_alive():
            if not stream_task.done():
                stream_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await stream_task
            else:
                _ = stream_task.result()  # surface a failed round (§9)
            await self._send_json(
                ws,
                WSErrorFrame(
                    node=None,
                    code="LOCK_LOST",
                    message="Thread exclusion was lost; this run must stop",
                    retryable=False,
                ).model_dump(),
            )
            self._log.bind(thread_id=thread_id).warning("saul_lock_lost_aborting")
            return True
        # Stream finished while the lease held: re-raise its failure (§9).
        stream_task.result()
        return False

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
        """Schema-gate the thread before anything runs; None = refused.

        Persisted values may predate schema_version: hydrate a copy of them
        to version-check, refusing incompatible shapes with a typed frame —
        never by resuming unmigrated state (§9). Hydration is VALIDATION
        here, not run input: the graph starts from ``initial_input`` only.
        An empty delta with an empty checkpoint (nothing to gate) returns
        ``{}``.
        """
        persisted = await self._graph.aget_state(config)
        if persisted.values:
            # §9 gate only: persisted values ride along inside the checkpoint,
            # so hydration here exists to REFUSE incompatible shapes, not to
            # feed the run. The graph gets only this turn's delta — replaying
            # the full hydrated state would also clobber the pending-turn
            # channels (plan, current_step, …) the checkpoint legitimately
            # holds for a paused-then-resumed thread (§6).
            try:
                _ = hydrate_state(dict(persisted.values))
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
                        message=(
                            "Saved session state is incompatible with this runtime; "
                            "start a new session"
                        ),
                        retryable=False,
                    ).model_dump(),
                )
                return None
        if not initial_input:
            return {}  # empty delta + empty checkpoint: nothing to gate
        try:
            return hydrate_state(dict(initial_input))
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

    async def _resume_after_reconnect(
        self,
        ws: WebSocket,
        first_resume: WSResumeMessage,
        *,
        recorder: TurnRecorder,
        config: RunnableConfig,
        thread_id: str,
    ) -> Command[Any] | None:
        """§6 reconnect-resume: hydrate gate, re-emit the saved HITL frame,
        then treat the client's first-frame resume as its reply.

        Doc §6 248: connect, hydrate gate, identity check, reconcile, expiry
        (all done in the gate), re-emit the HITL frame re-derived from
        ``state.tasks`` — the reconnect client's ``resume`` frame IS the reply
        to that re-emitted prompt. Returns the ``Command(resume=...)`` for the
        loop, or ``None`` when the reconnect was refused.
        """
        snapshot = await self._graph.aget_state(config)
        interrupts = self._interrupts_of(snapshot)
        if not interrupts:
            await self._send_json(
                ws,
                WSErrorFrame(
                    node=None,
                    code="NOT_PAUSED",
                    message="Thread has no pending interrupt to resume",
                    retryable=False,
                ).model_dump(),
            )
            return None
        if first_resume.thread_id != thread_id:
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
        # hydrate_state gate (§6 248): refuse incompatible persisted state
        # before any resume payload is applied.
        hydrated = await self._hydrate_or_refuse(ws, config, {}, thread_id)
        if hydrated is None:
            return None
        await self._emit_hitl_prompt(ws, thread_id, interrupts[0].value, recorder, snapshot)
        return await self._apply_resume(
            first_resume, ws=ws, thread_id=thread_id, recorder=recorder, config=config
        )

    async def _session_first_input(
        self,
        *,
        ws: WebSocket,
        start_msg: WSStartMessage | None,
        first_resume: WSResumeMessage | None,
        recorder: TurnRecorder,
        config: RunnableConfig,
        thread_id: str,
        user_id: str,
    ) -> tuple[Any, bool] | None:
        """First graph input for the session, plus whether output already streamed.

        - reconnect-resume (§6): re-emit the saved HITL prompt and treat the
          client's first-frame resume as its reply → ``Command(resume=...)``;
          the checkpoint already holds this turn's prior output.
        - fresh start (D2): hydrate-gate the initial state and run it.
        ``None`` means the session was refused and the refusal is on the wire.
        """
        if first_resume is not None:
            command = await self._resume_after_reconnect(
                ws,
                first_resume,
                recorder=recorder,
                config=config,
                thread_id=thread_id,
            )
            if command is None:
                return None
            return command, True
        if start_msg is None:
            # Fresh turn without a start frame is a protocol violation — the
            # router only delivers start here when no reconnect-resume exists.
            await self._send_json(
                ws,
                WSErrorFrame(
                    node=None,
                    code="INVALID_FIRST_FRAME",
                    message="Fresh turn requires a 'start' frame",
                    retryable=False,
                ).model_dump(),
            )
            return None
        initial_input = await self._hydrate_or_refuse(
            ws,
            config,
            self._build_initial_state(start_msg, thread_id, user_id),
            thread_id,
        )
        if initial_input is None:
            return None
        return initial_input, False

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
        user_id: str,
        first_resume: WSResumeMessage | None = None,
    ) -> bool:
        """Connect-time discipline (doc §12.5 / §6): reconcile then expiry.

        Reconcile projects whatever the checkpoint holds *before* anything new
        runs, so a crash mid-turn heals via the lineage keys, not replay. On a
        paused thread:

        - ``start`` frame within the 24 h window → refuse
          SESSION_PAUSED_AWAITING_RESUME (D2 — a resume claim, not a new turn).
        - ``start`` frame past the window → the turn is closed
          abandoned_expired by the reconcile projection above and the new
          start proceeds (close-and-proceed, D2).
        - ``resume`` frame with a valid claim and a paused interrupt within
          the window → proceed to the resume-reconnect path (§6).
        - ``resume`` frame past the window → SESSION_PAUSE_EXPIRED: the turn
          is already closed by the projector; nothing may resume (§8).
        - anything else on a paused thread → refused. The OWNER identity is
          gated for EVERY first frame above, start and resume alike (§9).
        """
        await recorder.reconcile()
        snapshot = await self._graph.aget_state(config)

        # Identity gate — EVERY first frame (§9): the state's owner is the
        # only user this thread may serve, before anything about its state
        # (paused or not) is revealed. Threads with no persisted owner yet
        # (clean checkpoint lineage) gate on acquisition alone.
        state_user = snapshot.values.get("user_id") if snapshot.values else None
        if state_user is not None and str(state_user) != user_id:
            self._log.bind(thread_id=thread_id).warning("saul_first_frame_identity_refused")
            await self._send_json(
                ws,
                WSErrorFrame(
                    node=None,
                    code="THREAD_NOT_OWNED",
                    message="This thread belongs to another user",
                    retryable=False,
                ).model_dump(),
            )
            return False

        paused = [interrupt for task in snapshot.tasks for interrupt in task.interrupts]
        if not paused:
            if first_resume is not None:
                await self._send_json(
                    ws,
                    WSErrorFrame(
                        node=None,
                        code="NOT_PAUSED",
                        message="Thread has no pending interrupt to resume",
                        retryable=False,
                    ).model_dump(),
                )
                return False
            return True

        created_at = _coerce_ts(snapshot.created_at)
        past_window = created_at is not None and (datetime.now(tz=UTC) - created_at) > PAUSE_WINDOW

        if first_resume is None:
            if past_window:
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

        # Resume-reconnect claim: the window decides next (§8) — the owner
        # identity already passed above (same gate as the start frame).
        if past_window:
            self._log.bind(thread_id=thread_id).info("saul_paused_turn_expired_resume_refused")
            await self._send_json(
                ws,
                WSErrorFrame(
                    node=None,
                    code="SESSION_PAUSE_EXPIRED",
                    message="Paused turn exceeded its 24 h window and was closed",
                    retryable=False,
                ).model_dump(),
            )
            return False
        # Inside the window: the normal loop resumes after re-derivation —
        # the schema gate and HITL re-emit live in _resume_after_reconnect.
        return True

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

    async def _emit_hitl_prompt(
        self,
        ws: WebSocket,
        thread_id: str,
        interrupt_value: dict[str, Any],
        recorder: TurnRecorder,
        state_snapshot: object,
    ) -> bool:
        """Re-emit the saved HITL frame (live loop and reconnect-resume, §6).

        T1 rides right after the frame is on the wire (never delaying the
        client). Returns False when the pre-wait expiry check already closed
        the turn and put SESSION_PAUSE_EXPIRED on the wire (§8).
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
            return False

        self._log.info(
            "saul_hitl_interrupt_sent",
            thread_id=thread_id,
            interrupt_type=interrupt_type,
        )
        return True

    async def _apply_resume(
        self,
        inbound: WSResumeMessage,
        *,
        ws: WebSocket,
        thread_id: str,
        recorder: TurnRecorder,
        config: RunnableConfig,
    ) -> Command[Any] | None:
        """Turn a validated resume frame into ``Command(resume=...)`` (§5.2).

        Shared by the live receive loop and the reconnect-resume first frame.
        A resume arriving past the 24 h window is refused with
        SESSION_PAUSE_EXPIRED — the projector closed the turn (§8).
        """
        self._log.info(
            "saul_hitl_resume_received",
            thread_id=thread_id,
            action=inbound.action,
        )

        # §8 expiry authority: the deadline is re-checked from the FRESH
        # snapshot at the actual resume attempt — a reply that arrives
        # after the 24 h window (the user walked away with the prompt on
        # screen) still closes the turn instead of resuming it.
        fresh_snapshot = await self._graph.aget_state(config)
        fresh_created = _coerce_ts(getattr(fresh_snapshot, "created_at", None))
        if fresh_created is not None and (datetime.now(tz=UTC) - fresh_created) > PAUSE_WINDOW:
            await recorder.reconcile()  # appends turn_closed(abandoned_expired)
            self._log.bind(thread_id=thread_id).info("saul_resume_after_expiry_refused")
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

    async def _handle_hitl(
        self,
        ws: WebSocket,
        thread_id: str,
        interrupt_value: dict[str, Any],
        recorder: TurnRecorder,
        state_snapshot: object,
        *,
        config: RunnableConfig,
    ) -> Command[Any] | None:
        """Emit HITL frame, wait for WSResumeMessage, return Command(resume=...).

        A paused turn past the 24 h window is refused with
        SESSION_PAUSE_EXPIRED (§8) — the projector closed it.
        """
        delivered = await self._emit_hitl_prompt(
            ws, thread_id, interrupt_value, recorder, state_snapshot
        )
        if not delivered:
            return None

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

            return await self._apply_resume(
                inbound, ws=ws, thread_id=thread_id, recorder=recorder, config=config
            )

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
