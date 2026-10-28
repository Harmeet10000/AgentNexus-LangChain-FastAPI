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
from collections.abc import Mapping
from datetime import UTC, datetime
from typing import TYPE_CHECKING, assert_never, cast
from uuid import uuid4

from fastapi import WebSocket
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command, StateSnapshot
from pydantic import ValidationError
from redis.exceptions import RedisError
from returns.result import Failure, Success

from app.config import get_settings
from app.features.agent_saul.budget import BudgetLedger, BudgetLimits
from app.features.agent_saul.document_source import (
    DOCUMENT_SOURCE_CONFIG_KEY,
    DocumentReadiness,
    DocumentTextFailureCode,
    document_content_source_from_config,
    postgres_document_content_source,
)
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
from app.features.agent_saul.exceptions import (
    AgentSaulDocumentIngestionFailedError,
    AgentSaulDocumentNotFoundError,
    AgentSaulDocumentNotReadyError,
    AgentSaulInfrastructureError,
    AgentSaulValidationError,
)
from app.features.agent_saul.graph.state import (
    GRAPH_NODE_NAMES,
    STATE_SCHEMA_VERSION,
    HITLInterruptType,
    PlanStep,
    StateSchemaVersionError,
    WorkflowStatus,
    hydrate_state,
    initial_ingestion_state,
    legacy_resume_update,
)
from app.features.agent_saul.runtime import thread_owner_violates
from app.features.agent_saul.thread_lock import ThreadLock
from app.features.agent_saul.transcript.projection import PAUSE_WINDOW
from app.features.agent_saul.transcript.recorder import PlanResumeEvidence, TurnRecorder
from app.features.agent_saul.transcript.repository import ChatTranscriptRepository
from app.features.auth.tenancy import LegalRole
from app.features.auth.websocket_security import (
    WebSocketIdleTimeout,
    WebSocketSecurityViolation,
)
from app.kernel.result import log_expected_failure
from app.observability.logging import logger, trace_layer

if TYPE_CHECKING:
    from typing import Any

    from redis.asyncio import Redis
    from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

    from app.features.agent_saul.document_source import (
        DocumentContentResult,
        DocumentContentSource,
        DocumentTextFailure,
    )
    from app.features.agent_saul.dto import (
        WSOutbound,
        WSStartMessage,
    )
    from app.features.agent_saul.exceptions import AgentSaulError, AgentSaulResult
    from app.features.agent_saul.runtime import SaulThreadContext
    from app.features.auth.account_store import AuthAccountStore
    from app.features.auth.tenant_repository import TenantMembershipService
    from app.features.auth.websocket_security import (
        WebSocketSecurityContext,
        WebSocketSecurityService,
    )

# Status emitted when a node starts — keeps WS client progress bar accurate.
_NODE_STATUS_MAP: dict[str, WorkflowStatus] = {
    "gateway": WorkflowStatus.INITIALIZED,
    "qna": WorkflowStatus.QNA_CLARIFICATION,
    "orchestrator": WorkflowStatus.PLAN_PENDING,
    "planner_generate": WorkflowStatus.PLAN_PENDING,
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


def _document_preflight_error(
    failure: DocumentTextFailure,
    *,
    user_id: str,
    doc_id: str,
) -> AgentSaulError:
    """Document-content source failure → agent_saul's HTTP-facing union.

    Same user-scoping semantics as the WS thread owner gate: a foreign
    doc_id reads as NOT_FOUND, revealing nothing about other users' rows.
    """
    match failure.code:
        case DocumentTextFailureCode.NOT_FOUND:
            return AgentSaulDocumentNotFoundError(
                message=failure.message or "Requested document does not exist for this user",
                details={"source": "document_content_source"},
                doc_id=doc_id,
                user_id=user_id,
            )
        case DocumentTextFailureCode.UNAVAILABLE:
            return AgentSaulInfrastructureError(
                message="Document content source is unavailable",
                details={"doc_id": doc_id, "user_id": user_id},
            )
        case DocumentTextFailureCode.EMPTY:
            return AgentSaulValidationError(
                message="Requested document has no retrievable content",
                details={"doc_id": doc_id, "user_id": user_id},
            )
        case _ as unreachable:
            assert_never(unreachable)


def _document_readiness_error(
    readiness: DocumentReadiness,
    *,
    user_id: str,
    doc_id: str,
) -> AgentSaulError | None:
    """Translate a persisted ingestion state into a session admission error."""
    if readiness.status in {"completed", "completed_with_warnings"}:
        return None
    details: dict[str, object] = {
        "doc_id": doc_id,
        "user_id": user_id,
        "ingestion_status": readiness.status,
        "warning_count": readiness.warning_count,
        "warnings": list(readiness.warnings),
    }
    if readiness.error_code is not None:
        details["error_code"] = readiness.error_code
    if readiness.error_message is not None:
        details["error_message"] = readiness.error_message
    if readiness.status == "failed":
        return AgentSaulDocumentIngestionFailedError(
            message=readiness.error_message or "Document ingestion failed",
            details=details,
            source="document_content_source",
            doc_id=doc_id,
            status=readiness.status,
        )
    return AgentSaulDocumentNotReadyError(
        message="Document ingestion is not complete",
        details=details,
        source="document_content_source",
        doc_id=doc_id,
        status=readiness.status,
    )


async def _source_readiness(
    source: DocumentContentSource,
    *,
    user_id: str,
    doc_id: str,
    tenant_id: str | None,
) -> DocumentContentResult[DocumentReadiness]:
    """Read readiness while keeping old source doubles source-compatible.

    The protocol now includes ``check_readiness``.  The compatibility fallback
    is intentionally only for injected legacy test doubles and old deployments;
    the production Postgres adapter always implements the status query.
    """
    checker = getattr(source, "check_readiness", None)
    if not callable(checker):
        return Success(DocumentReadiness(status="completed"))
    kwargs: dict[str, str] = {"user_id": user_id, "doc_id": doc_id}
    if tenant_id is not None:
        kwargs["tenant_id"] = tenant_id
    return await cast("Any", checker)(**kwargs)


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
        context: SaulThreadContext,
        ws_security: WebSocketSecurityService,
        ws_context: WebSocketSecurityContext,
        *,
        session_factory: async_sessionmaker[AsyncSession],
        account_store: AuthAccountStore | None = None,
        tenant_membership_service: TenantMembershipService | None = None,
    ) -> None:
        self._graph = graph
        self._redis = redis
        self._context = context
        self._correlation_id = context.correlation_id
        self._ws_security = ws_security
        self._ws_context = ws_context
        self._session_factory = session_factory
        self._account_store = account_store
        self._tenant_membership_service = tenant_membership_service
        self._log = logger.bind(correlation_id=context.correlation_id)

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    @staticmethod
    def create_session(
        *, websocket_url: str, thread_id: str
    ) -> AgentSaulResult[CreateSessionResponse]:
        """Create the HTTP pre-flight response before entering the WS boundary."""
        return Success(CreateSessionResponse(thread_id=thread_id, ws_url=websocket_url))

    @staticmethod
    async def create_session_for_document(
        *,
        websocket_url: str,
        thread_id: str,
        doc_id: str,
        user_id: str,
        document_source: DocumentContentSource,
        tenant_id: str | None = None,
        allowed_doc_ids: list[str] | None = None,
    ) -> AgentSaulResult[CreateSessionResponse]:
        """HTTP pre-flight with document existence + ownership validation.

        The gateway read style keeps shape-only doc_id checks at WS admission;
        this endpoint closes the hole where a nonexistent or foreign doc_id
        sailed through to a turn that fails at ingestion. A rejected doc is a
        typed failure on the Result, never a raise.
        """
        requested_ids = list(dict.fromkeys(allowed_doc_ids or [doc_id]))
        if doc_id not in requested_ids:
            requested_ids.insert(0, doc_id)
        for requested_doc_id in requested_ids:
            kwargs: dict[str, str] = {"user_id": user_id, "doc_id": requested_doc_id}
            if tenant_id is not None:
                kwargs["tenant_id"] = tenant_id
            preflight = await document_source.ensure_document(**kwargs)
            if isinstance(preflight, Failure):
                failure = preflight.failure()
                error = _document_preflight_error(failure, user_id=user_id, doc_id=requested_doc_id)
                log_expected_failure(error, operation="create_session_document_preflight")
                return Failure(error)
            readiness = await _source_readiness(
                document_source,
                user_id=user_id,
                doc_id=requested_doc_id,
                tenant_id=tenant_id,
            )
            if isinstance(readiness, Failure):
                failure = readiness.failure()
                error = _document_preflight_error(failure, user_id=user_id, doc_id=requested_doc_id)
                log_expected_failure(error, operation="create_session_readiness_preflight")
                return Failure(error)
            readiness_error = _document_readiness_error(
                readiness.unwrap(), user_id=user_id, doc_id=requested_doc_id
            )
            if readiness_error is not None:
                log_expected_failure(
                    readiness_error, operation="create_session_readiness_preflight"
                )
                return Failure(readiness_error)
        return AgentSaulService.create_session(websocket_url=websocket_url, thread_id=thread_id)

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

        Connection-lifetime discipline (doc §12.5): lock acquisition → persisted
        owner gate → reconcile → discovery bootstrap → hydrate/input gate → run
        loop. Every boundary update rides the recorder; ``finally`` runs T3 only
        after bootstrap and always releases the lock — the TTL fence is only
        for process death.

        ``first_resume`` carries a reconnect-resume (§6): the client's first
        frame is a resume for a paused thread; the gate re-derives the saved
        HITL prompt instead of starting a fresh turn (D2 governs start frames
        only).
        """
        config = self._build_config(thread_id)

        lock = ThreadLock(self._redis, thread_id)
        recorder = TurnRecorder(
            repo=ChatTranscriptRepository(self._session_factory),
            graph=self._graph,
            config=config,
            thread_id=thread_id,
            user_id=user_id,
            redis=self._redis,
            lock=lock,
            timeout_ms=get_settings().SAUL_PROJECTION_TIMEOUT_MS,
        )
        config["callbacks"] = [recorder.usage_handler]

        acquired = await lock.acquire()
        if not acquired:
            await self._refuse_thread(ws, thread_id)
            return

        transcript_ready = False
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
            bootstrap = await recorder.bootstrap_session()
            if isinstance(bootstrap, Failure):
                error = bootstrap.failure()
                await self._send_json(
                    ws,
                    WSErrorFrame(
                        node=None,
                        code="SESSION_BOOT_FAILED",
                        message="Transcript could not be initialised; run refused",
                        retryable=True,
                    ).model_dump(),
                )
                self._log.bind(
                    thread_id=thread_id,
                    error_code=str(error.code),
                    error_operation=getattr(error, "operation", None),
                ).warning("saul_t0_refused_run")
                return
            transcript_ready = True
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
            current_input = setup[0]

            step = 0
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
            # T3 may only project after the owner-bound discovery anchor is
            # established. A refused foreign orphan must not be claimed by
            # cleanup (§5.2 / §9).
            if transcript_ready:
                try:
                    await recorder.on_abort()
                except Exception as exc:  # noqa: BLE001 — cleanup must not mask session error
                    exc.add_note(f"thread_id={thread_id}, operation=t3_abort")
                    self._log.bind(
                        thread_id=thread_id,
                        error=str(exc),
                        error_notes=list(getattr(exc, "__notes__", ())),
                    ).warning("saul_t3_abort_failed")
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
            # The EVENT is the loss signal — the waiter task's done-ness is
            # not: the cleanup below cancels (and so completes) it even when
            # the stream just finished under a healthy lease.
            if lock.lease_lost.is_set() or not lock.is_alive():
                if not stream_task.done():
                    # Stop an ACTIVE graph execution BEFORE the frame goes
                    # out: once the fence is gone a successor may resume this
                    # thread and produce checkpoints this run would trample.
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
            # Stream finished while the lease held: surface its failure (§9)
            # — never mask a graph error as a successful round.
            stream_task.result()
            return False
        finally:
            # Cleanup on EVERY exit path — including the guard itself being
            # cancelled (client disconnect): run_session's cleanup releases
            # the lock, and a still-running stream would then write
            # checkpoints on a thread whose fence is gone.
            for task in (lost, stream_task):
                if not task.done():
                    task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await task

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
        await self._emit_hitl_prompt(
            ws, thread_id, interrupts[0].value, recorder, snapshot, config=config
        )
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
        scope_error = await self._validate_document_scope(start_msg, config, user_id)
        if scope_error is not None:
            readiness_code = (
                "DOCUMENT_INGESTION_FAILED"
                if scope_error.startswith("Document ingestion failed")
                else "DOCUMENT_NOT_READY"
                if scope_error.startswith("Document ingestion is not complete")
                else "DOCUMENT_READINESS_UNAVAILABLE"
                if scope_error.startswith("Document ingestion status is unavailable")
                else "DOCUMENT_SCOPE_INVALID"
            )
            await self._send_json(
                ws,
                WSErrorFrame(
                    node="gateway",
                    code=readiness_code,
                    message=scope_error,
                    retryable=readiness_code
                    in {"DOCUMENT_NOT_READY", "DOCUMENT_READINESS_UNAVAILABLE"},
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

    async def _validate_document_scope(
        self, start_msg: WSStartMessage, config: RunnableConfig, user_id: str
    ) -> str | None:
        """Prove every requested document belongs to the live tenant.

        The browser may select IDs, but it cannot widen the tenant boundary.
        A single-document legacy client follows the same check as a new
        multi-document client.
        """
        source = document_content_source_from_config(config)
        if source is None:
            return "Document content source is unavailable"
        authorized_doc_ids = set(self._context.allowed_doc_ids)
        for doc_id in start_msg.allowed_doc_ids:
            if doc_id not in authorized_doc_ids:
                return "One or more selected documents are outside the authorized corpus"
            kwargs: dict[str, str] = {"user_id": user_id, "doc_id": doc_id}
            if self._context.tenant_id:
                kwargs["tenant_id"] = self._context.tenant_id
            result = await source.ensure_document(**kwargs)
            if isinstance(result, Failure):
                return "One or more selected documents are unavailable"
            readiness = await _source_readiness(
                source,
                user_id=user_id,
                doc_id=doc_id,
                tenant_id=self._context.tenant_id or None,
            )
            if isinstance(readiness, Failure):
                return "Document ingestion status is unavailable"
            current = readiness.unwrap()
            if current.status == "failed":
                return f"Document ingestion failed for {doc_id}" + (
                    f": {current.error_message}" if current.error_message else ""
                )
            if current.status not in {"completed", "completed_with_warnings"}:
                warning_suffix = (
                    f" ({current.warning_count} warnings)" if current.warning_count else ""
                )
                return f"Document ingestion is not complete for {doc_id}{warning_suffix}"
        return None

    async def _document_scope_readiness_error(
        self,
        *,
        config: RunnableConfig,
        user_id: str,
        doc_ids: list[str],
    ) -> AgentSaulError | None:
        """Re-check ownership and terminal ingestion state for a saved scope."""
        source = document_content_source_from_config(config)
        if source is None:
            return AgentSaulInfrastructureError(
                message="Document content source is unavailable",
                details={"thread_id": self._context.thread_id},
                source="document_content_source",
            )
        for doc_id in doc_ids:
            kwargs: dict[str, str] = {"user_id": user_id, "doc_id": doc_id}
            if self._context.tenant_id:
                kwargs["tenant_id"] = self._context.tenant_id
            presence = await source.ensure_document(**kwargs)
            if isinstance(presence, Failure):
                return _document_preflight_error(presence.failure(), user_id=user_id, doc_id=doc_id)
            readiness = await _source_readiness(
                source,
                user_id=user_id,
                doc_id=doc_id,
                tenant_id=self._context.tenant_id or None,
            )
            if isinstance(readiness, Failure):
                return _document_preflight_error(
                    readiness.failure(), user_id=user_id, doc_id=doc_id
                )
            error = _document_readiness_error(readiness.unwrap(), user_id=user_id, doc_id=doc_id)
            if error is not None:
                return error
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
            review = final_state.get("human_review")
            await self._send_json(
                ws,
                WSDoneFrame(
                    thread_id=thread_id,
                    summary=summary,
                    self_reviewed=bool(
                        final_report.self_reviewed
                        if final_report is not None
                        else final_state.get("self_reviewed", False)
                    ),
                    reviewer_id=(
                        final_report.reviewer_id
                        if final_report is not None
                        else getattr(review, "reviewer_id", None)
                    ),
                    reviewer_role=(
                        final_report.reviewer_role
                        if final_report is not None
                        else getattr(review, "reviewer_role", None)
                    ),
                ).model_dump(),
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

    async def _gate_and_reconcile(  # noqa: PLR0912, PLR0914, PLR0915 — connect-time owner/status/HITL gate
        self,
        *,
        ws: WebSocket,
        recorder: TurnRecorder,
        config: RunnableConfig,
        thread_id: str,
        user_id: str,
        first_resume: WSResumeMessage | None = None,
    ) -> bool:
        """Connect-time discipline (doc §12.5 / §6): owner gate, reconcile, expiry.

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
        snapshot = await self._graph.aget_state(config)

        # Identity gate — EVERY first frame (§9): the state's owner is the
        # only user this thread may serve, before anything about its state
        # (paused or not) is revealed. Threads with no persisted owner yet
        # (clean checkpoint lineage) gate on acquisition alone.
        state_user = snapshot.values.get("user_id") if snapshot.values else None
        if thread_owner_violates(state_user, user_id):
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
        state_tenant = snapshot.values.get("tenant_id") if snapshot.values else None
        if state_tenant and self._context.tenant_id and state_tenant != self._context.tenant_id:
            await self._send_json(
                ws,
                WSErrorFrame(
                    node=None,
                    code="TENANT_SCOPE_MISMATCH",
                    message="This thread belongs to another tenant",
                    retryable=False,
                ).model_dump(),
            )
            return False

        raw_persisted_ids = snapshot.values.get("allowed_doc_ids") if snapshot.values else None
        persisted_ids = list(raw_persisted_ids) if isinstance(raw_persisted_ids, list) else []
        if not persisted_ids and snapshot.values:
            legacy_doc_id = snapshot.values.get("doc_id")
            if isinstance(legacy_doc_id, str) and legacy_doc_id:
                persisted_ids = [legacy_doc_id]
        if persisted_ids:
            source = document_content_source_from_config(config)
            if source is None:
                await self._send_json(
                    ws,
                    WSErrorFrame(
                        node="gateway",
                        code="DOCUMENT_SCOPE_INVALID",
                        message="Document content source is unavailable",
                        retryable=True,
                    ).model_dump(),
                )
                return False
            for doc_id in persisted_ids:
                if str(doc_id) not in set(self._context.allowed_doc_ids):
                    await self._send_json(
                        ws,
                        WSErrorFrame(
                            node="gateway",
                            code="DOCUMENT_SCOPE_INVALID",
                            message="A saved document is no longer available to this tenant",
                            retryable=False,
                        ).model_dump(),
                    )
                    return False
                kwargs: dict[str, str] = {"user_id": user_id, "doc_id": str(doc_id)}
                if self._context.tenant_id:
                    kwargs["tenant_id"] = self._context.tenant_id
                scope_result = await source.ensure_document(**kwargs)
                if isinstance(scope_result, Failure):
                    await self._send_json(
                        ws,
                        WSErrorFrame(
                            node="gateway",
                            code="DOCUMENT_SCOPE_INVALID",
                            message="A saved document is no longer available to this tenant",
                            retryable=False,
                        ).model_dump(),
                    )
                    return False
                readiness = await _source_readiness(
                    source,
                    user_id=user_id,
                    doc_id=str(doc_id),
                    tenant_id=self._context.tenant_id or None,
                )
                if isinstance(readiness, Failure):
                    await self._send_json(
                        ws,
                        WSErrorFrame(
                            node="gateway",
                            code="DOCUMENT_READINESS_UNAVAILABLE",
                            message="Document ingestion status is unavailable; retry shortly",
                            retryable=True,
                        ).model_dump(),
                    )
                    return False
                current = readiness.unwrap()
                if current.status not in {"completed", "completed_with_warnings"}:
                    code = (
                        "DOCUMENT_INGESTION_FAILED"
                        if current.status == "failed"
                        else "DOCUMENT_NOT_READY"
                    )
                    message = current.error_message or (
                        "Document ingestion is not complete; retry after ingestion finishes"
                    )
                    await self._send_json(
                        ws,
                        WSErrorFrame(
                            node="gateway",
                            code=code,
                            message=message,
                            retryable=current.status != "failed",
                        ).model_dump(),
                    )
                    return False

        # Only an authenticated owner may let checkpoint evidence reach the
        # transcript repository. In particular, an orphan checkpoint owned by
        # another user must be refused before reconciliation can create its
        # discovery row under the connecting user's identity.
        await recorder.reconcile()
        snapshot = await self._graph.aget_state(config)

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

    async def _persist_budget_pause(
        self, *, config: RunnableConfig, state_snapshot: object, thread_id: str
    ) -> bool:
        """Persist the active wall-clock segment at the actual interrupt boundary."""
        values = getattr(state_snapshot, "values", None)
        raw = values.get("budget_ledger") if isinstance(values, dict) else None
        if raw is None:
            return True
        try:
            ledger = raw if isinstance(raw, BudgetLedger) else BudgetLedger.model_validate(raw)
            paused = ledger.pause()
            updater = getattr(self._graph, "aupdate_state", None)
            if paused == ledger or not callable(updater):
                return True
            await updater(config, {"budget_ledger": paused})
        except Exception as exc:  # noqa: BLE001 — pause must fail closed
            exc.add_note(f"thread_id={thread_id}, operation=budget_pause")
            self._log.bind(thread_id=thread_id).exception("saul_budget_pause_persist_failed")
            return False
        else:
            return True

    async def _persist_budget_resume(
        self, *, config: RunnableConfig, state_snapshot: object, thread_id: str
    ) -> bool:
        """Start a new active segment only after a validated resume."""
        values = getattr(state_snapshot, "values", None)
        raw = values.get("budget_ledger") if isinstance(values, dict) else None
        if raw is None:
            return True
        try:
            ledger = raw if isinstance(raw, BudgetLedger) else BudgetLedger.model_validate(raw)
            resumed = ledger.resume()
            updater = getattr(self._graph, "aupdate_state", None)
            if resumed == ledger or not callable(updater):
                return True
            await updater(config, {"budget_ledger": resumed})
        except Exception as exc:  # noqa: BLE001 — resume must fail closed
            exc.add_note(f"thread_id={thread_id}, operation=budget_resume")
            self._log.bind(thread_id=thread_id).exception("saul_budget_resume_persist_failed")
            return False
        else:
            return True

    async def _emit_hitl_prompt(
        self,
        ws: WebSocket,
        thread_id: str,
        interrupt_value: dict[str, Any],
        recorder: TurnRecorder,
        state_snapshot: object,
        *,
        config: RunnableConfig,
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
        if not await self._persist_budget_pause(
            config=config, state_snapshot=state_snapshot, thread_id=thread_id
        ):
            return False
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

    async def _apply_resume(  # noqa: PLR0912, PLR0914, PLR0915 — resume authorization and evidence gate
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

        if not _recorder_resume_alive(recorder):
            await self._send_lock_lost(ws, thread_id)
            return None

        # §8 expiry authority: the deadline is re-checked from the FRESH
        # snapshot at the actual resume attempt — a reply that arrives
        # after the 24 h window (the user walked away with the prompt on
        # screen) still closes the turn instead of resuming it.
        fresh_snapshot = await self._graph.aget_state(config)
        fresh_values = fresh_snapshot.values if isinstance(fresh_snapshot.values, dict) else {}
        persisted_ids = fresh_values.get("allowed_doc_ids")
        if not isinstance(persisted_ids, list) or not persisted_ids:
            legacy_doc_id = fresh_values.get("doc_id")
            persisted_ids = [legacy_doc_id] if isinstance(legacy_doc_id, str) else []
        session_context = getattr(self, "_context", None)
        readiness_error = (
            await self._document_scope_readiness_error(
                config=config,
                user_id=str(fresh_values.get("user_id") or session_context.user_id),
                doc_ids=[str(doc_id) for doc_id in persisted_ids if isinstance(doc_id, str)],
            )
            if session_context is not None
            else None
        )
        if readiness_error is not None:
            details = readiness_error.details or {}
            status = str(details.get("ingestion_status") or "unavailable")
            code = (
                "DOCUMENT_INGESTION_FAILED"
                if isinstance(readiness_error, AgentSaulDocumentIngestionFailedError)
                else "DOCUMENT_NOT_READY"
                if isinstance(readiness_error, AgentSaulDocumentNotReadyError)
                else "DOCUMENT_READINESS_UNAVAILABLE"
            )
            await self._send_json(
                ws,
                WSErrorFrame(
                    node="gateway",
                    code=code,
                    message=readiness_error.message,
                    retryable=readiness_error.retryable,
                ).model_dump(),
            )
            self._log.bind(
                thread_id=thread_id,
                ingestion_status=status,
                error_code=str(readiness_error.code),
            ).warning("saul_resume_document_readiness_refused")
            return None
        live_roles, live_self_review, live_revision = await self._live_authorization()
        if session_context is not None and live_revision != session_context.role_revision:
            await self._send_json(
                ws,
                WSErrorFrame(
                    node="human_review",
                    code="AUTHORIZATION_CHANGED",
                    message="Authorization changed; reconnect with a fresh session",
                    retryable=False,
                ).model_dump(),
            )
            return None
        recorded_version = fresh_values.get("schema_version")
        if recorded_version not in {1, STATE_SCHEMA_VERSION - 1, STATE_SCHEMA_VERSION}:
            await self._send_json(
                ws,
                WSErrorFrame(
                    node=None,
                    code="STATE_SCHEMA_INCOMPATIBLE",
                    message="Paused state is not a supported resumable schema",
                    retryable=False,
                ).model_dump(),
            )
            return None
        fresh_interrupts = self._interrupts_of(fresh_snapshot)
        interrupt_type = _interrupt_type_from_snapshot(fresh_snapshot)
        if (
            live_roles is not None
            and interrupt_type
            in {HITLInterruptType.PLAN_APPROVAL, HITLInterruptType.HUMAN_REVIEW_REQUIRED}
            and inbound.action in {"approve", "reject", "modify"}
        ):
            if LegalRole.REVIEWER.value not in live_roles:
                await self._send_json(
                    ws,
                    WSErrorFrame(
                        node="human_review",
                        code="REVIEWER_ROLE_REQUIRED",
                        message="A live tenant reviewer role is required for approval or overrides",
                        retryable=False,
                    ).model_dump(),
                )
                return None
            if (
                not live_self_review
                and session_context is not None
                and str(session_context.user_id) == str(fresh_values.get("user_id"))
            ):
                await self._send_json(
                    ws,
                    WSErrorFrame(
                        node="human_review",
                        code="SELF_REVIEW_DISABLED",
                        message="This tenant requires an independent reviewer",
                        retryable=False,
                    ).model_dump(),
                )
                return None
        checkpoint_id = _snapshot_checkpoint_id(fresh_snapshot)
        decision_modified = (
            False
            if inbound.action == "approve"
            else True
            if inbound.action == "modify" and bool(inbound.modified_plan)
            else None
        )
        plan_resume = (
            PlanResumeEvidence._from_interrupt(
                fresh_interrupts[0],
                modified=decision_modified is True,
                checkpoint_id=checkpoint_id,
            )
            if fresh_interrupts and decision_modified is not None
            else None
        )
        if (
            decision_modified is not None
            and fresh_interrupts
            and _is_plan_approval_interrupt(fresh_interrupts[0])
            and plan_resume is None
        ):
            await self._send_json(
                ws,
                WSErrorFrame(
                    node="planner",
                    code="RESUME_EVIDENCE_INVALID",
                    message="Current plan approval cannot be bound to its checkpoint",
                    retryable=False,
                ).model_dump(),
            )
            return None
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

        legacy_plan, legacy_plan_error = _legacy_planner_proposal(fresh_snapshot)
        if legacy_plan_error is not None:
            await self._send_json(
                ws,
                WSErrorFrame(
                    node="planner",
                    code="LEGACY_PLAN_INCOMPATIBLE",
                    message="Saved execution plan is incompatible with this runtime",
                    retryable=False,
                ).model_dump(),
            )
            self._log.bind(thread_id=thread_id).warning("saul_legacy_plan_refused")
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
            # Reviewer authority is resolved from the authenticated session
            # and tenant membership. Never persist a client-supplied role.
            "reviewer_role": (
                LegalRole.REVIEWER.value
                if live_roles is not None and LegalRole.REVIEWER.value in live_roles
                else None
            ),
            "reviewer_id": session_context.user_id if session_context is not None else None,
        }
        if inbound.action in {"approve", "reject", "modify"} and session_context is not None:
            audit = getattr(recorder, "record_review_decision", None)
            if not callable(audit):
                await self._send_json(
                    ws,
                    WSErrorFrame(
                        node="human_review",
                        code="REVIEW_AUDIT_UNAVAILABLE",
                        message="Review decision could not be durably recorded",
                        retryable=True,
                    ).model_dump(),
                )
                return None
            audit_result = await audit(
                action=inbound.action,
                reviewer_id=session_context.user_id,
                reviewer_role=LegalRole.REVIEWER.value,
                self_reviewed=str(session_context.user_id) == str(fresh_values.get("user_id")),
                checkpoint_id=checkpoint_id,
            )
            if isinstance(audit_result, Failure):
                await self._send_json(
                    ws,
                    WSErrorFrame(
                        node="human_review",
                        code="REVIEW_AUDIT_UNAVAILABLE",
                        message="Review decision could not be durably recorded",
                        retryable=True,
                    ).model_dump(),
                )
                return None
        # T1' — project-before-resume (§5.2): retry once internally, then
        # proceed fail-open with a reconcile hint; never block the resume
        # on transcript health.
        if not _recorder_resume_alive(recorder):
            await self._send_lock_lost(ws, thread_id)
            return None
        resume_projection = await recorder.prepare_resume(plan_resume=plan_resume)
        if isinstance(resume_projection, Failure):
            self._log.bind(thread_id=thread_id).warning("saul_t1p_before_resume_project_failed")
        if not _recorder_resume_alive(recorder):
            await self._send_lock_lost(ws, thread_id)
            return None
        if not await self._persist_budget_resume(
            config=config, state_snapshot=fresh_snapshot, thread_id=thread_id
        ):
            await self._send_json(
                ws,
                WSErrorFrame(
                    node=None,
                    code="BUDGET_STATE_UNAVAILABLE",
                    message="Budget state could not be durably resumed",
                    retryable=True,
                ).model_dump(),
            )
            return None
        legacy_update = (
            _legacy_resume_update(fresh_snapshot, legacy_plan)
            if recorded_version != STATE_SCHEMA_VERSION
            else None
        )
        return Command(update=legacy_update, resume=resume_payload)

    async def _live_authorization(self) -> tuple[tuple[str, ...] | None, bool, int]:
        """Re-resolve live role state before applying a durable HITL reply."""
        session_context = getattr(self, "_context", None)
        membership_service = getattr(self, "_tenant_membership_service", None)
        if session_context is None:
            # Internal resume/replay callers can exercise this method without a
            # live WebSocket session.  There is no authenticated action to
            # authorize in that shape; production sessions always provide both
            # constructor-injected values through the router.
            return None, False, -1
        if membership_service is None:
            # A live session without the membership authority is a fail-closed
            # configuration: never turn a missing authorization dependency into
            # an approval capability.
            return (), False, -1
        result = await membership_service.get_live_authorization(
            tenant_id=session_context.tenant_id,
            user_id=session_context.user_id,
        )
        if isinstance(result, Failure):
            return (), False, -1
        legal_roles, self_review_allowed, role_revision = result.unwrap()
        return (
            tuple(role.value for role in legal_roles),
            self_review_allowed,
            role_revision,
        )

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
            ws, thread_id, interrupt_value, recorder, state_snapshot, config=config
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

            if not _recorder_resume_alive(recorder):
                await self._send_lock_lost(ws, thread_id)
                return None

            return await self._apply_resume(
                inbound, ws=ws, thread_id=thread_id, recorder=recorder, config=config
            )

    async def _send_lock_lost(self, ws: WebSocket, thread_id: str) -> None:
        await self._send_json(
            ws,
            WSErrorFrame(
                node=None,
                code="LOCK_LOST",
                message="Thread exclusion was lost; this run must stop",
                retryable=False,
            ).model_dump(),
        )
        self._log.bind(thread_id=thread_id).warning("saul_lock_lost_before_resume")

    async def _receive_json(self, ws: WebSocket) -> object:
        return await self._ws_security.receive_json(ws, self._ws_context)

    async def _send_json(self, ws: WebSocket, payload: object) -> None:
        security = getattr(self, "_ws_security", None)
        if security is not None:
            await security.send_json(ws, payload, self._ws_context)
            return
        # Narrow compatibility seam for direct unit/replay harnesses that
        # construct the service via ``__new__``; real sessions cannot reach
        # this branch because the router injects WebSocketSecurityService.
        sender = getattr(ws, "send_json", None)
        if callable(sender):
            await sender(payload)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_config(self, thread_id: str) -> RunnableConfig:
        """Run config for every round of this session's turn loop.

        The document content source is injected through ``configurable``: the
        Postgres checkpointer repopulates persisted configurable from
        ``thread_id``/``checkpoint_ns``/``checkpoint_id`` only, so this
        per-session object stays in-process while every resume round keeps
        carrying it.
        """
        return RunnableConfig(
            configurable={
                "thread_id": thread_id,
                DOCUMENT_SOURCE_CONFIG_KEY: postgres_document_content_source(self._session_factory),
            },
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
            "doc_id": start_msg.primary_doc_id or start_msg.doc_id or start_msg.allowed_doc_ids[0],
            "primary_doc_id": start_msg.primary_doc_id
            or start_msg.doc_id
            or start_msg.allowed_doc_ids[0],
            "allowed_doc_ids": list(start_msg.allowed_doc_ids),
            "tenant_id": self._context.tenant_id,
            "legal_roles": [role.value for role in self._context.legal_roles],
            "allowed_operations": list(self._context.allowed_operations),
            "capabilities": list(self._context.capabilities),
            "document_versions": [],
            "document_contracts": [],
            "self_review_allowed": self._context.self_review_allowed,
            "self_reviewed": False,
            "document_text": None,
            "ingestion_state": initial_ingestion_state(start_msg.allowed_doc_ids),
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
            "plan_rejection_feedback": None,
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
            # Kept as an empty compatibility field until the state schema's
            # server-owned authorization context lands; client permissions
            # are never copied into graph state.
            "permissions": {},
            "budget_ledger": BudgetLedger(
                limits=BudgetLimits(
                    model_calls=get_settings().SAUL_MAX_MODEL_CALLS,
                    retrieval_calls=get_settings().SAUL_MAX_RETRIEVAL_CALLS,
                    tool_calls=get_settings().SAUL_MAX_TOOL_CALLS,
                    deep_research_fanout=get_settings().SAUL_MAX_DEEP_RESEARCH_FANOUT,
                    token_usage=get_settings().SAUL_MAX_TOKEN_USAGE,
                    wall_clock_seconds=get_settings().SAUL_MAX_WALL_CLOCK_SECONDS,
                )
            ),
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


def _legacy_planner_proposal(snapshot: StateSnapshot) -> tuple[list[PlanStep] | None, str | None]:
    """Recover a proposal from an old combined planner checkpoint.

    Before the planner/approval split, the proposal existed only in the
    pending interrupt payload; the node had not returned a state update yet.
    A resumed old checkpoint therefore has an empty ``state.plan``.  The
    service must copy that exact displayed proposal into the state update sent
    with ``Command(resume=...)`` or refuse explicitly when it is malformed.
    """
    values = snapshot.values if isinstance(snapshot.values, dict) else {}
    if values.get("plan"):
        return None, None
    for interrupt in AgentSaulService._interrupts_of(snapshot):
        value = getattr(interrupt, "value", None)
        if not isinstance(value, dict):
            continue
        interrupt_type = value.get("type")
        if interrupt_type not in {
            HITLInterruptType.PLAN_APPROVAL,
            HITLInterruptType.PLAN_APPROVAL.value,
        }:
            continue
        raw_plan = value.get("plan")
        if not isinstance(raw_plan, list) or not raw_plan:
            return None, "Saved planner proposal is missing or empty"
        try:
            return [PlanStep.model_validate(step) for step in raw_plan], None
        except ValidationError:
            return None, "Saved planner proposal is incompatible with this runtime"
    return None, None


def _legacy_resume_update(
    snapshot: StateSnapshot,
    proposal: list[PlanStep] | None,
) -> dict[str, Any]:
    """Build a safe v2 channel update for a direct legacy-node resume.

    A ``Command`` resumes the pending node directly; the graph entry
    ``state_hydration`` node is not revisited.  Carry the hydrated scalar
    channels and current schema marker explicitly, while omitting reducer
    channels whose replay would duplicate accumulated values.
    """
    values = snapshot.values if isinstance(snapshot.values, dict) else {}
    update = legacy_resume_update(dict(values))
    if proposal is not None:
        update["plan"] = proposal
    return update


def _snapshot_checkpoint_id(snapshot: StateSnapshot) -> str:
    configurable = (
        snapshot.config.get("configurable") if isinstance(snapshot.config, dict) else None
    )
    checkpoint_id = configurable.get("checkpoint_id") if isinstance(configurable, dict) else None
    return checkpoint_id if isinstance(checkpoint_id, str) else ""


def _is_plan_approval_interrupt(interrupt: object) -> bool:
    value = (
        interrupt.get("value", interrupt)
        if isinstance(interrupt, Mapping)
        else getattr(interrupt, "value", None)
    )
    return isinstance(value, Mapping) and value.get("type") in {
        HITLInterruptType.PLAN_APPROVAL,
        HITLInterruptType.PLAN_APPROVAL.value,
    }


def _interrupt_type_from_snapshot(snapshot: StateSnapshot) -> HITLInterruptType | None:
    """Return the typed HITL kind without trusting client resume fields."""
    interrupts = [interrupt for task in snapshot.tasks for interrupt in task.interrupts]
    if not interrupts:
        return None
    value = getattr(interrupts[0], "value", None)
    raw = value.get("type") if isinstance(value, Mapping) else None
    try:
        return HITLInterruptType(raw)
    except (TypeError, ValueError):
        return None


def _recorder_resume_alive(recorder: TurnRecorder) -> bool:
    checker = getattr(recorder, "resume_is_allowed", None)
    return checker() if callable(checker) else True
