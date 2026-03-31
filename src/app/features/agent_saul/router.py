"""
Agent Saul router.

Endpoints:
    POST /agent-saul/sessions      → create thread_id, return WS URL
    WS   /agent-saul/ws/{thread_id} → main streaming endpoint

The router is intentionally thin.  Every non-trivial decision lives in
AgentSaulService.  Routers handle: WS lifecycle, dependency injection,
validation of the first inbound frame, and structured error emission.
"""

from contextlib import suppress
from uuid import uuid4

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse

from app.features.agent_saul.dependencies import (
    AgentSaulDepsAnnotated,
    AgentSaulWebSocketSecurityContextAnnotated,
)
from app.features.agent_saul.dto import (
    CreateSessionRequest,
    CreateSessionResponse,
    WSErrorFrame,
    WSStartMessage,
    ws_inbound_adapter,
)
from app.features.agent_saul.service import AgentSaulService
from app.features.auth.dependencies import CurrentClaims
from app.features.auth.websocket_security import WebSocketSecurityViolation
from app.shared.response_type import APIResponse
from app.utils import http_response, logger
from app.utils.exceptions import ValidationException

router = APIRouter(
    prefix="/agent-saul",
    tags=["agent-saul"],
)


# ---------------------------------------------------------------------------
# HTTP: session pre-flight
# ---------------------------------------------------------------------------


@router.post(
    "/sessions",
    response_model=APIResponse[CreateSessionResponse],
    summary="Create a new Agent Saul session and receive a thread_id + WS URL",
)
async def create_session(
    body: CreateSessionRequest,
    _deps: AgentSaulDepsAnnotated,
    claims: CurrentClaims,
) -> JSONResponse:
    thread_id = str(uuid4())
    ws_url = f"ws://{{host}}/api/v1/agent-saul/ws/{thread_id}"

    log = logger.bind(user_id=claims.sub, thread_id=thread_id, doc_id=body.doc_id)
    log.info("saul_session_created")

    return http_response(
        message="Session created",
        data=CreateSessionResponse(thread_id=thread_id, ws_url=ws_url),
    )


# ---------------------------------------------------------------------------
# WebSocket: main streaming endpoint
# ---------------------------------------------------------------------------


@router.websocket("/ws/{thread_id}")
async def saul_ws_endpoint(
    websocket: WebSocket,
    thread_id: str,
    deps: AgentSaulDepsAnnotated,
    security_context: AgentSaulWebSocketSecurityContextAnnotated,
) -> None:
    """
    WebSocket session lifecycle:

      1. Accept connection.
      2. Receive first frame — MUST be WSStartMessage.
      3. Delegate entirely to AgentSaulService.run_session().
      4. Service owns the inner loop: streaming → HITL → resume → streaming.
      5. On WSDoneFrame or exception: close with appropriate code.

    The router never calls graph methods directly.
    """
    correlation_id = str(uuid4())
    log = logger.bind(
        user_id=security_context.user_id,
        thread_id=thread_id,
        correlation_id=correlation_id,
    )

    try:
        security_service = websocket.app.state.websocket_security
        await websocket.accept()
        await security_service.register_connection(security_context)

        # ---- First frame validation ----------------------------------------
        raw = await security_service.receive_json(websocket, security_context)
        first_msg = ws_inbound_adapter.validate_python(raw)

        if not isinstance(first_msg, WSStartMessage):
            await security_service.send_json(
                websocket,
                WSErrorFrame(
                    node=None,
                    code="INVALID_FIRST_FRAME",
                    message=f"Expected 'start' message, got '{first_msg.type}'",
                    retryable=False,
                ).model_dump(),
                security_context,
            )
            await websocket.close(code=4000)
            return

        # Thread ID from URL takes precedence; client may also embed it in
        # WSStartMessage.thread_id for validation, but URL param is canonical.
        if first_msg.thread_id and first_msg.thread_id != thread_id:
            await security_service.send_json(
                websocket,
                WSErrorFrame(
                    node=None,
                    code="THREAD_ID_MISMATCH",
                    message="thread_id in URL and message body do not match",
                    retryable=False,
                ).model_dump(),
                security_context,
            )
            await websocket.close(code=4001)
            return

        log.info("saul_ws_session_started", doc_id=first_msg.doc_id)

        # ---- Delegate to service -------------------------------------------
        service = AgentSaulService(
            graph=deps.graph,
            redis=deps.redis,
            correlation_id=correlation_id,
            ws_security=security_service,
            ws_context=security_context,
        )
        await service.run_session(
            ws=websocket,
            start_msg=first_msg,
            thread_id=thread_id,
            user_id=security_context.user_id,
        )

    except WebSocketSecurityViolation as exc:
        security_service = websocket.app.state.websocket_security
        await security_service.close_with_violation(websocket, security_context, exc)

    except WebSocketDisconnect as exc:
        log.info("saul_ws_disconnected", code=exc.code)

    except ValidationException as exc:
        log.warning("saul_ws_validation_error", error=str(exc))
        with suppress(Exception):
            await websocket.app.state.websocket_security.send_json(
                websocket,
                WSErrorFrame(
                    node=None,
                    code="VALIDATION_ERROR",
                    message=str(exc),
                    retryable=False,
                ).model_dump(),
                security_context,
            )
        await websocket.close(code=4002)

    except Exception as exc:
        log.exception("saul_ws_unhandled_error", error=str(exc))
        with suppress(Exception):
            await websocket.app.state.websocket_security.send_json(
                websocket,
                WSErrorFrame(
                    node=None,
                    code="INTERNAL_ERROR",
                    message="An unexpected error occurred",
                    retryable=True,
                ).model_dump(),
                security_context,
            )
        await websocket.close(code=1011)
    finally:
        with suppress(Exception):
            await websocket.app.state.websocket_security.unregister_connection(security_context)
