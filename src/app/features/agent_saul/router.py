"""
Agent Saul router.

Endpoints:
    POST /agent-saul/sessions      → create thread_id, return WS URL
    WS   /agent-saul/ws/{thread_id} → main streaming endpoint

The router is intentionally thin.  Every non-trivial decision lives in
AgentSaulService.  Routers handle: WS lifecycle, dependency injection,
validation of the first inbound frame, and structured error emission.
"""

from collections.abc import AsyncIterator
from contextlib import suppress
from json import dumps
from typing import Annotated, Any
from uuid import uuid4

from fastapi import APIRouter, Depends, Path, Request, Response, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse
from returns.result import Failure

from app.config import get_settings
from app.features.auth.dependencies import CurrentClaims
from app.features.auth.websocket_security import WebSocketSecurityViolation
from app.kernel.result import render_result
from app.observability.logging import logger
from app.resilience.rate_limit import get_rate_limiter
from app.utils import APIResponse, ErrorCode, ValidationException

from .dependencies import (
    AgentSaulDepsAnnotated,
    AgentSaulDepsWsAnnotated,
    AgentSaulWebSocketSecurityContextAnnotated,
    get_transcript_repository,
)
from .dto import (
    CreateSessionRequest,
    CreateSessionResponse,
    WSErrorFrame,
    WSResumeMessage,
    WSStartMessage,
    ws_inbound_adapter,
)
from .service import AgentSaulService
from .transcript.repository import ChatTranscriptRepository

_EXPORT_PAGE_ROWS = 500  # §8 export: bounded read size; pages stream until short page

router = APIRouter(
    prefix="/agent-saul",
    tags=["agent-saul"],
)

limit_agent_session_creation = get_rate_limiter(10, 10, 60, scope="agent-sessions")


# ---------------------------------------------------------------------------
# HTTP: session pre-flight
# ---------------------------------------------------------------------------


def get_websocket_url(request: Request, path: str) -> str:
    """Task 3.5: Proxy-aware absolute WebSocket URL.

    Honors X-Forwarded-Proto / X-Forwarded-Host only when
    FASTAPI_GUARD_TRUSTED_PROXIES is configured — the same trust decision the
    HTTP middleware applies to forwarded headers.
    """
    settings = get_settings()
    trusted = bool(settings.FASTAPI_GUARD_TRUSTED_PROXIES)
    forwarded_proto = request.headers.get("x-forwarded-proto") if trusted else None
    forwarded_host = request.headers.get("x-forwarded-host") if trusted else None

    scheme = (
        forwarded_proto.split(",")[0].strip()
        if forwarded_proto
        else ("wss" if request.url.scheme == "https" else "ws")
    )
    host = (
        forwarded_host.split(",")[0].strip()
        if forwarded_host
        else request.headers.get("host") or request.url.netloc
    )
    return f"{scheme}://{host}{path}"


@router.post(
    "/sessions",
    summary="Create a new Agent Saul session and receive a thread_id + WS URL",
    dependencies=[Depends(limit_agent_session_creation)],
)
async def create_session(
    body: CreateSessionRequest,
    _deps: AgentSaulDepsAnnotated,
    claims: CurrentClaims,
    request: Request,
    response: Response,
) -> APIResponse[CreateSessionResponse]:
    thread_id = str(uuid4())
    ws_path = f"{get_settings().API_PREFIX}/agent-saul/ws/{thread_id}"
    ws_url = get_websocket_url(request, ws_path)

    log = logger.bind(user_id=claims.sub, thread_id=thread_id, doc_id=body.doc_id)
    log.info("saul_session_created")

    result = AgentSaulService.create_session(websocket_url=ws_url, thread_id=thread_id)
    return render_result(
        result,
        response,
        message="Session created",
    )


# ---------------------------------------------------------------------------
# HTTP: chat transcript export (design §3.3/D3)
# ---------------------------------------------------------------------------


@router.get("/sessions/{session_id}/export", response_model=None)
async def export_session_transcript(
    session_id: Annotated[str, Path(min_length=1, max_length=100)],
    repo: Annotated[ChatTranscriptRepository, Depends(get_transcript_repository)],
    claims: CurrentClaims,
    response: Response,
) -> StreamingResponse | APIResponse[Any]:
    """Stream one session's transcript in display order (turn_seq, ordinal).

    NDJSON, one row per line, so a client can render large transcripts
    incrementally. Redacted rows export AS redacted: ``content`` is null and
    ``redacted`` is true — erasure is represented, never reconstructed (§3.2).
    Reads are schema-version gated in the repository (§9 fail closed).
    """
    user_id = claims.sub
    meta_result = await repo.get_session_meta(session_id=session_id, user_id=user_id)
    if isinstance(meta_result, Failure):
        # Every transcript error variant is a FeatureError; render_result
        # derives the transport status from error.kind — a future variant
        # renders instead of raising (§10, router-renders-result).
        return render_result(meta_result, response)

    _ = meta_result.unwrap()  # fail-closed gate passed: owner + schema verified

    async def rows() -> AsyncIterator[str]:
        # Keyset-paged bounded reads streamed page by page: memory stays flat
        # and appends mid-export can never repeat or skip a page (D3 order is
        # the (turn_seq, ordinal) the reader itself walks).
        cursor_seq: int | None = None
        cursor_ord: int | None = None
        cursor_id: int | None = None
        while True:
            result = await repo.read_transcript(
                session_id=session_id,
                user_id=user_id,
                limit=_EXPORT_PAGE_ROWS,
                after_turn_seq=cursor_seq,
                after_ordinal=cursor_ord,
                after_id=cursor_id,
            )
            if isinstance(result, Failure):
                # Surface the typed error as a terminal line, never a silent
                # truncation; driver detail is already stripped (§10).
                error_payload = result.failure()
                yield (
                    dumps(
                        {
                            "error": str(error_payload.code),
                            "message": error_payload.message,
                        }
                    )
                    + "\n"
                )
                return
            page_rows = list(result.unwrap())
            if not page_rows:
                return
            for item in page_rows:
                yield dumps(item.model_dump(mode="json"), default=str) + "\n"
            if len(page_rows) < _EXPORT_PAGE_ROWS:
                return
            last = page_rows[-1]
            cursor_seq, cursor_ord, cursor_id = last.turn_seq, last.ordinal, last.row_id

    return StreamingResponse(
        rows(),
        media_type="application/x-ndjson",
        headers={"Content-Disposition": f'attachment; filename="transcript-{session_id}.ndjson"'},
    )


# ---------------------------------------------------------------------------
# WebSocket: main streaming endpoint
# ---------------------------------------------------------------------------


@router.websocket("/ws/{thread_id}")
async def saul_ws_endpoint(
    websocket: WebSocket,
    thread_id: str,
    deps: AgentSaulDepsWsAnnotated,
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

    async def _handle_ws_session() -> None:
        security_service = websocket.app.state.websocket_security
        await websocket.accept()
        await security_service.register_connection(security_context, websocket)
        raw = await security_service.receive_json(websocket, security_context)
        first_msg = ws_inbound_adapter.validate_python(raw)

        # First frame protocol: a ``start`` starts a turn (new or existing
        # thread); a ``resume`` is the reconnect-resume claim (§6) for a
        # thread paused in a HITL interrupt. Anything else is refused.
        start_msg: WSStartMessage | None = None
        first_resume: WSResumeMessage | None = None
        if isinstance(first_msg, WSStartMessage):
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
            start_msg = first_msg
        elif isinstance(first_msg, WSResumeMessage):
            if first_msg.thread_id != thread_id:
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
            first_resume = first_msg
        else:
            await security_service.send_json(
                websocket,
                WSErrorFrame(
                    node=None,
                    code="INVALID_FIRST_FRAME",
                    message=f"Expected 'start' or 'resume' message, got '{first_msg.type}'",
                    retryable=False,
                ).model_dump(),
                security_context,
            )
            await websocket.close(code=4000)
            return

        log.info(
            "saul_ws_session_started",
            doc_id=start_msg.doc_id if start_msg is not None else None,
            reconnect_resume=first_resume is not None,
        )
        service = AgentSaulService(
            graph=deps.graph,
            redis=deps.redis,
            correlation_id=correlation_id,
            ws_security=security_service,
            ws_context=security_context,
            session_factory=deps.session_factory,
        )
        await service.run_session(
            ws=websocket,
            start_msg=start_msg,
            first_resume=first_resume,
            thread_id=thread_id,
            user_id=security_context.user_id,
        )

    try:
        await _handle_ws_session()
    except WebSocketSecurityViolation as exc:
        security_service = websocket.app.state.websocket_security
        await security_service.close_with_violation(websocket, exc)

    except WebSocketDisconnect as exc:
        log.info("saul_ws_disconnected", code=exc.code)

    except ValidationException as exc:
        log.warning("saul_ws_validation_error", error=str(exc))
        with suppress(Exception):
            await websocket.app.state.websocket_security.send_json(
                websocket,
                WSErrorFrame(
                    node=None,
                    code=ErrorCode.VALIDATION_ERROR,
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
