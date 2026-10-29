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
from typing import TYPE_CHECKING, Annotated, Any
from uuid import uuid4

from fastapi import APIRouter, Depends, Path, Request, Response, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse
from returns.result import Failure

from app.config import get_settings
from app.features.auth.adapters.persistence.mongodb.user_store import AuthAccountStore
from app.features.auth.api.http.dependencies import (
    CurrentAuthorizationContext,
    CurrentClaims,
    TenantMembershipServiceDep,
    WebSocketAuthorizationContext,
    get_account_store,
)
from app.features.auth.api.websocket.security import WebSocketSecurityViolation
from app.features.documents.dependencies import DocumentAuthorizationDep
from app.kernel.result import render_result
from app.observability.logging import logger
from app.resilience.rate_limit import get_rate_limiter
from app.utils import APIResponse, ErrorCode, ValidationException

from .dependencies import (
    AgentSaulDepsWsAnnotated,
    AgentSaulWebSocketSecurityContextAnnotated,
    DocumentContentSourceDep,
    TranscriptExporterDep,
)
from .dto import (
    CreateSessionRequest,
    CreateSessionResponse,
    WSErrorFrame,
    WSResumeMessage,
    WSStartMessage,
    ws_inbound_adapter,
)
from .runtime import SaulThreadContext
from .service import AgentSaulService
from .transcript.export import ExportTranscript

if TYPE_CHECKING:
    from .transcript.export import PreparedTranscriptExport

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
async def create_session(  # noqa: PLR0917 — FastAPI dependency injection boundary
    body: CreateSessionRequest,
    document_source: DocumentContentSourceDep,
    authorization: CurrentAuthorizationContext,
    document_authorization: DocumentAuthorizationDep,
    request: Request,
    response: Response,
) -> APIResponse[CreateSessionResponse]:
    thread_id = str(uuid4())
    ws_path = f"{get_settings().API_PREFIX}/agent-saul/ws/{thread_id}"
    ws_url = get_websocket_url(request, ws_path)

    log = logger.bind(
        user_id=authorization.user_id,
        tenant_id=authorization.tenant_id,
        thread_id=thread_id,
        doc_id=body.primary_doc_id,
    )

    # ``CurrentAuthorizationContext`` intentionally carries no document
    # allowlist: auth owns identity/roles while the documents repository owns
    # the tenant corpus capability.  Resolve that capability here before
    # narrowing the client selection.  Without this dependency every normal
    # create-session request intersects against the context's empty default
    # and is reduced to an empty scope before preflight.
    scoped_doc_ids = document_authorization.narrow_document_ids(body.allowed_doc_ids)
    primary_doc_id = body.primary_doc_id or body.doc_id
    if primary_doc_id not in scoped_doc_ids:
        primary_doc_id = scoped_doc_ids[0] if scoped_doc_ids else ""

    result = await AgentSaulService.create_session_for_document(
        websocket_url=ws_url,
        thread_id=thread_id,
        doc_id=primary_doc_id,
        allowed_doc_ids=list(scoped_doc_ids),
        user_id=authorization.user_id,
        tenant_id=authorization.tenant_id,
        document_source=document_source,
    )
    if isinstance(result, Failure):
        log.warning(
            "saul_session_creation_document_rejected",
            error_code=str(result.failure().code),
        )
    else:
        log.info("saul_session_created")
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
    exporter: TranscriptExporterDep,
    claims: CurrentClaims,
    response: Response,
) -> StreamingResponse | APIResponse[Any]:
    """Stream one session's transcript in display order (turn_seq, ordinal).

    NDJSON, one row per line, so a client can render large transcripts
    incrementally. Redacted rows export AS redacted: ``content`` is null and
    ``redacted`` is true — erasure is represented, never reconstructed (§3.2).
    Ownership and schema gating happen in preflight BEFORE streaming begins.
    Pagination and cursor management are owned by PreparedTranscriptExport.
    """
    # Preflight: ownership + schema gate. Fails with proper HTTP status.
    prepare_result = await exporter.prepare(
        ExportTranscript(session_id=session_id, requesting_user_id=claims.sub)
    )
    if isinstance(prepare_result, Failure):
        return render_result(prepare_result, response, message="Transcript export failed")

    prepared: PreparedTranscriptExport = prepare_result.unwrap()

    # Stream typed events, encode as NDJSON in the transport adapter
    async def rows() -> AsyncIterator[str]:
        async for event in prepared.events():
            # Serialize with snake_case (preserves existing wire format)
            yield dumps(event.model_dump(mode="json", by_alias=False)) + "\n"

    return StreamingResponse(
        rows(),
        media_type="application/x-ndjson",
        headers={"Content-Disposition": f'attachment; filename="transcript-{session_id}.ndjson"'},
    )


# ---------------------------------------------------------------------------
# WebSocket: main streaming endpoint
# ---------------------------------------------------------------------------


@router.websocket("/ws/{thread_id}")
async def saul_ws_endpoint(  # noqa: PLR0917
    websocket: WebSocket,
    thread_id: str,
    deps: AgentSaulDepsWsAnnotated,
    security_context: AgentSaulWebSocketSecurityContextAnnotated,
    authorization: WebSocketAuthorizationContext,
    document_authorization: DocumentAuthorizationDep,
    account_store: Annotated[AuthAccountStore, Depends(get_account_store)],
    tenant_membership_service: TenantMembershipServiceDep,
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
    # ONE thread-context construction: identity and correlation are built
    # here (runtime.SaulThreadContext) and consumed by logs and the service.
    context = SaulThreadContext(
        user_id=authorization.user_id,
        thread_id=thread_id,
        tenant_id=authorization.tenant_id,
        legal_roles=authorization.legal_roles,
        self_review_allowed=authorization.self_review_allowed,
        role_revision=authorization.role_revision,
        allowed_doc_ids=document_authorization.allowed_doc_ids,
        allowed_operations=authorization.allowed_operations,
        capabilities=authorization.capabilities,
    )
    # The security dependency intentionally only authenticates the token. Bind
    # the live tenant before registering presence/audit events.
    security_context = security_context.model_copy(update={"tenant_id": authorization.tenant_id})
    log = logger.bind(
        user_id=context.user_id,
        thread_id=context.thread_id,
        correlation_id=context.correlation_id,
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
            context=context,
            ws_security=security_service,
            ws_context=security_context,
            session_factory=deps.session_factory,
            account_store=account_store,
            tenant_membership_service=tenant_membership_service,
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
