"""
Agent Saul HTTP router.

Endpoints:
    POST /agent-saul/sessions            → create thread_id, return WS URL
    GET  /agent-saul/sessions/{id}/export → NDJSON transcript export

The WebSocket streaming endpoint lives in
``app.features.agent_saul.api.websocket.endpoint`` (registered onto this
router object at import time — see the bottom of this module).

The router is intentionally thin.  Every non-trivial decision lives in
AgentSaulService.  Routers handle: transport lifecycle, dependency injection,
validation of the first inbound frame, and structured error emission.
"""

from collections.abc import AsyncIterator
from json import dumps
from typing import TYPE_CHECKING, Annotated, Any
from uuid import uuid4

from fastapi import APIRouter, Depends, Path, Request, Response
from fastapi.responses import StreamingResponse
from returns.result import Failure

from app.config import get_settings
from app.features.agent_saul.api.dto import CreateSessionRequest, CreateSessionResponse
from app.features.agent_saul.api.http.dependencies import (
    DocumentContentSourceDep,
    TranscriptExporterDep,
)
from app.features.agent_saul.application.service import AgentSaulService
from app.features.agent_saul.application.transcript.export import ExportTranscript
from app.features.auth.api.http.dependencies import CurrentAuthorizationContext, CurrentClaims
from app.features.documents.api.http.dependencies import DocumentAuthorizationDep
from app.kernel.result import render_result
from app.observability.logging import logger
from app.resilience.rate_limit import get_rate_limiter
from app.utils import APIResponse

if TYPE_CHECKING:
    from app.features.agent_saul.application.transcript.export import PreparedTranscriptExport

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
# WebSocket endpoint — lives in the websocket transport package; importing it
# registers the WS route on the shared router object (single prefix/tags).
# ---------------------------------------------------------------------------


from app.features.agent_saul.api.websocket import endpoint as _saul_ws_endpoint  # noqa: E402, F401
