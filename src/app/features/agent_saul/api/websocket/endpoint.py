"""Agent Saul WebSocket streaming endpoint (WS /agent-saul/ws/{thread_id}).

The WS transport scope of the agent-saul router. The handler owns only the
connection lifecycle — accept, first-frame validation, and structured error
emission — and delegates the entire session to AgentSaulService. It registers
itself on the shared ``router`` object from the HTTP module so the feature
keeps one URL prefix and tag set.
"""

from contextlib import suppress
from typing import Annotated

from fastapi import Depends, WebSocket, WebSocketDisconnect

from app.features.agent_saul.api.dto import (
    WSErrorFrame,
    WSResumeMessage,
    WSStartMessage,
    ws_inbound_adapter,
)
from app.features.agent_saul.api.http.dependencies import (
    AgentSaulDepsWsAnnotated,
    AgentSaulWebSocketSecurityContextAnnotated,
)
from app.features.agent_saul.api.http.router import router
from app.features.agent_saul.application.runtime import SaulThreadContext
from app.features.agent_saul.application.service import AgentSaulService
from app.features.auth.adapters.persistence.mongodb.user_store import AuthAccountStore
from app.features.auth.api.http.dependencies import (
    TenantMembershipServiceDep,
    WebSocketAuthorizationContext,
    get_account_store,
)
from app.features.auth.api.websocket.security import WebSocketSecurityViolation
from app.features.documents.api.http.dependencies import DocumentAuthorizationWsDep
from app.observability.logging import logger
from app.utils import ErrorCode, ValidationException


@router.websocket("/ws/{thread_id}")
async def saul_ws_endpoint(  # noqa: PLR0917
    websocket: WebSocket,
    thread_id: str,
    deps: AgentSaulDepsWsAnnotated,
    security_context: AgentSaulWebSocketSecurityContextAnnotated,
    authorization: WebSocketAuthorizationContext,
    document_authorization: DocumentAuthorizationWsDep,
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
