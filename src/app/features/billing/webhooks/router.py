"""Razorpay webhook endpoint (public, HMAC-verified) + admin replay."""

from typing import Annotated

from fastapi import APIRouter, Depends, Header, Request, Response
from pydantic import ValidationError
from returns.result import Failure, Success

from app.features.auth import require_role
from app.features.auth.model import UserRole
from app.shared.result import render_result
from app.utils import APIResponse, logger

from .dependencies import WebhookServiceDep
from .dto import WebhookPayload
from .errors import WebhookValidationError, WebhookVerificationError

router = APIRouter(tags=["billing-webhooks"])


def _extract_event_id(payload: dict[str, object]) -> str:
    event = payload.get("event")
    if isinstance(event, dict):
        event_id = event.get("id")
        if isinstance(event_id, str):
            return event_id
    return ""


@router.post("/webhooks/razorpay")
async def razorpay_webhook(
    request: Request,
    service: WebhookServiceDep,
    response: Response,
    signature: Annotated[str | None, Header(alias="X-Razorpay-Signature")] = None,
    event_id_header: Annotated[str | None, Header(alias="X-Razorpay-Event-Id")] = None,
) -> APIResponse[dict[str, object]]:
    raw_body = (await request.body()).decode("utf-8")
    if signature is None:
        return render_result(
            Failure(WebhookVerificationError(message="Missing X-Razorpay-Signature header")),
            response,
            message="Webhook processed",
        )
    verification = service.verify_signature(raw_body=raw_body, signature=signature)
    if isinstance(verification, Failure):
        return render_result(Failure(verification.failure()), response, message="Webhook processed")

    try:
        payload = WebhookPayload.model_validate_json(raw_body)
    except ValidationError as exc:
        return render_result(
            Failure(
                WebhookValidationError(
                    message="Invalid webhook payload", details={"error": str(exc)}
                )
            ),
            response,
            message="Webhook processed",
        )
    # Razorpay delivers the unique event identifier in a header; the body
    # carries only the event type, which must never key idempotency.
    event_id = event_id_header or _extract_event_id(payload.payload)
    if not event_id:
        logger.bind(operation="webhook", event=payload.event).warning(
            "Webhook payload missing event id"
        )
    result = await service.process(
        event_id=event_id or payload.event,
        event_type=payload.event,
        payload=payload.payload,
    )
    if isinstance(result, Failure):
        return render_result(Failure(result.failure()), response, message="Webhook processed")
    return render_result(
        Success({"accepted": result.unwrap()}), response, message="Webhook processed"
    )


@router.post(
    "/admin/billing/webhooks/{event_id}/replay",
    dependencies=[Depends(require_role(UserRole.ADMIN))],
)
async def replay_webhook(
    event_id: str, service: WebhookServiceDep, response: Response
) -> APIResponse[dict[str, object]]:
    result = await service.replay(event_id)
    if isinstance(result, Failure):
        return render_result(Failure(result.failure()), response, message="Webhook event replayed")
    event = result.unwrap()
    return render_result(
        Success(
            {
                "event_id": str(event.id),
                "event_type": event.event_type,
                "status": event.status,
                "retry_count": event.retry_count,
            }
        ),
        response,
        message="Webhook event replayed",
    )
