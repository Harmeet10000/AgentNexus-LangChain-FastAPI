"""Razorpay webhook endpoint (public, HMAC-verified) + admin replay."""

from typing import Annotated

from fastapi import APIRouter, Depends, Header, Request

from app.features.auth import require_role
from app.features.auth.model import UserRole
from app.features.webhooks.exceptions import WebhookVerificationException
from app.utils import APIResponse, http_response, logger

from .dependencies import WebhookServiceDep
from .dto import WebhookPayload

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
    signature: Annotated[str | None, Header(alias="X-Razorpay-Signature")] = None,
) -> APIResponse[dict[str, object]]:
    raw_body = (await request.body()).decode("utf-8")
    if signature is None:
        msg = "Missing X-Razorpay-Signature header"
        raise WebhookVerificationException(msg)
    service.verify_signature(raw_body=raw_body, signature=signature)

    payload = WebhookPayload.model_validate_json(raw_body)
    event_id = _extract_event_id(payload.payload)
    if not event_id:
        logger.bind(operation="webhook", event=payload.event).warning(
            "Webhook payload missing event id"
        )
    handled = await service.process(
        event_id=event_id or payload.event,
        event_type=payload.event,
        payload=payload.payload,
    )
    return http_response("Webhook processed", {"accepted": handled})


@router.post(
    "/admin/billing/webhooks/{event_id}/replay",
    dependencies=[Depends(require_role(UserRole.ADMIN))],
)
async def replay_webhook(
    event_id: str, service: WebhookServiceDep
) -> APIResponse[dict[str, object]]:
    event = await service.replay(event_id)
    return http_response(
        "Webhook event replayed",
        {
            "event_id": str(event.id),
            "event_type": event.event_type,
            "status": event.status,
            "retry_count": event.retry_count,
        },
    )
