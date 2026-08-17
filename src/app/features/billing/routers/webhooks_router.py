"""Razorpay webhook endpoint (public, HMAC-verified)."""

from typing import Annotated

from fastapi import APIRouter, Header, Request

from app.features.billing.dto import WebhookPayload
from app.features.billing.exceptions import WebhookVerificationException
from app.features.billing.response import bill_response
from app.utils import APIResponse, logger

from ..dependencies import WebhookServiceDep

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
    return bill_response("Webhook processed", {"accepted": handled})
