"""Billing admin endpoints."""

from typing import Annotated

from fastapi import APIRouter, Depends, Query

from app.config import get_settings
from app.features.auth import require_role
from app.features.auth.model import UserRole
from app.features.billing.dto import DunningConfigResponse
from app.features.billing.response import bill_response
from app.utils import APIResponse

from ..dependencies import DunningServiceDep, WebhookServiceDep

router = APIRouter(prefix="/admin/billing", tags=["billing-admin"])


@router.get(
    "/dunning",
    dependencies=[Depends(require_role(UserRole.ADMIN))],
)
async def get_dunning_config() -> APIResponse[DunningConfigResponse]:
    settings = get_settings()
    config = DunningConfigResponse(
        retry_delay_days=list(settings.BILLING_DUNNING_RETRY_DAYS),
        max_retries=settings.BILLING_MAX_RETRIES,
    )
    return bill_response("Dunning configuration", config)


@router.post(
    "/dunning/run",
    dependencies=[Depends(require_role(UserRole.ADMIN))],
)
async def run_dunning_now(
    service: DunningServiceDep,
    limit: Annotated[int, Query(ge=1, le=500)] = 200,
) -> APIResponse[dict[str, object]]:
    due = await service.find_due_for_retry(limit=limit)
    results = []
    for subscription in due:
        updated = await service.execute_retry(subscription)
        results.append({"subscription_id": str(updated.id), "status": updated.status})
    return bill_response(
        "Dunning run complete", {"checked": len(due), "retried": len(results), "items": results}
    )


@router.post(
    "/webhooks/{event_id}/replay",
    dependencies=[Depends(require_role(UserRole.ADMIN))],
)
async def replay_webhook(
    event_id: str, service: WebhookServiceDep
) -> APIResponse[dict[str, object]]:
    event = await service.replay(event_id)
    return bill_response(
        "Webhook event replayed",
        {
            "event_id": str(event.id),
            "event_type": event.event_type,
            "status": event.status,
            "retry_count": event.retry_count,
        },
    )
