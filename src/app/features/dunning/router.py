"""Billing admin endpoints: dunning configuration and manual runs."""

from typing import Annotated

from fastapi import APIRouter, Depends, Query

from app.config import get_settings
from app.features.auth import require_role
from app.features.auth.model import UserRole
from app.utils import APIResponse, http_response

from .dependencies import DunningServiceDep
from .dto import DunningConfigResponse

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
    return http_response("Dunning configuration", config)


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
    return http_response(
        "Dunning run complete", {"checked": len(due), "retried": len(results), "items": results}
    )
