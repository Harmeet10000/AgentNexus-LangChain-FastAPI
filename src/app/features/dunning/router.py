"""Billing admin endpoints: dunning configuration and manual runs."""

from typing import TYPE_CHECKING, Annotated

from fastapi import APIRouter, Depends, Query, Response
from returns.result import Failure, Success

from app.config import get_settings
from app.features.auth import require_role
from app.features.auth.model import UserRole
from app.shared.result import render_result
from app.utils import APIResponse, http_response

from .dependencies import DunningServiceDep
from .dto import DunningConfigResponse

if TYPE_CHECKING:
    from .errors import DunningResult

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
    response: Response,
    limit: Annotated[int, Query(ge=1, le=500)] = 200,
) -> APIResponse[dict[str, object]]:
    due_result = await service.find_due_for_retry(limit=limit)
    if isinstance(due_result, Failure):
        failure: DunningResult[dict[str, object]] = Failure(due_result.failure())
        return render_result(failure, response, message="Dunning run complete")
    due = due_result.unwrap()
    results = []
    for subscription in due:
        updated_result = await service.execute_retry(subscription)
        if isinstance(updated_result, Failure):
            failure = Failure(updated_result.failure())
            return render_result(failure, response, message="Dunning run complete")
        updated = updated_result.unwrap()
        results.append({"subscription_id": str(updated.id), "status": updated.status})
    return render_result(
        Success({"checked": len(due), "retried": len(results), "items": results}),
        response,
        message="Dunning run complete",
    )
