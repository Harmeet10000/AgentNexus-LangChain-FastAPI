"""Billing admin endpoints: dunning configuration and manual runs."""

from typing import TYPE_CHECKING, Annotated

from fastapi import APIRouter, Depends, Query, Response
from returns.result import Failure, Success

from app.config import get_settings
from app.features.auth.api.http.dependencies import require_role
from app.features.auth.domain.user import UserRole
from app.kernel.result import render_result
from app.utils import APIResponse, http_response

from .dependencies import DunningServiceDep
from .dto import DunningConfigResponse

if TYPE_CHECKING:
    from .exceptions import DunningResult

router = APIRouter(prefix="/admin/billing", tags=["billing-admin"])


def _persisted_payment_link_url(subscription: object) -> str | None:
    """Read the retry URL from the durable subscription attempt record."""
    metadata = getattr(subscription, "metadata_", None)
    if not isinstance(metadata, dict):
        return None
    raw_attempts = metadata.get("dunning_attempts")
    if not isinstance(raw_attempts, list) or not raw_attempts:
        return None
    latest = raw_attempts[-1]
    if not isinstance(latest, dict):
        return None
    short_url = latest.get("short_url")
    return short_url if isinstance(short_url, str) and short_url else None


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
        result_item: dict[str, object] = {
            "subscription_id": str(updated.id),
            "status": updated.status,
        }
        # ``short_url`` is persisted inside the JSONB dunning attempt record
        # by DunningService.  The attribute fallback preserves compatibility
        # with older in-memory callers while production responses use data
        # that survives a process restart and is queryable from the database.
        payment_link_url = _persisted_payment_link_url(updated)
        if payment_link_url is None:
            payment_link_url = getattr(updated, "payment_link_url", None)
        if isinstance(payment_link_url, str) and payment_link_url:
            result_item["payment_link_url"] = payment_link_url
        results.append(result_item)
    return render_result(
        Success({"checked": len(due), "retried": len(results), "items": results}),
        response,
        message="Dunning run complete",
    )
