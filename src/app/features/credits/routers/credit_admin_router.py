"""Admin/Portal credit endpoints."""

from typing import Annotated

from fastapi import APIRouter, Depends, Path, Query, Response, status

from app.features.auth import CurrentClaims, require_role
from app.features.auth.model import UserRole
from app.shared.result import render_result
from app.utils import APIResponse, ForbiddenException, ValidationException

from ..dependencies import CreditServiceDep
from ..dto import (
    CreditBalanceResponse,
    CreditGrantDTO,
    CreditGrantResponse,
    CreditHistoryResponse,
)

router = APIRouter(prefix="/credits", tags=["credits"])


@router.post(
    "",
    status_code=status.HTTP_201_CREATED,
    dependencies=[Depends(require_role(UserRole.ADMIN))],
)
async def grant_credit(
    dto: CreditGrantDTO,
    service: CreditServiceDep,
    claims: CurrentClaims,  # noqa: ARG001 — FastAPI DI dependency
    response: Response,
) -> APIResponse[CreditGrantResponse]:
    """Grant credit to a user (Requirement 49). Admin only."""
    target_user_id = dto.metadata_.get("target_user_id")
    if not target_user_id:
        message = "target_user_id is required"
        raise ValidationException(message)
    result = await service.grant_credit(user_id=target_user_id, dto=dto)
    return render_result(
        result, response, message="Credit granted", success_status=status.HTTP_201_CREATED
    )


@router.get("/balance/{user_id}")
async def get_credit_balance(
    user_id: Annotated[str, Path(min_length=1)],
    service: CreditServiceDep,
    claims: CurrentClaims,
    response: Response,
) -> APIResponse[CreditBalanceResponse]:
    """Get user's total available credit balance (Requirement 52.1)."""
    msg = "Cannot view other users' credit balances"
    if claims.sub != user_id and claims.role != UserRole.ADMIN.value:
        raise ForbiddenException(msg)
    result = await service.get_credit_balance(user_id)
    return render_result(result, response, message="Credit balance")


@router.get("/history/{user_id}")
async def get_credit_history(
    user_id: Annotated[str, Path(min_length=1)],
    service: CreditServiceDep,
    claims: CurrentClaims,
    response: Response,
    *,
    limit: Annotated[int, Query(ge=1, le=100)] = 50,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> APIResponse[CreditHistoryResponse]:
    """Get user's credit and consumption history (Requirement 52.2)."""
    msg = "Cannot view other users' credit history"
    if claims.sub != user_id and claims.role != UserRole.ADMIN.value:
        raise ForbiddenException(msg)
    result = await service.get_credit_history(user_id, limit=limit, offset=offset)
    return render_result(result, response, message="Credit history")
