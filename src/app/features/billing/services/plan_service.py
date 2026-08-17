"""Billing plan management service."""

from __future__ import annotations

from typing import TYPE_CHECKING

from returns.result import Failure
from sqlalchemy import select

from app.config import get_settings
from app.features.billing.clients.razorpay_client import RazorpayClient
from app.features.billing.exceptions import InvalidStateTransitionException
from app.features.billing.models import AuditAction, AuditLog, Plan
from app.features.billing.response import failure_envelope
from app.shared.result import InfrastructureAppError
from app.utils import ValidationException, logger

from ..dto import PlanResponse

if TYPE_CHECKING:
    from app.features.billing.repositories import BillingRepositories
    from app.features.billing.response import ServiceResult

    from ..dto import PlanCreateDTO, PlanUpdateDTO


def _plan_to_response(plan: Plan) -> PlanResponse:
    return PlanResponse(
        id=str(plan.id),
        parent_plan_id=str(plan.parent_plan_id) if plan.parent_plan_id else None,
        razorpay_plan_id=plan.razorpay_plan_id,
        name=plan.name,
        description=plan.description,
        amount=plan.amount,
        currency=plan.currency,
        interval=plan.interval,
        interval_count=plan.interval_count,
        trial_period_days=plan.trial_period_days,
        tax_rate=plan.tax_rate,
        refund_policy=plan.refund_policy,
        is_active=plan.is_active,
        features=plan.features,
        metadata=plan.metadata_,
        created_at=plan.created_at,
        updated_at=plan.updated_at,
    )


class PlanService:
    """CRUD for billing plans, mirrored to Razorpay when configured."""

    def __init__(self, repos: BillingRepositories, razorpay: RazorpayClient | None = None) -> None:
        self.repos = repos
        self.razorpay = razorpay or RazorpayClient()

    async def list_plans(
        self, *, include_inactive: bool = False, limit: int = 50, offset: int = 0
    ) -> ServiceResult[list[PlanResponse]]:
        if include_inactive:
            try:
                statement = (
                    select(Plan).order_by(Plan.created_at.desc()).limit(limit).offset(offset)
                )
                result = await self.repos.session.execute(statement)
                return [_plan_to_response(p) for p in result.scalars().all()]
            except Exception as exc:  # noqa: BLE001 — bounded helper
                logger.bind(operation="list_plans").warning("list_plans failed", error=str(exc))

                return failure_envelope(
                    InfrastructureAppError(
                        code="DB_ERROR",
                        message="Database error while listing plans",
                        details={"error": str(exc)},
                        source="plan_service",
                    ),
                    operation="list_plans",
                )

        result = await self.repos.plans.list_active()
        if isinstance(result, Failure):
            return failure_envelope(result.failure(), operation="list_plans")
        return [_plan_to_response(p) for p in result.unwrap()]

    async def get_plan(self, plan_id: str) -> ServiceResult[PlanResponse]:
        result = await self.repos.plans.find_by_id(plan_id)
        if isinstance(result, Failure):
            return failure_envelope(result.failure(), operation="get_plan")
        plan = result.unwrap()
        if plan is None:
            msg = "Plan not found"
            raise ValidationException(msg)
        return _plan_to_response(plan)

    async def create_plan(self, dto: PlanCreateDTO, *, user_id: str) -> ServiceResult[PlanResponse]:
        existing = await self.repos.plans.find_by_name(dto.name)
        if isinstance(existing, Failure):
            return failure_envelope(existing.failure(), operation="create_plan")
        if existing.unwrap() is not None:
            msg = f"A plan named '{dto.name}' already exists"
            raise ValidationException(msg)

        plan = Plan(
            name=dto.name,
            description=dto.description,
            amount=dto.amount,
            currency=dto.currency,
            interval=dto.interval.value,
            interval_count=dto.interval_count,
            trial_period_days=dto.trial_period_days,
            tax_rate=dto.tax_rate,
            refund_policy=dto.refund_policy,
            features=dto.features,
            metadata_=dto.metadata,
        )
        result = await self.repos.plans.create(plan)
        if isinstance(result, Failure):
            return failure_envelope(result.failure(), operation="create_plan")
        created = result.unwrap()

        if self._sync_razorpay_enabled():
            try:
                razorpay_plan = await self.razorpay.create_plan(
                    name=created.name,
                    amount=created.amount,
                    interval=created.interval,
                    currency=created.currency,
                    period=created.interval_count,
                )
                await self.repos.plans.update(
                    created, values={"razorpay_plan_id": razorpay_plan.get("id")}
                )
                created.razorpay_plan_id = razorpay_plan.get("id")
            except Exception as exc:  # noqa: BLE001 — Razorpay is optional in dev
                logger.bind(operation="create_plan").warning(
                    "Razorpay plan sync skipped", error=str(exc)
                )

        await self.repos.audit.create(
            AuditLog(
                entity_type="plan",
                entity_id=str(created.id),
                action=AuditAction.SUBSCRIPTION_CREATED.value,
                user_id=user_id,
                changes={"name": created.name, "amount": created.amount},
            )
        )
        return _plan_to_response(created)

    async def update_plan(  # noqa: PLR0912
        self, plan_id: str, dto: PlanUpdateDTO, *, user_id: str
    ) -> ServiceResult[PlanResponse]:
        """Create a new plan version; keep the current one for existing subscribers.

        Requirement 24: updates never mutate the original record. A new Plan
        row is created (``parent_plan_id`` → current plan) and the previous
        version is deactivated so new subscriptions bind to the latest pricing.
        """
        result = await self.repos.plans.find_by_id(plan_id)
        if isinstance(result, Failure):
            return failure_envelope(result.failure(), operation="update_plan")
        plan = result.unwrap()
        if plan is None:
            msg = "Plan not found"
            raise ValidationException(msg)
        if not plan.is_active:
            raise InvalidStateTransitionException(current="inactive", target="update")

        values: dict[str, object] = {}
        if dto.name is not None:
            values["name"] = dto.name
        if dto.description is not None:
            values["description"] = dto.description
        if dto.amount is not None:
            values["amount"] = dto.amount
        if dto.currency is not None:
            values["currency"] = dto.currency
        if dto.interval is not None:
            values["interval"] = dto.interval.value
        if dto.interval_count is not None:
            values["interval_count"] = dto.interval_count
        if dto.trial_period_days is not None:
            values["trial_period_days"] = dto.trial_period_days
        if dto.tax_rate is not None:
            values["tax_rate"] = dto.tax_rate
        if dto.refund_policy is not None:
            values["refund_policy"] = dto.refund_policy
        if dto.features is not None:
            values["features"] = dto.features
        if dto.metadata is not None:
            values["metadata_"] = dto.metadata
        if not values:
            return _plan_to_response(plan)

        archive_result = await self.repos.plans.archive(plan_id)
        if isinstance(archive_result, Failure):
            return failure_envelope(archive_result.failure(), operation="update_plan")

        new_version = Plan(
            parent_plan_id=plan.id,
            razorpay_plan_id=plan.razorpay_plan_id,
            name=dto.name or plan.name,
            description=dto.description if dto.description is not None else plan.description,
            amount=dto.amount if dto.amount is not None else plan.amount,
            currency=dto.currency if dto.currency is not None else plan.currency,
            interval=dto.interval.value if dto.interval is not None else plan.interval,
            interval_count=(
                dto.interval_count if dto.interval_count is not None else plan.interval_count
            ),
            trial_period_days=(
                dto.trial_period_days if dto.trial_period_days is not None else plan.trial_period_days
            ),
            tax_rate=dto.tax_rate if dto.tax_rate is not None else plan.tax_rate,
            refund_policy=dto.refund_policy or plan.refund_policy,
            features=dto.features if dto.features is not None else plan.features,
            metadata_=dto.metadata if dto.metadata is not None else plan.metadata_,
            is_active=True,
        )
        created_result = await self.repos.plans.create(new_version)
        if isinstance(created_result, Failure):
            restore = await self.repos.plans.update(plan, values={"is_active": True})
            if isinstance(restore, Failure):
                logger.bind(operation="update_plan").error(restore.failure().message)
            return failure_envelope(created_result.failure(), operation="update_plan")
        created = created_result.unwrap()

        await self.repos.audit.create(
            AuditLog(
                entity_type="plan",
                entity_id=str(created.id),
                action="plan.updated",
                user_id=user_id,
                changes={**values, "parent_plan_id": str(plan.id)},
            )
        )
        return _plan_to_response(created)

    async def archive_plan(self, plan_id: str, *, user_id: str) -> ServiceResult[PlanResponse]:
        result = await self.repos.plans.archive(plan_id)
        if isinstance(result, Failure):
            return failure_envelope(result.failure(), operation="archive_plan")
        plan = result.unwrap()
        if plan is None:
            msg = "Plan not found"
            raise ValidationException(msg)
        await self.repos.audit.create(
            AuditLog(
                entity_type="plan",
                entity_id=str(plan.id),
                action="plan.archived",
                user_id=user_id,
                changes={"is_active": False},
            )
        )
        return _plan_to_response(plan)

    @staticmethod
    def _sync_razorpay_enabled() -> bool:
        return bool(get_settings().RAZORPAY_KEY_ID)
