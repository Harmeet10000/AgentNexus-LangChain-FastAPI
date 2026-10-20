"""Billing plan management service."""

from __future__ import annotations

import hashlib
import json
from typing import TYPE_CHECKING, Protocol, cast

from returns.result import Failure, Success
from sqlalchemy import select
from sqlalchemy.exc import SQLAlchemyError

from app.config import get_settings
from app.features.audit.models import AuditAction, AuditLog
from app.features.billing.providers import CreatePlanOperation, ProviderPlan
from app.features.billing.subscriptions.models import (
    ProviderOperationRecord,
    ProviderOperationState,
)
from app.kernel.result import add_database_error_note
from app.observability.logging import logger, trace_layer

from .dto import PlanResponse
from .exceptions import (
    PlanConflictError,
    PlanInfrastructureError,
    PlanNotFoundError,
    PlanValidationError,
)
from .models import Plan

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

    from app.features.audit.repository import AuditLogRepository
    from app.features.billing.plans.repository import PlanRepository
    from app.features.billing.subscriptions.coordinator import SubscriptionLifecycleCoordinator

    from .dto import PlanCreateDTO, PlanUpdateDTO
    from .exceptions import PlanResult


class _LegacyPlanProvider(Protocol):
    async def create_plan(
        self, *, name: str, amount: int, interval: str, currency: str, period: int
    ) -> dict[str, object]: ...


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

    def __init__(
        self,
        session: AsyncSession,
        plans: PlanRepository,
        audit: AuditLogRepository,
        *,
        lifecycle_coordinator: SubscriptionLifecycleCoordinator | None = None,
        razorpay: object | None = None,
    ) -> None:
        self.session = session
        self.plans = plans
        self.audit = audit
        self.lifecycle_coordinator = lifecycle_coordinator
        self._legacy_razorpay = razorpay

    @staticmethod
    def _derived_operation_key(prefix: str, payload: object) -> str:
        dump = getattr(payload, "model_dump", None)
        normalized = dump(mode="json") if callable(dump) else vars(payload)
        encoded = json.dumps(normalized, sort_keys=True, separators=(",", ":"), default=str)
        digest = hashlib.sha256(encoded.encode()).hexdigest()[:24]
        return f"{prefix}:{digest}"

    @classmethod
    def _request_binding(
        cls,
        operation: str,
        *,
        user_id: str,
        request_key: str | None,
        payload: object,
    ) -> tuple[str, str, str]:
        """Return durable operation key, idempotency scope, and fingerprint."""
        dump = getattr(payload, "model_dump", None)
        normalized = dump(mode="json") if callable(dump) else vars(payload)
        canonical = json.dumps(
            {
                "operation": operation,
                "user_id": user_id,
                "request_key": request_key or "<derived>",
                "payload": normalized,
            },
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        )
        fingerprint = hashlib.sha256(canonical.encode()).hexdigest()
        scope = f"{user_id}:{request_key or f'derived:{fingerprint}'}"
        return f"{operation}:{fingerprint[:48]}", scope, fingerprint

    async def _find_bound_request(self, scope: str, fingerprint: str) -> PlanResult[Plan | None]:
        finder = getattr(self.plans, "find_by_idempotency_scope", None)
        if finder is None:
            return Success(None)
        result = await finder(scope)
        if isinstance(result, Failure):
            return result
        plan = result.unwrap()
        if plan is None:
            return Success(None)
        stored = (plan.metadata_ or {}).get("billing_request_fingerprint")
        if stored != fingerprint:
            return Failure(
                PlanConflictError(
                    message="Idempotency key is already bound to a different plan request",
                    details={"scope": scope},
                    source="plan_service",
                )
            )
        return Success(plan)

    @trace_layer("service")
    async def list_plans(
        self, *, include_inactive: bool = False, limit: int = 50, offset: int = 0
    ) -> PlanResult[list[PlanResponse]]:
        if include_inactive:
            try:
                statement = (
                    select(Plan).order_by(Plan.created_at.desc()).limit(limit).offset(offset)
                )
                result = await self.session.execute(statement)
                return Success([_plan_to_response(p) for p in result.scalars().all()])
            except SQLAlchemyError as exc:
                add_database_error_note(exc, table="plans", operation="list_plans")
                logger.bind(operation="list_plans", table="plans", error=str(exc)).exception(
                    "list_plans failed"
                )
                return Failure(
                    PlanInfrastructureError(
                        message="Database error while listing plans",
                        details={"error": str(exc)},
                        source="plan_service",
                        operation="list_plans",
                    )
                )

        result = await self.plans.list_active(limit=limit, offset=offset)
        if isinstance(result, Failure):
            return result
        return Success([_plan_to_response(p) for p in result.unwrap()])

    @trace_layer("service")
    async def get_plan(self, plan_id: str) -> PlanResult[PlanResponse]:
        result = await self.plans.find_by_id(plan_id)
        if isinstance(result, Failure):
            return result
        plan = result.unwrap()
        if plan is None:
            return Failure(
                PlanNotFoundError(
                    message="Plan not found",
                    details={"plan_id": plan_id},
                    source="plan_service",
                    plan_id=plan_id,
                )
            )
        return Success(_plan_to_response(plan))

    @trace_layer("service")
    async def create_plan(  # noqa: PLR0912,PLR0914 -- idempotent create/recovery stages are explicit
        self,
        dto: PlanCreateDTO,
        *,
        user_id: str,
        idempotency_key: str | None = None,
    ) -> PlanResult[PlanResponse]:
        operation_key, idempotency_scope, request_fingerprint = self._request_binding(
            "billing:plan:create", user_id=user_id, request_key=idempotency_key, payload=dto
        )
        bound = await self._find_bound_request(idempotency_scope, request_fingerprint)
        if isinstance(bound, Failure):
            return bound
        existing_plan = bound.unwrap()
        if existing_plan is not None:
            if self._sync_razorpay_enabled() and not existing_plan.razorpay_plan_id:
                synced = await self._sync_provider_plan(existing_plan)
                if isinstance(synced, Failure):
                    return synced
                existing_plan = synced.unwrap()
            return Success(_plan_to_response(existing_plan))
        find_by_operation_key = getattr(self.plans, "find_by_operation_key", None)
        if find_by_operation_key is not None:
            existing_operation = await find_by_operation_key(operation_key)
            if isinstance(existing_operation, Failure):
                return existing_operation
            existing_plan = existing_operation.unwrap()
            if existing_plan is not None:
                if self._sync_razorpay_enabled() and not existing_plan.razorpay_plan_id:
                    synced = await self._sync_provider_plan(existing_plan)
                    if isinstance(synced, Failure):
                        return synced
                    existing_plan = synced.unwrap()
                return Success(_plan_to_response(existing_plan))
        existing = await self.plans.find_by_name(dto.name)
        if isinstance(existing, Failure):
            return existing
        if existing.unwrap() is not None:
            return Failure(
                PlanConflictError(
                    message=f"A plan named '{dto.name}' already exists",
                    details={"name": dto.name},
                    source="plan_service",
                )
            )

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
            metadata_={
                **dto.metadata,
                "billing_operation_key": operation_key,
                "billing_idempotency_scope": idempotency_scope,
                "billing_request_fingerprint": request_fingerprint,
                "billing_provider_sync": "pending" if self._sync_razorpay_enabled() else "disabled",
            },
        )
        result = await self.plans.create(plan)
        if isinstance(result, Failure):
            # A concurrent request can win the active-name/operation race.
            # Recover its committed row instead of surfacing a false conflict.
            recovered = await self._find_bound_request(idempotency_scope, request_fingerprint)
            recovered_plan = recovered.unwrap() if isinstance(recovered, Success) else None
            if recovered_plan is not None:
                return Success(_plan_to_response(recovered_plan))
            return result
        created = result.unwrap()

        if self._sync_razorpay_enabled():
            synced = await self._sync_provider_plan(created)
            if isinstance(synced, Failure):
                return synced
            created = synced.unwrap()

        audit_result = await self.audit.create(
            AuditLog(
                entity_type="plan",
                entity_id=str(created.id),
                action=AuditAction.PLAN_CREATED.value,
                user_id=user_id,
                changes={"name": created.name, "amount": created.amount},
            )
        )
        if isinstance(audit_result, Failure):
            error = audit_result.failure()
            return Failure(
                inner_value=PlanInfrastructureError(
                    message=error.message,
                    details=error.details,
                    source="plan_service",
                    operation="create_audit_log",
                )
            )
        return Success(inner_value=_plan_to_response(created))

    @trace_layer("service")
    async def update_plan(  # noqa: PLR0912, PLR0914, PLR0915
        self,
        plan_id: str,
        dto: PlanUpdateDTO,
        *,
        user_id: str,
        idempotency_key: str | None = None,
    ) -> PlanResult[PlanResponse]:
        """Create a new plan version; keep the current one for existing subscribers.

        Requirement 24: updates never mutate the original record. A new Plan
        row is created (``parent_plan_id`` → current plan) and the previous
        version is deactivated so new subscriptions bind to the latest pricing.
        """
        operation_key, idempotency_scope, request_fingerprint = self._request_binding(
            f"billing:plan:update:{plan_id}",
            user_id=user_id,
            request_key=idempotency_key,
            payload=dto,
        )
        bound = await self._find_bound_request(idempotency_scope, request_fingerprint)
        if isinstance(bound, Failure):
            return bound
        existing_version = bound.unwrap()
        if existing_version is not None:
            if self._sync_razorpay_enabled() and not existing_version.razorpay_plan_id:
                synced = await self._sync_provider_plan(existing_version)
                if isinstance(synced, Failure):
                    return synced
                existing_version = synced.unwrap()
            return Success(_plan_to_response(existing_version))
        find_by_operation_key = getattr(self.plans, "find_by_operation_key", None)
        if find_by_operation_key is not None:
            existing_operation = await find_by_operation_key(operation_key)
            if isinstance(existing_operation, Failure):
                return existing_operation
            existing_version = existing_operation.unwrap()
            if existing_version is not None:
                if self._sync_razorpay_enabled() and not existing_version.razorpay_plan_id:
                    synced = await self._sync_provider_plan(existing_version)
                    if isinstance(synced, Failure):
                        return synced
                    existing_version = synced.unwrap()
                return Success(_plan_to_response(existing_version))
        result: PlanResult[Plan | None] = await self.plans.find_by_id(plan_id)
        if isinstance(result, Failure):
            return result
        plan = result.unwrap()
        if plan is None:
            return Failure(
                inner_value=PlanNotFoundError(
                    message="Plan not found",
                    details={"plan_id": plan_id},
                    source="plan_service",
                    plan_id=plan_id,
                )
            )
        if not plan.is_active:
            return Failure(
                inner_value=PlanValidationError(
                    message="Cannot update inactive plan",
                    details={"plan_id": plan_id},
                    source="plan_service",
                )
            )

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
            return Success(_plan_to_response(plan))

        archive_result = await self.plans.archive(plan_id)
        if isinstance(archive_result, Failure):
            return archive_result

        financial_changed = any(
            value is not None
            for value in (dto.amount, dto.currency, dto.interval, dto.interval_count)
        )
        new_version = Plan(
            parent_plan_id=plan.id,
            razorpay_plan_id=None if financial_changed else plan.razorpay_plan_id,
            name=dto.name or plan.name,
            description=dto.description if dto.description is not None else plan.description,
            amount=dto.amount if dto.amount is not None else plan.amount,
            currency=dto.currency if dto.currency is not None else plan.currency,
            interval=dto.interval.value if dto.interval is not None else plan.interval,
            interval_count=(
                dto.interval_count if dto.interval_count is not None else plan.interval_count
            ),
            trial_period_days=(
                dto.trial_period_days
                if dto.trial_period_days is not None
                else plan.trial_period_days
            ),
            tax_rate=dto.tax_rate if dto.tax_rate is not None else plan.tax_rate,
            refund_policy=dto.refund_policy or plan.refund_policy,
            features=dto.features if dto.features is not None else plan.features,
            metadata_=dto.metadata if dto.metadata is not None else plan.metadata_,
            is_active=True,
        )
        new_version.metadata_["billing_operation_key"] = operation_key
        new_version.metadata_["billing_idempotency_scope"] = idempotency_scope
        new_version.metadata_["billing_request_fingerprint"] = request_fingerprint
        if self._sync_razorpay_enabled():
            new_version.metadata_["billing_provider_sync"] = "pending"
        created_result = await self.plans.create(new_version)
        if isinstance(created_result, Failure):
            restore = await self.plans.update(plan, values={"is_active": True})
            if isinstance(restore, Failure):
                restore_error = restore.failure()
                logger.bind(
                    operation="update_plan",
                    error=restore_error.message,
                    details=restore_error.details,
                ).error("Plan restore failed")
            return created_result
        created = created_result.unwrap()

        if self._sync_razorpay_enabled() and financial_changed:
            synced = await self._sync_provider_plan(created)
            if isinstance(synced, Failure):
                return synced
            created = synced.unwrap()

        audit_result = await self.audit.create(
            AuditLog(
                entity_type="plan",
                entity_id=str(created.id),
                action=AuditAction.PLAN_UPDATED.value,
                user_id=user_id,
                changes={**values, "parent_plan_id": str(plan.id)},
            )
        )
        if isinstance(audit_result, Failure):
            error = audit_result.failure()
            return Failure(
                inner_value=PlanInfrastructureError(
                    message=error.message,
                    details=error.details,
                    source="plan_service",
                    operation="create_audit_log",
                )
            )
        return Success(inner_value=_plan_to_response(created))

    @trace_layer("service")
    async def archive_plan(self, plan_id: str, *, user_id: str) -> PlanResult[PlanResponse]:
        result: PlanResult[Plan | None] = await self.plans.archive(plan_id)
        if isinstance(result, Failure):
            return result
        plan: Plan | None = result.unwrap()
        if plan is None:
            return Failure(
                inner_value=PlanNotFoundError(
                    message="Plan not found",
                    details={"plan_id": plan_id},
                    source="plan_service",
                    plan_id=plan_id,
                )
            )
        audit_result = await self.audit.create(
            entry=AuditLog(
                entity_type="plan",
                entity_id=str(plan.id),
                action=AuditAction.PLAN_ARCHIVED.value,
                user_id=user_id,
                changes={"is_active": False},
            )
        )
        if isinstance(audit_result, Failure):
            error = audit_result.failure()
            return Failure(
                inner_value=PlanInfrastructureError(
                    message=error.message,
                    details=error.details,
                    source="plan_service",
                    operation="create_audit_log",
                )
            )
        return Success(inner_value=_plan_to_response(plan))

    @trace_layer("service")
    async def recover_pending_provider_plans(self, *, limit: int = 100) -> PlanResult[int]:
        """Bind local plans to already-addressable provider operations.

        A sweep only considers a durable ``create_plan`` ledger row that is
        already succeeded or ambiguous.  It therefore reconciles an existing
        provider resource and cannot create a second remote plan merely
        because a worker crashed before persisting the local binding.
        """
        try:
            plans = list(
                (
                    await self.session.execute(
                        select(Plan)
                        .where(Plan.razorpay_plan_id.is_(None))
                        .order_by(Plan.created_at.asc())
                        .limit(limit)
                    )
                )
                .scalars()
                .all()
            )
            recovered = 0
            for plan in plans:
                # The provider ledger key is the stable correlation used by
                # ``_sync_provider_plan``.  The HTTP idempotency key is a
                # request fingerprint and is intentionally not derivable from
                # the local plan UUID.
                key = f"billing:plan:create:{plan.id}"
                operation = (
                    await self.session.execute(
                        select(ProviderOperationRecord).where(
                            ProviderOperationRecord.correlation_id == key,
                            ProviderOperationRecord.operation_kind == "create_plan",
                            ProviderOperationRecord.state.in_(
                                (
                                    ProviderOperationState.PENDING.value,
                                    ProviderOperationState.SUCCEEDED.value,
                                    ProviderOperationState.UNKNOWN.value,
                                    ProviderOperationState.RECONCILIATION.value,
                                )
                            ),
                        )
                    )
                ).scalar_one_or_none()
                if operation is None:
                    continue
                synced = await self._sync_provider_plan(plan)
                if isinstance(synced, Failure):
                    logger.bind(
                        operation="recover_pending_provider_plans",
                        plan_id=str(plan.id),
                        error=synced.failure().message,
                    ).warning("Provider plan binding recovery deferred")
                    continue
                recovered += 1
            return Success(recovered)
        except SQLAlchemyError as exc:
            add_database_error_note(
                exc, table="provider_operation_records", operation="recover_pending_provider_plans"
            )
            await self.session.rollback()
            return Failure(
                PlanInfrastructureError(
                    message="Database error while recovering provider plan bindings",
                    details={"error": str(exc)},
                    source="plan_service",
                    operation="recover_pending_provider_plans",
                )
            )

    @staticmethod
    def _sync_razorpay_enabled() -> bool:
        return bool(get_settings().RAZORPAY_KEY_ID)

    async def _sync_provider_plan(self, plan: Plan) -> PlanResult[Plan]:
        correlation_id = f"billing:plan:create:{plan.id}"
        operation = CreatePlanOperation(
            name=plan.name,
            amount=plan.amount,
            currency=plan.currency,
            period=plan.interval,
            interval=plan.interval_count,
            correlation_id=correlation_id,
            notes={"plan_id": str(plan.id)},
        )
        if self.lifecycle_coordinator is None:
            if self._legacy_razorpay is None:
                return Failure(
                    PlanInfrastructureError(
                        message="Billing provider coordinator is not configured",
                        source="plan_service",
                        operation="create_plan",
                    )
                )
            try:
                payload = await cast("_LegacyPlanProvider", self._legacy_razorpay).create_plan(
                    name=plan.name,
                    amount=plan.amount,
                    interval=plan.interval,
                    currency=plan.currency,
                    period=plan.interval_count,
                )
                provider_id = payload.get("id")
            except Exception as exc:  # noqa: BLE001 — explicit legacy test adapter only
                exc.add_note(f"operation=create_plan, plan_id={plan.id}")
                return Failure(
                    PlanInfrastructureError(
                        message="Billing provider plan creation failed",
                        source="plan_service",
                        operation="create_plan",
                    )
                )
            if not isinstance(provider_id, str) or not provider_id:
                return Failure(
                    PlanInfrastructureError(
                        message="Billing provider did not return a plan id",
                        source="plan_service",
                        operation="create_plan",
                    )
                )
        else:
            result = await self.lifecycle_coordinator.execute_provider_operation(
                operation, subscription_id=None, idempotency_key=correlation_id
            )
            if isinstance(result, Failure):
                error = result.failure()
                return Failure(
                    PlanInfrastructureError(
                        message=error.message,
                        details=error.details,
                        source="plan_service",
                        operation="create_plan",
                    )
                )
            value = result.unwrap().value
            if not isinstance(value, ProviderPlan):
                return Failure(
                    PlanInfrastructureError(
                        message="Billing provider returned an invalid plan",
                        source="plan_service",
                        operation="create_plan",
                    )
                )
            provider_id = value.provider_plan_id
        return await self.plans.update(
            plan,
            values={
                "razorpay_plan_id": provider_id,
                "metadata_": {**plan.metadata_, "billing_provider_sync": "synced"},
            },
        )
