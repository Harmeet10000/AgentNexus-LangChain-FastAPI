"""Subscription persistence operations with optimistic locking."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from typing import TYPE_CHECKING, Protocol, cast, override
from uuid import UUID

from returns.result import Failure, Success
from sqlalchemy import func, or_, select, update
from sqlalchemy.exc import IntegrityError, SQLAlchemyError

from app.features.audit.models import AuditLog
from app.features.billing.providers import (
    CancelSubscriptionOperation,
    ChangeSubscriptionPlanOperation,
    CreateCustomerOperation,
    CreatePaymentLinkOperation,
    CreatePlanOperation,
    CreateRefundOperation,
    CreateSubscriptionOperation,
    FindCustomerByEmailOperation,
    FindPaymentLinkByCorrelationOperation,
    FindPlanByCorrelationOperation,
    FindRefundByCorrelationOperation,
    FindSubscriptionByCorrelationOperation,
    ObserveSubscriptionOperation,
    PauseSubscriptionOperation,
    ProviderAuthenticationError,
    ProviderCustomer,
    ProviderErrorCode,
    ProviderInvalidRequestError,
    ProviderManualReviewRequiredError,
    ProviderNotFoundError,
    ProviderOperationKind,
    ProviderOutcomeUnknownError,
    ProviderPaymentLink,
    ProviderPlan,
    ProviderRateLimitedError,
    ProviderReceipt,
    ProviderReconciliationRequiredError,
    ProviderSubscriptionState,
    ProviderSuccess,
    ProviderUnavailableError,
    ProviderUnexpectedError,
    ResumeSubscriptionOperation,
)
from app.kernel.result import add_database_error_note
from app.observability.logging import trace_layer

from ..exceptions import (
    LifecycleCoordinatorPersistenceError,
    LifecycleCoordinatorValidationError,
    LifecycleCoordinatorVersionConflictError,
)
from .coordinator import (
    CoordinatorStatus,
    IntentStatus,
    LifecycleOutcome,
    LifecyclePersistence,
    LifecycleRecord,
    ProviderOperationIntent,
)
from .exceptions import (
    SubscriptionDuplicateError,
    SubscriptionInvalidTransitionError,
    SubscriptionNotFoundError,
    SubscriptionTransientInfrastructureError,
    SubscriptionValidationError,
    SubscriptionVersionConflictError,
)
from .lifecycle import (
    ActiveState,
    AuthenticatedState,
    CancellationIntent,
    CancelledState,
    CreatedState,
    EffectKind,
    ExpiredState,
    HaltedState,
    PastDueState,
    PausedState,
    PendingUpgrade,
    RemoteSyncState,
    SubscriptionSnapshot,
    SubscriptionStateKind,
    TransitionEffect,
    TransitionOutcome,
    TransitionPlan,
)
from .models import (
    ProviderOperationRecord,
    ProviderOperationState,
    Subscription,
    SubscriptionStatus,
)


@dataclass(frozen=True, slots=True)
class ProviderOperationInput:
    """Normalized facts required to create a durable provider operation."""

    correlation_id: str
    idempotency_key: str
    operation_kind: str
    subscription_id: UUID | None = None
    request_data: Mapping[str, object] | None = None


@dataclass(frozen=True, slots=True)
class ProviderOperationUpdate:
    """Patch accepted by the persistence seam after a provider attempt."""

    state: ProviderOperationState | str
    attempts: int | None = None
    provider_resource_id: str | None = None
    error_code: str | None = None
    error_message: str | None = None
    observation_data: Mapping[str, object] | None = None


class SubscriptionLifecyclePersistence(Protocol):
    """Narrow persistence seam for a future lifecycle coordinator.

    The protocol is intentionally defined next to the repository adapter: the
    coordinator needs snapshots and operation-ledger transitions, not the
    subscription listing/legacy billing methods on ``SubscriptionRepository``.
    """

    async def load_snapshot(
        self, subscription_id: str | UUID
    ) -> SubscriptionResult[SubscriptionSnapshot]: ...

    async def save_snapshot(
        self, snapshot: SubscriptionSnapshot, *, expected_version: int
    ) -> SubscriptionResult[SubscriptionSnapshot]: ...

    async def create_provider_operation(
        self, operation: ProviderOperationInput
    ) -> SubscriptionResult[ProviderOperationRecord]: ...

    async def find_provider_operation(
        self, idempotency_key: str
    ) -> SubscriptionResult[ProviderOperationRecord | None]: ...

    async def find_provider_operation_by_correlation(
        self, correlation_id: str
    ) -> SubscriptionResult[ProviderOperationRecord | None]: ...

    async def update_provider_operation(
        self, operation_id: str | UUID, patch: ProviderOperationUpdate
    ) -> SubscriptionResult[ProviderOperationRecord]: ...


if TYPE_CHECKING:
    from collections.abc import Mapping
    from typing import Any

    from sqlalchemy.ext.asyncio import AsyncSession
    from sqlalchemy.sql.selectable import Select

    from app.features.audit.repository import AuditLogRepository
    from app.features.billing.providers import ProviderError, ProviderOperation

    from .coordinator import LifecycleCoordinatorResult
    from .exceptions import SubscriptionResult
    from .lifecycle import SubscriptionState


# Requirement 4.1-4.6: allowed state transitions.
_ALLOWED_TRANSITIONS: dict[SubscriptionStatus, set[SubscriptionStatus]] = {
    SubscriptionStatus.CREATED: {SubscriptionStatus.AUTHENTICATED},
    SubscriptionStatus.AUTHENTICATED: {SubscriptionStatus.ACTIVE},
    SubscriptionStatus.ACTIVE: {
        SubscriptionStatus.PAST_DUE,
        SubscriptionStatus.PAUSED,
        SubscriptionStatus.CANCELLED,
    },
    SubscriptionStatus.PAST_DUE: {SubscriptionStatus.ACTIVE, SubscriptionStatus.HALTED},
    SubscriptionStatus.HALTED: {SubscriptionStatus.ACTIVE, SubscriptionStatus.CANCELLED},
    SubscriptionStatus.PAUSED: {SubscriptionStatus.ACTIVE, SubscriptionStatus.CANCELLED},
    SubscriptionStatus.CANCELLED: set(),
    SubscriptionStatus.EXPIRED: set(),
}


def validate_transition(
    current: str | SubscriptionStatus, target: str | SubscriptionStatus
) -> bool:
    """Check whether a status transition is allowed by the state machine."""
    try:
        current_status = SubscriptionStatus(current)
        target_status = SubscriptionStatus(target)
    except ValueError:
        # A row holding a status outside the enum (no DB CHECK constraint)
        # is not a transition the machine allows; the caller maps False to
        # the typed invalid-transition failure instead of a 500.
        return False
    return target_status in _ALLOWED_TRANSITIONS.get(current_status, set())


class SubscriptionRepository:
    """Repository for subscription lifecycle."""

    def __init__(self, session: AsyncSession) -> None:
        self.session: AsyncSession = session

    @trace_layer("repository")
    async def create(self, subscription: Subscription) -> SubscriptionResult[Subscription]:
        try:
            self.session.add(subscription)
            await self.session.flush()
            return Success(subscription)
        except IntegrityError as exc:
            add_database_error_note(exc, table="subscriptions")
            await self.session.rollback()
            return Failure(
                SubscriptionDuplicateError(
                    message="An active subscription already exists for this user and plan",
                    details={
                        "user_id": subscription.user_id,
                        "plan_id": str(subscription.plan_id),
                        "error": str(exc),
                    },
                    source="subscription_repository",
                    user_id=subscription.user_id,
                    plan_id=str(subscription.plan_id),
                )
            )
        except SQLAlchemyError as exc:
            add_database_error_note(exc, table="subscriptions")
            await self.session.rollback()
            return Failure(
                SubscriptionTransientInfrastructureError(
                    message="Database error while creating subscription",
                    details={"error": str(exc)},
                    source="subscription_repository",
                    operation="create",
                )
            )

    @trace_layer("repository")
    async def find_by_id(
        self, subscription_id: str | UUID
    ) -> SubscriptionResult[Subscription | None]:
        try:
            statement: Select[tuple[Subscription]] = select(Subscription).where(
                Subscription.id == subscription_id,
                Subscription.deleted_at.is_(None),
            )
            result = await self.session.execute(statement)
            subscription: Subscription | None = result.scalar_one_or_none()
            if subscription is None:
                return Failure(
                    SubscriptionNotFoundError(
                        message="Subscription not found",
                        details={"subscription_id": str(subscription_id)},
                        source="subscription_repository",
                        subscription_id=str(subscription_id),
                    )
                )
            return Success(subscription)
        except SQLAlchemyError as exc:
            add_database_error_note(exc, table="subscriptions")
            await self.session.rollback()
            return Failure(
                SubscriptionTransientInfrastructureError(
                    message="Database error while fetching subscription",
                    details={"subscription_id": str(subscription_id), "error": str(exc)},
                    source="subscription_repository",
                    operation="find_by_id",
                )
            )

    @trace_layer("repository")
    async def find_by_razorpay_id(
        self, razorpay_subscription_id: str
    ) -> SubscriptionResult[Subscription | None]:
        try:
            statement: Select[tuple[Subscription]] = select(Subscription).where(
                Subscription.razorpay_subscription_id == razorpay_subscription_id,
                Subscription.deleted_at.is_(None),
            )
            result = await self.session.execute(statement)
            subscription: Subscription | None = result.scalar_one_or_none()
            if subscription is None:
                return Failure(
                    SubscriptionNotFoundError(
                        message="Subscription not found for Razorpay ID",
                        details={"razorpay_subscription_id": razorpay_subscription_id},
                        source="subscription_repository",
                    )
                )
            return Success(subscription)
        except SQLAlchemyError as exc:
            add_database_error_note(exc, table="subscriptions")
            await self.session.rollback()
            return Failure(
                SubscriptionTransientInfrastructureError(
                    message="Database error while fetching subscription by Razorpay ID",
                    details={
                        "razorpay_subscription_id": razorpay_subscription_id,
                        "error": str(exc),
                    },
                    source="subscription_repository",
                    operation="find_by_razorpay_id",
                )
            )

    @trace_layer("repository")
    async def find_by_user_and_plan(
        self, user_id: str, plan_id: str | UUID
    ) -> SubscriptionResult[Subscription | None]:
        try:
            statement: Select[tuple[Subscription]] = (
                select(Subscription)
                .where(
                    Subscription.user_id == user_id,
                    Subscription.plan_id == plan_id,
                    Subscription.status.in_(
                        [
                            SubscriptionStatus.CREATED,
                            SubscriptionStatus.AUTHENTICATED,
                            SubscriptionStatus.ACTIVE,
                            SubscriptionStatus.PAST_DUE,
                            SubscriptionStatus.PAUSED,
                            SubscriptionStatus.HALTED,
                        ]
                    ),
                    Subscription.deleted_at.is_(None),
                )
                .limit(1)
            )
            result = await self.session.execute(statement)
            return Success(result.scalar_one_or_none())
        except SQLAlchemyError as exc:
            add_database_error_note(exc, table="subscriptions")
            await self.session.rollback()
            return Failure(
                SubscriptionTransientInfrastructureError(
                    message="Database error while finding subscription by user and plan",
                    details={"user_id": user_id, "plan_id": str(plan_id), "error": str(exc)},
                    source="subscription_repository",
                    operation="find_by_user_and_plan",
                )
            )

    @trace_layer("repository")
    async def list_by_user(
        self,
        user_id: str,
        *,
        status: SubscriptionStatus | None = None,
        plan_id: str | UUID | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> SubscriptionResult[tuple[list[Subscription], int]]:
        """Return (items, total_count) for the requested slice."""
        try:
            conditions = [Subscription.user_id == user_id, Subscription.deleted_at.is_(None)]
            if status is not None:
                conditions.append(Subscription.status == status)
            if plan_id is not None:
                conditions.append(Subscription.plan_id == plan_id)

            total_statement = select(func.count()).select_from(Subscription).where(*conditions)
            total = (await self.session.execute(total_statement)).scalar_one()

            statement: Select[tuple[Subscription]] = (
                select(Subscription)
                .where(*conditions)
                .order_by(Subscription.created_at.desc())
                .limit(limit)
                .offset(offset)
            )
            result = await self.session.execute(statement)
            return Success((list(result.scalars().all()), int(total)))
        except SQLAlchemyError as exc:
            add_database_error_note(exc, table="subscriptions")
            await self.session.rollback()
            return Failure(
                SubscriptionTransientInfrastructureError(
                    message="Database error while listing subscriptions",
                    details={"user_id": user_id, "error": str(exc)},
                    source="subscription_repository",
                    operation="list_by_user",
                )
            )

    @trace_layer("repository")
    async def update_with_lock(
        self,
        subscription: Subscription,
        expected_version: int,
        *,
        values: dict[str, object],
    ) -> SubscriptionResult[Subscription]:
        """Optimistic-lock update (Requirement 29).

        Runs ``UPDATE ... WHERE id = :id AND version = :expected`` and
        bumps the version in the same statement, so concurrent writers
        cannot lose updates.
        """
        try:
            statement = (
                update(Subscription)
                .where(
                    Subscription.id == subscription.id,
                    Subscription.version == expected_version,
                )
                .values(**values, version=Subscription.version + 1, updated_at=datetime.now(tz=UTC))
                .returning(Subscription)
            )
            result = await self.session.execute(statement)
            updated = result.scalar_one_or_none()
            if updated is None:
                return Failure(
                    SubscriptionVersionConflictError(
                        message="Subscription was modified concurrently; refetch and retry",
                        details={
                            "subscription_id": str(subscription.id),
                            "expected_version": expected_version,
                        },
                        source="subscription_repository",
                        subscription_id=str(subscription.id),
                        expected_version=expected_version,
                    )
                )
            return Success(updated)
        except SQLAlchemyError as exc:
            add_database_error_note(exc, table="subscriptions")
            await self.session.rollback()
            return Failure(
                SubscriptionTransientInfrastructureError(
                    message="Database error while updating subscription",
                    details={"subscription_id": str(subscription.id), "error": str(exc)},
                    source="subscription_repository",
                    operation="update_with_lock",
                )
            )

    @trace_layer("repository")
    async def update_status(
        self,
        subscription: Subscription,
        new_status: SubscriptionStatus,
        *,
        expected_version: int,
        extra_values: dict[str, object] | None = None,
    ) -> SubscriptionResult[Subscription]:
        """Update status with state-machine validation and optimistic locking."""
        if not validate_transition(subscription.status, new_status):
            return Failure(
                SubscriptionInvalidTransitionError(
                    message=f"Invalid subscription state transition: {subscription.status} -> {new_status}",
                    details={
                        "subscription_id": str(subscription.id),
                        "current": subscription.status,
                        "target": new_status.value,
                    },
                    source="subscription_repository",
                    subscription_id=str(subscription.id),
                    current=subscription.status,
                    target=new_status.value,
                )
            )
        values: dict[str, object] = {"status": new_status.value}
        if extra_values:
            values.update(extra_values)
        return await self.update_with_lock(subscription, expected_version, values=values)

    @trace_layer("repository")
    async def increment_retry_count(
        self, subscription: Subscription, *, expected_version: int
    ) -> SubscriptionResult[Subscription]:
        return await self.update_with_lock(
            subscription,
            expected_version,
            values={"retry_count": Subscription.retry_count + 1},
        )

    @trace_layer("repository")
    async def reset_retry_count(
        self, subscription: Subscription, *, expected_version: int
    ) -> SubscriptionResult[Subscription]:
        return await self.update_with_lock(
            subscription, expected_version, values={"retry_count": 0}
        )

    # ---- lifecycle coordinator persistence seam -------------------------

    @trace_layer("repository")
    async def load_snapshot(
        self, subscription_id: str | UUID
    ) -> SubscriptionResult[SubscriptionSnapshot]:
        result = await self.find_by_id(subscription_id)
        if isinstance(result, Failure):
            return result
        return _to_lifecycle_snapshot(result.unwrap())

    @trace_layer("repository")
    async def save_snapshot(
        self, snapshot: SubscriptionSnapshot, *, expected_version: int
    ) -> SubscriptionResult[SubscriptionSnapshot]:
        values = _snapshot_values(snapshot)
        result = await self.update_with_lock(
            _subscription_ref(snapshot), expected_version, values=values
        )
        if isinstance(result, Failure):
            return result
        return _to_lifecycle_snapshot(result.unwrap())

    @trace_layer("repository")
    async def create_provider_operation(
        self, operation: ProviderOperationInput
    ) -> SubscriptionResult[ProviderOperationRecord]:
        record = ProviderOperationRecord(
            subscription_id=operation.subscription_id,
            correlation_id=operation.correlation_id,
            idempotency_key=operation.idempotency_key,
            operation_kind=operation.operation_kind,
            request_data=dict(operation.request_data or {}),
        )
        try:
            self.session.add(record)
            await self.session.flush()
            return Success(record)
        except IntegrityError as exc:
            add_database_error_note(
                exc,
                table="provider_operation_records",
                operation="create_provider_operation",
                context={"idempotency_key": operation.idempotency_key},
            )
            await self.session.rollback()
            # A duplicate idempotency key is a successful durable lookup, not
            # permission to issue the mutation a second time.
            existing = await self._find_operation(operation.idempotency_key)
            if existing is not None:
                return Success(existing)
            return Failure(
                SubscriptionTransientInfrastructureError(
                    message="Database error while recording provider operation",
                    details={"idempotency_key": operation.idempotency_key},
                    source="subscription_repository",
                    operation="create_provider_operation",
                )
            )
        except SQLAlchemyError as exc:
            add_database_error_note(exc, table="provider_operation_records")
            await self.session.rollback()
            return Failure(
                SubscriptionTransientInfrastructureError(
                    message="Database error while recording provider operation",
                    details={"idempotency_key": operation.idempotency_key},
                    source="subscription_repository",
                    operation="create_provider_operation",
                )
            )

    @trace_layer("repository")
    async def find_provider_operation(
        self, idempotency_key: str
    ) -> SubscriptionResult[ProviderOperationRecord | None]:
        try:
            return Success(await self._find_operation(idempotency_key))
        except SQLAlchemyError as exc:
            add_database_error_note(exc, table="provider_operation_records")
            await self.session.rollback()
            return Failure(
                SubscriptionTransientInfrastructureError(
                    message="Database error while finding provider operation",
                    details={"idempotency_key": idempotency_key},
                    source="subscription_repository",
                    operation="find_provider_operation",
                )
            )

    async def _find_operation(self, idempotency_key: str) -> ProviderOperationRecord | None:
        statement = select(ProviderOperationRecord).where(
            ProviderOperationRecord.idempotency_key == idempotency_key
        )
        result = await self.session.execute(statement)
        return result.scalar_one_or_none()

    @trace_layer("repository")
    async def find_provider_operation_by_correlation(
        self, correlation_id: str
    ) -> SubscriptionResult[ProviderOperationRecord | None]:
        try:
            statement = (
                select(ProviderOperationRecord)
                .where(ProviderOperationRecord.correlation_id == correlation_id)
                .order_by(ProviderOperationRecord.created_at.desc())
                .limit(1)
            )
            result = await self.session.execute(statement)
            return Success(result.scalar_one_or_none())
        except SQLAlchemyError as exc:
            add_database_error_note(exc, table="provider_operation_records")
            await self.session.rollback()
            return Failure(
                SubscriptionTransientInfrastructureError(
                    message="Database error while finding provider operation by correlation",
                    details={"correlation_id": correlation_id},
                    source="subscription_repository",
                    operation="find_provider_operation_by_correlation",
                )
            )

    @trace_layer("repository")
    async def update_provider_operation(
        self, operation_id: str | UUID, patch: ProviderOperationUpdate
    ) -> SubscriptionResult[ProviderOperationRecord]:
        state = _operation_state(patch.state)
        if state is None:
            return Failure(
                SubscriptionValidationError(
                    message="Unknown provider operation state",
                    details={"state": str(patch.state)},
                    source="subscription_repository",
                )
            )
        values: dict[str, object] = {"state": state.value, "updated_at": datetime.now(tz=UTC)}
        if patch.attempts is not None:
            if patch.attempts < 0:
                return Failure(
                    SubscriptionValidationError(
                        message="Provider operation attempts must be non-negative",
                        details={"attempts": patch.attempts},
                        source="subscription_repository",
                    )
                )
            values["attempts"] = patch.attempts
            attempt_time = datetime.now(tz=UTC)
            values["last_attempt_at"] = attempt_time
            if patch.attempts > 0:
                values["started_at"] = attempt_time
        if patch.provider_resource_id is not None:
            values["provider_resource_id"] = patch.provider_resource_id
        if patch.error_code is not None:
            values["error_code"] = _sanitize_provider_error(patch.error_code, limit=64)
        if patch.error_message is not None:
            values["error_message"] = _sanitize_provider_error(patch.error_message, limit=1000)
        if patch.observation_data is not None:
            values["observation_data"] = dict(patch.observation_data)
        if state in {
            ProviderOperationState.SUCCEEDED,
            ProviderOperationState.FAILED,
            ProviderOperationState.MANUAL_REVIEW,
        }:
            values["completed_at"] = datetime.now(tz=UTC)
        try:
            statement = (
                update(ProviderOperationRecord)
                .where(ProviderOperationRecord.id == operation_id)
                .values(**values)
                .returning(ProviderOperationRecord)
            )
            result = await self.session.execute(statement)
            updated = result.scalar_one_or_none()
            if updated is None:
                return Failure(
                    SubscriptionNotFoundError(
                        message="Provider operation not found",
                        details={"operation_id": str(operation_id)},
                        source="subscription_repository",
                    )
                )
            return Success(updated)
        except SQLAlchemyError as exc:
            add_database_error_note(exc, table="provider_operation_records")
            await self.session.rollback()
            return Failure(
                SubscriptionTransientInfrastructureError(
                    message="Database error while updating provider operation",
                    details={"operation_id": str(operation_id)},
                    source="subscription_repository",
                    operation="update_provider_operation",
                )
            )


class SqlAlchemyLifecyclePersistence(SubscriptionRepository, LifecyclePersistence):
    """SQL adapter for the coordinator's transaction and operation ledger.

    ``stage`` is deliberately the only place that advances the local
    lifecycle version.  It commits the desired snapshot and every provider
    intent before returning, so a process crash cannot leave a remote call
    without a durable operation key.  Provider acknowledgements update only
    integration facts; they never create a second lifecycle transition.
    """

    _COMMAND_KIND = "__lifecycle_command__"

    def __init__(self, session: AsyncSession, *, audit: AuditLogRepository | None = None) -> None:
        super().__init__(session)
        # Kept as an explicit dependency for the production composition root;
        # stage writes directly with the shared session to preserve atomicity.
        self.audit = audit

    @override
    async def load(self, subscription_id: str) -> LifecycleCoordinatorResult[LifecycleRecord]:
        try:
            row = await self._subscription(subscription_id)
            if row is None:
                return _coordinator_failure("load", "Subscription was not found")
            snapshot = _to_lifecycle_snapshot(row)
            if isinstance(snapshot, Failure):
                return _coordinator_failure("load", snapshot.failure().message)
            return Success(_lifecycle_record(row, snapshot.unwrap()))
        except SQLAlchemyError as exc:
            add_database_error_note(exc, table="subscriptions", operation="load")
            await self.session.rollback()
            return Failure(_coordinator_failure("load", _safe_db_error(exc)).failure())

    @override
    async def find_command(
        self, idempotency_key: str
    ) -> LifecycleCoordinatorResult[LifecycleOutcome | None]:
        try:
            statement = select(ProviderOperationRecord).where(
                ProviderOperationRecord.idempotency_key == idempotency_key,
                ProviderOperationRecord.operation_kind == self._COMMAND_KIND,
            )
            row = (await self.session.execute(statement)).scalar_one_or_none()
            if row is None:
                return Success(None)
            payload = row.observation_data.get("outcome")
            outcome = _outcome_from_payload(payload)
            if outcome is None:
                return _coordinator_failure("find_command", "Stored command outcome is invalid")
            return Success(outcome)
        except SQLAlchemyError as exc:
            add_database_error_note(
                exc, table="provider_operation_records", operation="find_command"
            )
            await self.session.rollback()
            return Failure(_coordinator_failure("find_command", _safe_db_error(exc)).failure())

    @override
    async def find_intents(
        self, idempotency_key: str
    ) -> LifecycleCoordinatorResult[tuple[ProviderOperationIntent, ...]]:
        try:
            statement = (
                select(ProviderOperationRecord)
                .where(
                    ProviderOperationRecord.idempotency_key.like(f"{idempotency_key}:%"),
                    ProviderOperationRecord.operation_kind != self._COMMAND_KIND,
                )
                .order_by(ProviderOperationRecord.created_at.asc())
            )
            rows = list((await self.session.execute(statement)).scalars().all())
            intents: list[ProviderOperationIntent] = []
            for row in rows:
                operation = _operation_from_payload(row.request_data.get("operation"))
                if operation is None:
                    return _coordinator_failure(
                        "find_intents", "Stored provider operation is invalid"
                    )
                result, error = _intent_payload(row)
                intents.append(
                    ProviderOperationIntent(
                        key=row.idempotency_key,
                        subscription_id=str(row.subscription_id),
                        operation=operation,
                        status=_intent_status(row.state),
                        result=result,
                        error=error,
                    )
                )
            return Success(tuple(intents))
        except SQLAlchemyError as exc:
            add_database_error_note(
                exc, table="provider_operation_records", operation="find_intents"
            )
            await self.session.rollback()
            return Failure(_coordinator_failure("find_intents", _safe_db_error(exc)).failure())

    @override
    async def pending_intents(
        self, *, limit: int = 500, lease_age_seconds: int = 300
    ) -> LifecycleCoordinatorResult[tuple[ProviderOperationIntent, ...]]:
        """List durable intents eligible for the crash-recovery sweep.

        A PENDING row with a recent ``started_at`` belongs to a live worker
        lease and is left alone.  An unstarted row is safe for the first
        attempt; an expired lease is deliberately returned for correlation
        read-only reconciliation, never for a second mutation.  Ambiguous
        rows are always eligible because their next operation is a read.
        """
        try:
            lease_age = timedelta(seconds=max(0, lease_age_seconds))
            cutoff = datetime.now(tz=UTC) - lease_age
            statement = (
                select(ProviderOperationRecord)
                .where(
                    ProviderOperationRecord.operation_kind != self._COMMAND_KIND,
                    or_(
                        ProviderOperationRecord.state.in_(
                            (
                                ProviderOperationState.UNKNOWN.value,
                                ProviderOperationState.RECONCILIATION.value,
                            )
                        ),
                        (ProviderOperationRecord.state == ProviderOperationState.PENDING.value)
                        & or_(
                            ProviderOperationRecord.started_at.is_(None),
                            ProviderOperationRecord.started_at <= cutoff,
                        ),
                    ),
                )
                .order_by(ProviderOperationRecord.updated_at.asc())
                .limit(limit)
            )
            rows = list((await self.session.execute(statement)).scalars().all())
            intents: list[ProviderOperationIntent] = []
            for row in rows:
                operation = _operation_from_payload(row.request_data.get("operation"))
                if operation is None:
                    return _coordinator_failure(
                        "pending_intents", "Stored provider operation is invalid"
                    )
                result, error = _intent_payload(row)
                intents.append(
                    ProviderOperationIntent(
                        key=row.idempotency_key,
                        subscription_id=str(row.subscription_id) if row.subscription_id else None,
                        operation=operation,
                        status=IntentStatus.PENDING,
                        result=result,
                        error=error,
                    )
                )
            return Success(tuple(intents))
        except SQLAlchemyError as exc:
            add_database_error_note(
                exc, table="provider_operation_records", operation="pending_intents"
            )
            await self.session.rollback()
            return Failure(_coordinator_failure("pending_intents", _safe_db_error(exc)).failure())

    @override
    async def find_plan(
        self, idempotency_key: str
    ) -> LifecycleCoordinatorResult[TransitionPlan | None]:
        try:
            statement = (
                select(ProviderOperationRecord)
                .where(
                    ProviderOperationRecord.idempotency_key.like(f"{idempotency_key}:%"),
                    ProviderOperationRecord.operation_kind != self._COMMAND_KIND,
                )
                .order_by(ProviderOperationRecord.created_at.asc())
                .limit(1)
            )
            row = (await self.session.execute(statement)).scalar_one_or_none()
            if row is None:
                return Success(None)
            plan = _plan_from_payload(row.request_data.get("plan"))
            return Success(plan)
        except SQLAlchemyError as exc:
            add_database_error_note(exc, table="provider_operation_records", operation="find_plan")
            await self.session.rollback()
            return Failure(_coordinator_failure("find_plan", _safe_db_error(exc)).failure())

    @override
    async def stage(  # noqa: PLR0912 -- transaction and idempotency branches are explicit
        self,
        record: LifecycleRecord,
        after: SubscriptionSnapshot,
        *,
        idempotency_key: str,
        expected_version: int,
        intents: tuple[ProviderOperationIntent, ...],
        audit_effects: tuple[TransitionEffect, ...],
        plan: TransitionPlan,
    ) -> LifecycleCoordinatorResult[LifecycleRecord]:
        """Atomically commit local desired state and provider intents."""

        try:
            current = await self._subscription(record.snapshot.id)
            if current is None:
                return _coordinator_failure("stage", "Subscription disappeared")
            if current.version != expected_version:
                return Failure(
                    LifecycleCoordinatorVersionConflictError(
                        message="Subscription changed while lifecycle transition was staged",
                        source="subscription_lifecycle_persistence",
                        subscription_id=record.snapshot.id,
                        expected_version=expected_version,
                    )
                )
            values = _snapshot_values(after)
            metadata = dict(current.metadata_ or {})
            if after.pending_upgrade is None:
                metadata.pop("pending_upgrade", None)
            else:
                metadata["pending_upgrade"] = _pending_upgrade_payload(after.pending_upgrade)
            values["metadata_"] = metadata
            updated = (
                await self.session.execute(
                    update(Subscription)
                    .where(
                        Subscription.id == current.id,
                        Subscription.version == expected_version,
                    )
                    .values(**values, version=Subscription.version + 1, updated_at=_now())
                    .returning(Subscription)
                )
            ).scalar_one_or_none()
            if updated is None:
                return Failure(
                    LifecycleCoordinatorVersionConflictError(
                        message="Subscription changed while lifecycle transition was staged",
                        source="subscription_lifecycle_persistence",
                        subscription_id=record.snapshot.id,
                        expected_version=expected_version,
                    )
                )
            audit_payload = [_effect_payload(effect) for effect in audit_effects]
            # Audit rows are part of the same transaction as the optimistic
            # snapshot and operation ledger.  Keep this write direct rather
            # than going through AuditLogRepository, whose error path rolls
            # back the whole session and would violate the savepoint rule.
            audit_rows = audit_effects or (None,)
            for effect in audit_rows:
                if effect is None:
                    action = "subscription.lifecycle.staged"
                    changes = {"command_idempotency_key": idempotency_key}
                else:
                    action = f"subscription.lifecycle.{effect.operation}"[:64]
                    changes = _effect_payload(effect)
                self.session.add(
                    AuditLog(
                        entity_type="subscription",
                        entity_id=str(updated.id),
                        action=action,
                        user_id=updated.user_id,
                        changes=changes,
                        metadata_={
                            "lifecycle_version": after.version,
                            "idempotency_key": idempotency_key,
                        },
                    )
                )
            await self.session.flush()
            for intent in intents:
                operation = ProviderOperationRecord(
                    subscription_id=updated.id,
                    correlation_id=intent.operation.correlation_id,
                    idempotency_key=intent.key,
                    operation_kind=intent.operation.kind.value,
                    request_data={
                        "operation": _operation_payload(intent.operation),
                        "audit_effects": audit_payload,
                        "command_idempotency_key": idempotency_key,
                        "plan": _plan_payload(plan),
                    },
                )
                # Use a savepoint for the idempotency race.  A duplicate key
                # means another worker already durably staged this intent; it
                # must not roll back the newly staged local snapshot.
                try:
                    async with self.session.begin_nested():
                        self.session.add(operation)
                        await self.session.flush()
                except IntegrityError as exc:
                    add_database_error_note(
                        exc,
                        table="provider_operation_records",
                        operation="stage_intent_savepoint",
                    )
                    existing = await self._operation(intent.key)
                    if existing is not None:
                        continue
                    # This path means the uniqueness error was not the
                    # expected idempotency race.  Roll back before
                    # propagating it; the normal duplicate path keeps the
                    # outer staged transaction intact.
                    await self.session.rollback()
                    return Failure(
                        _coordinator_failure(
                            "stage", "Provider operation key duplicate could not be resolved"
                        ).failure()
                    )
            await self.session.commit()
            return Success(_lifecycle_record(updated, after))
        except IntegrityError as exc:
            add_database_error_note(exc, table="provider_operation_records", operation="stage")
            await self.session.rollback()
            return Failure(
                _coordinator_failure("stage", "Provider operation key already exists").failure()
            )
        except SQLAlchemyError as exc:
            add_database_error_note(exc, table="subscriptions", operation="stage")
            await self.session.rollback()
            return Failure(_coordinator_failure("stage", _safe_db_error(exc)).failure())

    @override
    async def intent(self, key: str) -> LifecycleCoordinatorResult[ProviderOperationIntent | None]:
        try:
            statement = select(ProviderOperationRecord).where(
                ProviderOperationRecord.idempotency_key == key,
                ProviderOperationRecord.operation_kind != self._COMMAND_KIND,
            )
            row = (await self.session.execute(statement)).scalar_one_or_none()
            if row is None:
                return Success(None)
            operation = _operation_from_payload(row.request_data.get("operation"))
            if operation is None:
                return _coordinator_failure("intent", "Stored provider operation is invalid")
            result, error = _intent_payload(row)
            return Success(
                ProviderOperationIntent(
                    key=row.idempotency_key,
                    subscription_id=str(row.subscription_id),
                    operation=operation,
                    status=_intent_status(row.state),
                    result=result,
                    error=error,
                )
            )
        except SQLAlchemyError as exc:
            add_database_error_note(exc, table="provider_operation_records", operation="intent")
            await self.session.rollback()
            return Failure(_coordinator_failure("intent", _safe_db_error(exc)).failure())

    @override
    async def claim_intent(self, key: str) -> LifecycleCoordinatorResult[bool]:
        """Claim a never-started pending intent with one compare-and-set.

        The old implementation incremented ``attempts`` from ``intent``
        itself.  Two readers could therefore both observe PENDING and both
        send the mutation.  A conditional UPDATE makes the lease acquisition
        the serialization point: only the row whose ``started_at`` is still
        NULL can become the first sender.  A process crash leaves that
        timestamp durable, making recovery correlation-read-only.
        """
        try:
            statement = (
                update(ProviderOperationRecord)
                .where(
                    ProviderOperationRecord.idempotency_key == key,
                    ProviderOperationRecord.operation_kind != self._COMMAND_KIND,
                    ProviderOperationRecord.state == ProviderOperationState.PENDING.value,
                    ProviderOperationRecord.started_at.is_(None),
                )
                .values(
                    attempts=ProviderOperationRecord.attempts + 1,
                    started_at=_now(),
                    last_attempt_at=_now(),
                    updated_at=_now(),
                )
            )
            result = await self.session.execute(statement)
            await self.session.commit()
            return Success(result.rowcount == 1)
        except SQLAlchemyError as exc:
            add_database_error_note(
                exc, table="provider_operation_records", operation="claim_intent"
            )
            await self.session.rollback()
            return Failure(_coordinator_failure("claim_intent", _safe_db_error(exc)).failure())

    @override
    async def claim_capture_payment(
        self,
        *,
        payment_link_key: str,
        subscription_id: str,
        payment_id: str,
    ) -> LifecycleCoordinatorResult[bool]:
        """CAS the first captured payment onto one succeeded payment link."""
        try:
            statement = (
                update(ProviderOperationRecord)
                .where(
                    ProviderOperationRecord.idempotency_key == payment_link_key,
                    ProviderOperationRecord.subscription_id == _uuid_value(subscription_id),
                    ProviderOperationRecord.operation_kind == "create_payment_link",
                    ProviderOperationRecord.state == ProviderOperationState.SUCCEEDED.value,
                    ProviderOperationRecord.observation_data["captured_payment_id"]
                    .as_string()
                    .is_(None),
                )
                .values(
                    observation_data=func.jsonb_set(
                        ProviderOperationRecord.observation_data,
                        "{captured_payment_id}",
                        func.to_jsonb(payment_id),
                        True,
                    ),
                    updated_at=_now(),
                )
            )
            result = await self.session.execute(statement)
            await self.session.commit()
            if result.rowcount == 1:
                return Success(True)

            current = await self._operation(payment_link_key)
            captured = (
                (current.observation_data or {}).get("captured_payment_id")
                if current is not None
                else None
            )
            valid_link = current is not None and (
                str(current.subscription_id) == subscription_id
                and current.operation_kind == "create_payment_link"
                and current.state == ProviderOperationState.SUCCEEDED.value
            )
            if valid_link and captured == payment_id:
                return Success(False)
            if valid_link and captured is not None:
                return Failure(
                    LifecycleCoordinatorValidationError(
                        message="Payment-link operation is already paired with another capture",
                        details={
                            "payment_link_key": payment_link_key,
                            "subscription_id": subscription_id,
                            "captured_payment_id": captured,
                            "conflicting_payment_id": payment_id,
                            "manual_review": True,
                        },
                        source="subscription_lifecycle_repository",
                    )
                )
            return Failure(
                LifecycleCoordinatorValidationError(
                    message="Payment-link operation is not a valid succeeded link for capture",
                    details={
                        "payment_link_key": payment_link_key,
                        "subscription_id": subscription_id,
                        "manual_review": True,
                    },
                    source="subscription_lifecycle_repository",
                )
            )
        except SQLAlchemyError as exc:
            add_database_error_note(
                exc, table="provider_operation_records", operation="claim_capture_payment"
            )
            await self.session.rollback()
            return Failure(
                _coordinator_failure("claim_capture_payment", _safe_db_error(exc)).failure()
            )

    @override
    async def record_intent(
        self, intent: ProviderOperationIntent
    ) -> LifecycleCoordinatorResult[None]:
        """Durably record coordinator-generated follow-up intents."""

        try:
            existing = await self._operation(intent.key)
            if existing is not None:
                return Success(None)
            self.session.add(
                ProviderOperationRecord(
                    subscription_id=_uuid_value(intent.subscription_id),
                    correlation_id=intent.operation.correlation_id,
                    idempotency_key=intent.key,
                    operation_kind=intent.operation.kind.value,
                    request_data={"operation": _operation_payload(intent.operation)},
                )
            )
            await self.session.commit()
            return Success(None)
        except IntegrityError as exc:
            add_database_error_note(
                exc, table="provider_operation_records", operation="record_intent_duplicate"
            )
            await self.session.rollback()
            # A duplicate is safe: the durable key wins over a second write.
            if await self._operation(intent.key) is not None:
                return Success(None)
            return Failure(
                _coordinator_failure(
                    "record_intent", "Provider operation duplicate could not be resolved"
                ).failure()
            )
        except SQLAlchemyError as exc:
            add_database_error_note(
                exc, table="provider_operation_records", operation="record_intent"
            )
            await self.session.rollback()
            return Failure(_coordinator_failure("record_intent", _safe_db_error(exc)).failure())

    @override
    async def provider_outcome(
        self, intent: ProviderOperationIntent
    ) -> LifecycleCoordinatorResult[None]:
        """Persist the provider result and integration bindings before continuing."""

        try:
            row = await self._operation(intent.key)
            if row is None:
                return _coordinator_failure("provider_outcome", "Provider intent was not recorded")
            state = _provider_state(intent.status)
            current_state = _operation_state(row.state)
            if current_state is ProviderOperationState.SUCCEEDED and state is not current_state:
                return Success(None)
            if current_state is ProviderOperationState.MANUAL_REVIEW and state in {
                ProviderOperationState.PENDING,
                ProviderOperationState.UNKNOWN,
            }:
                return Success(None)
            if (
                current_state
                in {
                    ProviderOperationState.UNKNOWN,
                    ProviderOperationState.RECONCILIATION,
                }
                and state is ProviderOperationState.PENDING
            ):
                return Success(None)
            result_payload = _provider_success_payload(intent.result)
            error_payload = _provider_error_payload(intent.error)
            values: dict[str, object] = {
                "state": state.value,
                "observation_data": {
                    "result": result_payload,
                    "error": error_payload,
                },
                "updated_at": _now(),
            }
            if intent.result is not None:
                resource_id = _provider_resource_id(intent.result.value)
                if resource_id is not None:
                    values["provider_resource_id"] = resource_id
            if intent.error is not None:
                values["error_code"] = _sanitize_provider_error(intent.error.code.value, limit=64)
                values["error_message"] = _sanitize_provider_error(intent.error.message, limit=1000)
            if state in {
                ProviderOperationState.SUCCEEDED,
                ProviderOperationState.FAILED,
                ProviderOperationState.MANUAL_REVIEW,
            }:
                values["completed_at"] = _now()
            await self.session.execute(
                update(ProviderOperationRecord)
                .where(ProviderOperationRecord.id == row.id)
                .values(**values)
            )
            if intent.result is not None:
                integration_values = _integration_values(intent.result.value)
                if integration_values:
                    await self.session.execute(
                        update(Subscription)
                        .where(Subscription.id == _uuid_value(intent.subscription_id))
                        .values(**integration_values, updated_at=_now())
                    )
            await self.session.commit()
            return Success(None)
        except SQLAlchemyError as exc:
            add_database_error_note(
                exc, table="provider_operation_records", operation="provider_outcome"
            )
            await self.session.rollback()
            return Failure(_coordinator_failure("provider_outcome", _safe_db_error(exc)).failure())

    @override
    async def mark_sync_failure(
        self,
        subscription_id: str,
        *,
        expected_version: int,
        retryable: bool,
    ) -> LifecycleCoordinatorResult[None]:
        state = (
            RemoteSyncState.RETRYABLE_FAILURE if retryable else RemoteSyncState.PERMANENT_FAILURE
        )
        try:
            result = await self.session.execute(
                update(Subscription)
                .where(
                    Subscription.id == _uuid_value(subscription_id),
                    Subscription.version == expected_version,
                )
                .values(remote_sync=state.value, updated_at=_now())
            )
            if result.rowcount != 1:
                return Failure(
                    LifecycleCoordinatorVersionConflictError(
                        message="Subscription changed while recording provider failure",
                        source="subscription_lifecycle_persistence",
                        subscription_id=subscription_id,
                        expected_version=expected_version,
                    )
                )
            await self.session.commit()
            return Success(None)
        except SQLAlchemyError as exc:
            add_database_error_note(exc, table="subscriptions", operation="mark_sync_failure")
            await self.session.rollback()
            return Failure(_coordinator_failure("mark_sync_failure", _safe_db_error(exc)).failure())

    @override
    async def mark_observed_plan(
        self,
        subscription_id: str,
        *,
        expected_version: int,
        observed_plan_id: str,
    ) -> LifecycleCoordinatorResult[None]:
        """Persist a verified provider plan observation without a new lifecycle version."""
        try:
            row = await self._subscription(subscription_id)
            if row is None:
                return _coordinator_failure("mark_observed_plan", "Subscription was not found")
            if row.version != expected_version:
                return Failure(
                    LifecycleCoordinatorVersionConflictError(
                        message="Subscription changed while recording observed plan",
                        source="subscription_lifecycle_persistence",
                        subscription_id=subscription_id,
                        expected_version=expected_version,
                    )
                )
            metadata = dict(row.metadata_ or {})
            metadata.pop("pending_upgrade", None)
            await self.session.execute(
                update(Subscription)
                .where(Subscription.id == row.id, Subscription.version == expected_version)
                .values(
                    observed_plan_id=_uuid_value(observed_plan_id),
                    remote_sync=RemoteSyncState.SYNCED.value,
                    metadata_=metadata,
                    updated_at=_now(),
                )
            )
            await self.session.commit()
            return Success(None)
        except SQLAlchemyError as exc:
            add_database_error_note(exc, table="subscriptions", operation="mark_observed_plan")
            await self.session.rollback()
            return Failure(
                _coordinator_failure("mark_observed_plan", _safe_db_error(exc)).failure()
            )

    @override
    async def finish(  # noqa: PLR0912,PLR0914 -- terminal persistence has explicit conflict/error branches
        self,
        subscription_id: str,
        *,
        expected_version: int,
        idempotency_key: str,
        provider_results: tuple[ProviderSuccess[Any], ...],
        reconciled: bool,
        plan: TransitionPlan,
        sync_provider: bool = True,
    ) -> LifecycleCoordinatorResult[LifecycleOutcome]:
        try:
            row = await self._subscription(subscription_id)
            if row is None:
                return _coordinator_failure("finish", "Subscription disappeared")
            if row.version != expected_version:
                return Failure(
                    LifecycleCoordinatorVersionConflictError(
                        message="Subscription changed while lifecycle transition was finalized",
                        source="subscription_lifecycle_persistence",
                        subscription_id=subscription_id,
                        expected_version=expected_version,
                    )
                )
            provider_plan_external = _provider_plan_id(provider_results) if reconciled else None
            provider_plan = _local_observed_plan(row, provider_plan_external)
            observed_plan = str(row.observed_plan_id) if row.observed_plan_id is not None else None
            desired_plan = str(row.plan_id)
            desired_provider_plan = _desired_provider_plan(row)
            candidate_observed = provider_plan_external
            if candidate_observed is None and observed_plan == desired_plan:
                candidate_observed = desired_provider_plan
            values: dict[str, object] = {"updated_at": _now()}
            # A mutation acknowledgement proves that the requested operation
            # was accepted; it is not an authoritative plan observation. Only
            # a reconciliation/read may advance observed_plan_id, and the
            # local desired/observed comparison must agree before SYNCED.
            if reconciled and provider_plan is not None:
                parsed_plan = _uuid_value(provider_plan)
                if isinstance(parsed_plan, UUID):
                    values["observed_plan_id"] = parsed_plan
                    candidate_observed = provider_plan
            verified = _verified_convergence(
                plan,
                provider_results,
                desired_provider_plan=desired_provider_plan,
                provider_plan_external=provider_plan_external,
                reconciled=reconciled,
            )
            if sync_provider and verified:
                values["remote_sync"] = RemoteSyncState.SYNCED.value
            elif sync_provider:
                values["remote_sync"] = RemoteSyncState.PENDING.value
            if values.keys() != {"updated_at"}:
                await self.session.execute(
                    update(Subscription)
                    .where(Subscription.id == row.id, Subscription.version == expected_version)
                    .values(**values)
                )
            final_row = await self._subscription(subscription_id)
            if final_row is None:
                return _coordinator_failure("finish", "Subscription disappeared after finalization")
            snapshot_result = _to_lifecycle_snapshot(final_row)
            if isinstance(snapshot_result, Failure):
                return _coordinator_failure("finish", snapshot_result.failure().message)
            final_snapshot = snapshot_result.unwrap()
            if reconciled and provider_plan is not None:
                final_snapshot = replace(final_snapshot, observed_plan_id=provider_plan)
            outcome = LifecycleOutcome(
                snapshot=final_snapshot,
                plan=plan,
                status=CoordinatorStatus.RECONCILED if reconciled else CoordinatorStatus.APPLIED,
                provider_results=provider_results,
            )
            command = ProviderOperationRecord(
                subscription_id=final_row.id,
                correlation_id=idempotency_key,
                idempotency_key=idempotency_key,
                operation_kind=self._COMMAND_KIND,
                state=ProviderOperationState.SUCCEEDED.value,
                observation_data={"outcome": _outcome_payload(outcome)},
                completed_at=_now(),
            )
            self.session.add(command)
            await self.session.commit()
            return Success(outcome)
        except IntegrityError as exc:
            add_database_error_note(exc, table="provider_operation_records", operation="finish")
            await self.session.rollback()
            existing = await self.find_command(idempotency_key)
            if isinstance(existing, Success) and existing.unwrap() is not None:
                return existing
            return Failure(
                _coordinator_failure("finish", "Lifecycle command key already exists").failure()
            )
        except SQLAlchemyError as exc:
            add_database_error_note(exc, table="subscriptions", operation="finish")
            await self.session.rollback()
            return Failure(_coordinator_failure("finish", _safe_db_error(exc)).failure())

    async def _subscription(self, subscription_id: str) -> Subscription | None:
        statement = select(Subscription).where(
            Subscription.id == _uuid_value(subscription_id),
            Subscription.deleted_at.is_(None),
        )
        return (await self.session.execute(statement)).scalar_one_or_none()

    async def _operation(self, idempotency_key: str) -> ProviderOperationRecord | None:
        statement = select(ProviderOperationRecord).where(
            ProviderOperationRecord.idempotency_key == idempotency_key
        )
        return (await self.session.execute(statement)).scalar_one_or_none()


def _coordinator_failure(operation: str, message: str) -> Failure[Any]:
    return Failure(
        LifecycleCoordinatorPersistenceError(
            message=message,
            details={"operation": operation},
            source="subscription_lifecycle_persistence",
            operation=operation,
        )
    )


def _safe_db_error(error: BaseException) -> str:
    return _sanitize_provider_error(str(error), limit=1000)


def _now() -> datetime:
    return datetime.now(tz=UTC)


def _uuid_value(value: str | UUID | None) -> UUID | str | None:
    if value is None or isinstance(value, UUID):
        return value
    try:
        return UUID(value)
    except (ValueError, AttributeError):
        return value


def _lifecycle_record(
    subscription: Subscription, snapshot: SubscriptionSnapshot
) -> LifecycleRecord:
    metadata = subscription.metadata_ or {}
    plan_bindings = metadata.get("billing_provider_plan_ids")
    desired_provider_plan_id = None
    if isinstance(plan_bindings, dict):
        candidate = plan_bindings.get(snapshot.plan_id)
        if isinstance(candidate, str) and candidate:
            desired_provider_plan_id = candidate
    return LifecycleRecord(
        snapshot=snapshot,
        customer_email=_metadata_text(metadata, "billing_customer_email"),
        customer_contact=_metadata_text(metadata, "billing_customer_phone"),
        customer_name=_metadata_text(metadata, "billing_customer_name"),
        provider_customer_id=subscription.razorpay_customer_id,
        provider_subscription_id=subscription.razorpay_subscription_id,
        # ``provider_plan_id`` is never reconstructed from the local UUID.
        # Desired provider bindings are carried explicitly in metadata below.
        provider_plan_id=None,
        desired_provider_plan_id=desired_provider_plan_id,
    )


def _metadata_text(metadata: dict[str, object], key: str) -> str | None:
    value = metadata.get(key)
    return value if isinstance(value, str) and value else None


def _effect_payload(effect: TransitionEffect) -> dict[str, object]:
    return {
        "kind": effect.kind.value,
        "operation": effect.operation,
        "idempotency_key": effect.idempotency_key,
        "values": _json_value(dict(effect.values)),
    }


def _operation_payload(operation: ProviderOperation) -> dict[str, object]:
    return {
        "kind": operation.kind.value,
        "fields": _json_value(asdict(operation)),
    }


def _operation_from_payload(payload: object) -> ProviderOperation | None:
    if not isinstance(payload, dict):
        return None
    kind_value = payload.get("kind")
    decoded_fields = _decode(payload.get("fields", {}))
    fields = cast("dict[str, Any]", decoded_fields)
    if not isinstance(kind_value, str) or not isinstance(fields, dict):
        return None
    operation_types = (
        CreateCustomerOperation,
        FindCustomerByEmailOperation,
        CreateSubscriptionOperation,
        CancelSubscriptionOperation,
        PauseSubscriptionOperation,
        ResumeSubscriptionOperation,
        ChangeSubscriptionPlanOperation,
        ObserveSubscriptionOperation,
        FindSubscriptionByCorrelationOperation,
        CreatePaymentLinkOperation,
        FindPaymentLinkByCorrelationOperation,
        CreatePlanOperation,
        FindPlanByCorrelationOperation,
        CreateRefundOperation,
        FindRefundByCorrelationOperation,
    )
    constructors: dict[str, type[ProviderOperation]] = {
        operation_type.kind.value: operation_type for operation_type in operation_types
    }
    constructor = constructors.get(kind_value)
    if constructor is None:
        return None
    try:
        return constructor(**fields)
    except (TypeError, ValueError):
        return None


def _json_value(value: object) -> object:
    if isinstance(value, datetime):
        return {"__datetime__": value.isoformat()}
    if isinstance(value, UUID):
        return {"__uuid__": str(value)}
    if isinstance(value, StrEnum):
        return value.value
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return value


def _decode(value: object) -> object:
    if isinstance(value, list):
        return [_decode(item) for item in value]
    if isinstance(value, dict):
        values = cast("dict[str, object]", value)
        if set(values) == {"__datetime__"} and isinstance(values["__datetime__"], str):
            try:
                return datetime.fromisoformat(values["__datetime__"])
            except ValueError:
                return None
        if set(values) == {"__uuid__"} and isinstance(values["__uuid__"], str):
            try:
                return UUID(values["__uuid__"])
            except ValueError:
                return None
        return {key: _decode(item) for key, item in values.items()}
    return value


def _intent_status(value: str) -> IntentStatus:
    mapping = {
        ProviderOperationState.PENDING.value: IntentStatus.PENDING,
        ProviderOperationState.SUCCEEDED.value: IntentStatus.SUCCEEDED,
        ProviderOperationState.UNKNOWN.value: IntentStatus.UNKNOWN,
        ProviderOperationState.RECONCILIATION.value: IntentStatus.UNKNOWN,
        ProviderOperationState.MANUAL_REVIEW.value: IntentStatus.MANUAL_REVIEW,
        ProviderOperationState.FAILED.value: IntentStatus.FAILED,
    }
    return mapping.get(value, IntentStatus.PENDING)


def _provider_state(status: IntentStatus) -> ProviderOperationState:
    return {
        IntentStatus.PENDING: ProviderOperationState.PENDING,
        IntentStatus.SUCCEEDED: ProviderOperationState.SUCCEEDED,
        IntentStatus.TRANSIENT_FAILURE: ProviderOperationState.PENDING,
        IntentStatus.UNKNOWN: ProviderOperationState.UNKNOWN,
        IntentStatus.RECONCILIATION: ProviderOperationState.RECONCILIATION,
        IntentStatus.MANUAL_REVIEW: ProviderOperationState.MANUAL_REVIEW,
        IntentStatus.FAILED: ProviderOperationState.FAILED,
    }[status]


def _provider_resource_id(value: object) -> str | None:
    for field_name in (
        "provider_subscription_id",
        "provider_customer_id",
        "provider_payment_link_id",
        "provider_plan_id",
        "provider_refund_id",
    ):
        field = getattr(value, field_name, None)
        if isinstance(field, str) and field:
            return field
    return None


def _integration_values(value: object) -> dict[str, object]:
    if isinstance(value, ProviderCustomer):
        return {"razorpay_customer_id": value.provider_customer_id}
    if isinstance(value, ProviderSubscriptionState):
        values: dict[str, object] = {
            "razorpay_subscription_id": value.provider_subscription_id,
            "current_period_start": value.current_period_start,
            "current_period_end": value.current_period_end,
        }
        # Provider mutation acknowledgements are not observations.  A webhook
        # or explicit ObserveSubscription reconciliation owns this field.
        if value.ended_at is not None:
            values["ended_at"] = value.ended_at
        if value.cancel_at_cycle_end is not None:
            values["cancel_at_period_end"] = value.cancel_at_cycle_end
        return values
    return {}


def _provider_plan_id(results: tuple[ProviderSuccess[Any], ...]) -> str | None:
    for result in reversed(results):
        if isinstance(result.value, ProviderSubscriptionState) and result.value.plan_id:
            return result.value.plan_id
    return None


def _desired_provider_plan(subscription: Subscription) -> str | None:
    metadata = subscription.metadata_ or {}
    bindings = metadata.get("billing_provider_plan_ids")
    if isinstance(bindings, dict):
        value = bindings.get(str(subscription.plan_id))
        if isinstance(value, str) and value:
            return value
    # Historical rows may not carry a provider binding.  Treat that as an
    # unresolved integration fact; never send a local UUID to Razorpay and
    # never infer convergence from it.
    return None


def _local_observed_plan(subscription: Subscription, provider_plan: str | None) -> str | None:
    if provider_plan is None:
        return None
    metadata = subscription.metadata_ or {}
    bindings = metadata.get("billing_provider_plan_ids")
    if isinstance(bindings, dict):
        for local_id, provider_id in bindings.items():
            if provider_id == provider_plan and isinstance(local_id, str):
                return local_id
    return None


def _verified_convergence(
    plan: TransitionPlan,
    results: tuple[ProviderSuccess[Any], ...],
    *,
    desired_provider_plan: str | None,
    provider_plan_external: str | None,
    reconciled: bool,
) -> bool:
    """Return true only for a read-backed, operation-specific observation."""
    if not reconciled:
        return False
    state = next(
        (
            result.value
            for result in reversed(results)
            if isinstance(result.value, ProviderSubscriptionState)
        ),
        None,
    )
    if state is None:
        return False
    if desired_provider_plan is None or provider_plan_external != desired_provider_plan:
        return False
    operations = {effect.operation for effect in plan.effects}
    if "cancel_remote" in operations:
        cancel_at_end = next(
            (
                bool(dict(effect.values).get("cancel_at_period_end"))
                for effect in plan.effects
                if effect.operation == "cancel_remote"
            ),
            False,
        )
        if cancel_at_end:
            return state.cancel_at_cycle_end is True
        return state.ended_at is not None or state.status in {"cancelled", "completed", "expired"}
    if "pause_remote" in operations:
        return state.status == "paused"
    if "resume_remote" in operations:
        return state.status == "active"
    if plan.after.state.kind in {SubscriptionStateKind.CANCELLED, SubscriptionStateKind.EXPIRED}:
        return state.ended_at is not None or state.status in {"cancelled", "completed", "expired"}
    if plan.after.state.kind is SubscriptionStateKind.PAUSED:
        return state.status == "paused"
    if plan.after.state.kind is SubscriptionStateKind.ACTIVE:
        return state.status == "active"
    return state.status in {
        "created",
        "authenticated",
        "active",
        "paused",
        "cancelled",
        "completed",
        "expired",
    }


def _provider_success_payload(result: ProviderSuccess[Any] | None) -> object:
    if result is None:
        return None
    return {
        "operation": result.operation.value,
        "correlation_id": result.correlation_id,
        "value": {
            "type": type(result.value).__name__,
            "fields": _json_value(asdict(result.value)),
        },
    }


def _provider_success_from_payload(payload: object) -> ProviderSuccess[Any] | None:
    if not isinstance(payload, dict):
        return None
    payload = cast("dict[str, Any]", payload)
    operation = payload.get("operation")
    correlation_id = payload.get("correlation_id")
    value_payload = payload.get("value")
    if not isinstance(operation, str) or not isinstance(correlation_id, str):
        return None
    if not isinstance(value_payload, dict):
        return None
    fields = cast("dict[str, Any]", _decode(value_payload.get("fields", {})))
    value_type = value_payload.get("type")
    value_types: dict[str, type[Any]] = {
        "ProviderSubscriptionState": ProviderSubscriptionState,
        "ProviderCustomer": ProviderCustomer,
        "ProviderPaymentLink": ProviderPaymentLink,
        "ProviderPlan": ProviderPlan,
        "ProviderReceipt": ProviderReceipt,
    }
    constructor = value_types.get(str(value_type))
    if constructor is None or not isinstance(fields, dict):
        return None
    try:
        return ProviderSuccess(
            ProviderOperationKind(operation), correlation_id, constructor(**fields)
        )
    except (TypeError, ValueError):
        return None


def _provider_error_payload(error: ProviderError | None) -> object:
    if error is None:
        return None
    return {
        "code": error.code.value,
        "message": _sanitize_provider_error(error.message, limit=1000),
        "operation": error.operation.value if error.operation is not None else None,
        "correlation_id": error.correlation_id,
    }


def _provider_error_from_payload(payload: object) -> ProviderError | None:
    if not isinstance(payload, dict):
        return None
    payload = cast("dict[str, Any]", payload)
    code = payload.get("code")
    operation = payload.get("operation")
    correlation_id = payload.get("correlation_id")
    message = payload.get("message")
    if not isinstance(code, str):
        return None
    try:
        operation_value = ProviderOperationKind(operation) if isinstance(operation, str) else None
        kwargs: dict[str, object] = {
            "operation": operation_value,
            "correlation_id": correlation_id if isinstance(correlation_id, str) else None,
        }
        if isinstance(message, str):
            kwargs["message"] = message
        error_types: dict[ProviderErrorCode, type[ProviderError]] = {
            ProviderErrorCode.INVALID_REQUEST: ProviderInvalidRequestError,
            ProviderErrorCode.AUTHENTICATION: ProviderAuthenticationError,
            ProviderErrorCode.RATE_LIMITED: ProviderRateLimitedError,
            ProviderErrorCode.UNAVAILABLE: ProviderUnavailableError,
            ProviderErrorCode.NOT_FOUND: ProviderNotFoundError,
            ProviderErrorCode.OUTCOME_UNKNOWN: ProviderOutcomeUnknownError,
            ProviderErrorCode.MANUAL_REVIEW_REQUIRED: ProviderManualReviewRequiredError,
            ProviderErrorCode.RECONCILIATION_REQUIRED: ProviderReconciliationRequiredError,
            ProviderErrorCode.UNEXPECTED: ProviderUnexpectedError,
        }
        error_type = error_types.get(ProviderErrorCode(code))
        return error_type(**kwargs) if error_type is not None else None
    except (TypeError, ValueError):
        return None


def _intent_payload(
    row: ProviderOperationRecord,
) -> tuple[ProviderSuccess[Any] | None, ProviderError | None]:
    observation = row.observation_data or {}
    return (
        _provider_success_from_payload(observation.get("result")),
        _provider_error_from_payload(observation.get("error")),
    )


def _snapshot_payload(snapshot: SubscriptionSnapshot) -> dict[str, object]:
    return {
        "id": snapshot.id,
        "user_id": snapshot.user_id,
        "plan_id": snapshot.plan_id,
        "state": _json_value(asdict(snapshot.state)),
        "version": snapshot.version,
        "remote_sync": snapshot.remote_sync.value,
        "cancellation_intent": (
            {
                "requested_at": _json_value(snapshot.cancellation_intent.requested_at),
                "at_period_end": snapshot.cancellation_intent.at_period_end,
            }
            if snapshot.cancellation_intent is not None
            else None
        ),
        "observed_plan_id": snapshot.observed_plan_id,
        "pending_upgrade": (
            _pending_upgrade_payload(snapshot.pending_upgrade)
            if snapshot.pending_upgrade is not None
            else None
        ),
    }


def _snapshot_from_payload(payload: object) -> SubscriptionSnapshot | None:
    if not isinstance(payload, dict):
        return None
    payload = cast("dict[str, Any]", payload)
    state_payload = cast("dict[str, Any]", _decode(payload.get("state")))
    if not isinstance(state_payload, dict):
        return None
    kind = state_payload.get("kind")
    if not isinstance(kind, str):
        return None
    state_types: dict[str, type[Any]] = {
        SubscriptionStateKind.CREATED.value: CreatedState,
        SubscriptionStateKind.AUTHENTICATED.value: AuthenticatedState,
        SubscriptionStateKind.ACTIVE.value: ActiveState,
        SubscriptionStateKind.PAST_DUE.value: PastDueState,
        SubscriptionStateKind.HALTED.value: HaltedState,
        SubscriptionStateKind.CANCELLED.value: CancelledState,
        SubscriptionStateKind.PAUSED.value: PausedState,
        SubscriptionStateKind.EXPIRED.value: ExpiredState,
    }
    constructor = state_types.get(kind)
    if constructor is None:
        return None
    state_payload.pop("kind", None)
    try:
        state = constructor(**state_payload)
        intent_payload = payload.get("cancellation_intent")
        intent = None
        if isinstance(intent_payload, dict):
            requested_at = _decode(intent_payload.get("requested_at"))
            if not isinstance(requested_at, datetime):
                return None
            intent = CancellationIntent(
                requested_at=requested_at,
                at_period_end=bool(intent_payload.get("at_period_end", False)),
            )
        return SubscriptionSnapshot(
            id=str(payload["id"]),
            user_id=str(payload["user_id"]),
            plan_id=str(payload["plan_id"]),
            state=state,
            version=int(payload["version"]),
            remote_sync=RemoteSyncState(str(payload.get("remote_sync", "synced"))),
            cancellation_intent=intent,
            observed_plan_id=(
                str(payload["observed_plan_id"])
                if payload.get("observed_plan_id") is not None
                else None
            ),
            pending_upgrade=_pending_upgrade_from_payload(payload.get("pending_upgrade")),
        )
    except (KeyError, TypeError, ValueError):
        return None


def _plan_payload(plan: TransitionPlan) -> dict[str, object]:
    return {
        "outcome": plan.outcome.value,
        "before": _snapshot_payload(plan.before),
        "after": _snapshot_payload(plan.after),
        "expected_version": plan.expected_version,
        "effects": [_effect_payload(effect) for effect in plan.effects],
        "reason": plan.reason,
    }


def _plan_from_payload(payload: object) -> TransitionPlan | None:
    if not isinstance(payload, dict):
        return None
    payload = cast("dict[str, Any]", payload)
    before = _snapshot_from_payload(payload.get("before"))
    after = _snapshot_from_payload(payload.get("after"))
    if before is None or after is None:
        return None
    effects: list[TransitionEffect] = []
    raw_effects = cast("list[dict[str, Any]]", payload.get("effects", []))
    if not isinstance(raw_effects, list):
        return None
    try:
        for raw in raw_effects:
            values = _decode(raw.get("values", {}))
            effect_values = cast("dict[str, Any]", values)
            effects.append(
                TransitionEffect(
                    kind=EffectKind(str(raw["kind"])),
                    operation=str(raw["operation"]),
                    idempotency_key=str(raw["idempotency_key"]),
                    values=tuple((str(key), value) for key, value in effect_values.items()),
                )
            )
        return TransitionPlan(
            outcome=TransitionOutcome(str(payload["outcome"])),
            before=before,
            after=after,
            expected_version=int(payload["expected_version"]),
            effects=tuple(effects),
            reason=payload.get("reason") if isinstance(payload.get("reason"), str) else None,
        )
    except (KeyError, TypeError, ValueError):
        return None


def _outcome_payload(outcome: LifecycleOutcome) -> dict[str, object]:
    return {
        "snapshot": _snapshot_payload(outcome.snapshot),
        "plan": _plan_payload(outcome.plan),
        "status": outcome.status.value,
        "provider_results": [
            _provider_success_payload(result) for result in outcome.provider_results
        ],
    }


def _outcome_from_payload(payload: object) -> LifecycleOutcome | None:
    if not isinstance(payload, dict):
        return None
    payload = cast("dict[str, Any]", payload)
    snapshot = _snapshot_from_payload(payload.get("snapshot"))
    plan = _plan_from_payload(payload.get("plan"))
    if snapshot is None or plan is None:
        return None
    raw_results = payload.get("provider_results", [])
    if not isinstance(raw_results, list):
        return None
    results = tuple(
        result
        for result_payload in raw_results
        if (result := _provider_success_from_payload(result_payload)) is not None
    )
    try:
        return LifecycleOutcome(
            snapshot=snapshot,
            plan=plan,
            status=CoordinatorStatus(str(payload["status"])),
            provider_results=results,
        )
    except (KeyError, TypeError, ValueError):
        return None


def _operation_state(value: ProviderOperationState | str) -> ProviderOperationState | None:
    try:
        return ProviderOperationState(str(value).replace("-", "_"))
    except ValueError:
        return None


def _sanitize_provider_error(value: str, *, limit: int) -> str:
    """Keep durable provider errors safe for logs, operators, and payloads."""

    return " ".join(value.split())[:limit]


def _subscription_ref(snapshot: SubscriptionSnapshot) -> Subscription:
    """Build a lightweight ORM identity for an atomic versioned UPDATE."""

    return Subscription(id=snapshot.id, plan_id=snapshot.plan_id, user_id=snapshot.user_id)


def _snapshot_values(snapshot: SubscriptionSnapshot) -> dict[str, object]:
    intent = snapshot.cancellation_intent
    period_start = getattr(snapshot.state, "period_start", None)
    period_end = getattr(snapshot.state, "period_end", None)
    cancelled_at = snapshot.state.at if isinstance(snapshot.state, CancelledState) else None
    at_period_end = intent.at_period_end if intent is not None else False
    entitlement_until = period_end if at_period_end else cancelled_at
    pause_start = snapshot.state.start if isinstance(snapshot.state, PausedState) else None
    pause_end = snapshot.state.end if isinstance(snapshot.state, PausedState) else None
    retry_count = snapshot.state.retry_count if isinstance(snapshot.state, PastDueState) else 0
    return {
        "plan_id": snapshot.plan_id,
        "observed_plan_id": snapshot.observed_plan_id,
        "status": snapshot.state.kind.value,
        "current_period_start": period_start,
        "current_period_end": period_end,
        "pause_start": pause_start,
        "pause_end": pause_end,
        "retry_count": retry_count,
        "cancel_at_period_end": at_period_end,
        "cancellation_requested_at": intent.requested_at if intent is not None else None,
        "entitlement_until": entitlement_until,
        "cancelled_at": cancelled_at,
        "ended_at": cancelled_at,
        "remote_sync": snapshot.remote_sync.value,
    }


def _pending_upgrade_payload(pending: PendingUpgrade) -> dict[str, object]:
    return {
        "payment_link_key": pending.payment_link_key,
        "target_plan_id": pending.target_plan_id,
        "target_provider_plan_id": pending.target_provider_plan_id,
        "amount_paisa": pending.amount_paisa,
        "currency": pending.currency,
    }


def _pending_upgrade_from_payload(value: object) -> PendingUpgrade | None:
    if not isinstance(value, dict):
        return None
    payment_link_key = value.get("payment_link_key")
    target_plan_id = value.get("target_plan_id")
    target_provider_plan_id = value.get("target_provider_plan_id")
    amount_paisa = value.get("amount_paisa")
    currency = value.get("currency")
    if (
        not all(
            isinstance(item, str) and item
            for item in (payment_link_key, target_plan_id, target_provider_plan_id, currency)
        )
        or isinstance(amount_paisa, bool)
        or not isinstance(amount_paisa, int)
    ):
        return None
    return PendingUpgrade(
        payment_link_key=cast("str", payment_link_key),
        target_plan_id=cast("str", target_plan_id),
        target_provider_plan_id=cast("str", target_provider_plan_id),
        amount_paisa=amount_paisa,
        currency=cast("str", currency),
    )


def _to_lifecycle_snapshot(
    subscription: Subscription,
) -> SubscriptionResult[SubscriptionSnapshot]:
    state_name = str(subscription.status)
    period_start = subscription.current_period_start
    period_end = subscription.current_period_end
    state_by_name: dict[str, object] = {
        SubscriptionStatus.CREATED.value: CreatedState(),
        SubscriptionStatus.AUTHENTICATED.value: AuthenticatedState(),
        SubscriptionStatus.ACTIVE.value: ActiveState(period_start, period_end),
        SubscriptionStatus.PAST_DUE.value: PastDueState(
            retry_count=subscription.retry_count,
            period_start=period_start,
            period_end=period_end,
        ),
        SubscriptionStatus.HALTED.value: HaltedState(period_start, period_end),
        SubscriptionStatus.CANCELLED.value: CancelledState(
            at=subscription.cancelled_at,
            at_period_end=subscription.cancel_at_period_end,
        ),
        SubscriptionStatus.PAUSED.value: PausedState(
            start=subscription.pause_start,
            end=subscription.pause_end,
            period_start=period_start,
            period_end=period_end,
        ),
        SubscriptionStatus.EXPIRED.value: ExpiredState(),
    }
    state = cast("SubscriptionState | None", state_by_name.get(state_name))
    if state is None:
        return Failure(
            SubscriptionValidationError(
                message="Stored subscription status is not supported",
                details={"subscription_id": str(subscription.id), "status": state_name},
                source="subscription_repository",
            )
        )
    intent = (
        CancellationIntent(
            requested_at=subscription.cancellation_requested_at,
            at_period_end=subscription.cancel_at_period_end,
        )
        if subscription.cancellation_requested_at is not None
        else None
    )
    try:
        remote_sync = RemoteSyncState(subscription.remote_sync)
    except ValueError:
        return Failure(
            SubscriptionValidationError(
                message="Stored remote sync state is not supported",
                details={
                    "subscription_id": str(subscription.id),
                    "remote_sync": subscription.remote_sync,
                },
                source="subscription_repository",
            )
        )
    pending_upgrade = _pending_upgrade_from_payload(
        (subscription.metadata_ or {}).get("pending_upgrade")
    )
    return Success(
        SubscriptionSnapshot(
            id=str(subscription.id),
            user_id=subscription.user_id,
            plan_id=str(subscription.plan_id),
            state=state,
            version=subscription.version,
            remote_sync=remote_sync,
            cancellation_intent=intent,
            observed_plan_id=(
                str(subscription.observed_plan_id)
                if subscription.observed_plan_id is not None
                else None
            ),
            pending_upgrade=pending_upgrade,
        )
    )


# Explicit name for callers that want the production coordinator adapter.
# It still inherits the legacy repository surface for backward-compatible
# reads and existing payment/listing callers.
SubscriptionLifecycleRepository = SqlAlchemyLifecyclePersistence
