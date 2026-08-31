"""Subscription persistence operations with optimistic locking.

Rollback contract (ADR D8): classify → rollback → log → return — rollback
precedes any log so a logging failure cannot leave the session poisoned.
For batch callers, this rollback discards all uncommitted updates in the same
session. Each batch item must therefore use an independent session and
transaction; a savepoint on this session cannot contain `session.rollback()`.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING

from returns.result import Failure, Success
from sqlalchemy import func, select, update
from sqlalchemy.exc import IntegrityError, SQLAlchemyError

from app.shared.result import (
    ConflictAppError,
    InfrastructureAppError,
    NotFoundAppError,
    ValidationAppError,
)
from app.utils.codes import ErrorCode

from .model import Subscription, SubscriptionStatus

if TYPE_CHECKING:
    from uuid import UUID

    from sqlalchemy.ext.asyncio import AsyncSession
    from sqlalchemy.sql.selectable import Select

    from app.shared.result import AppResult


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
    current_status: SubscriptionStatus = (
        current if isinstance(current, SubscriptionStatus) else SubscriptionStatus(current)
    )
    target_status: SubscriptionStatus = (
        target if isinstance(target, SubscriptionStatus) else SubscriptionStatus(target)
    )
    return target_status in _ALLOWED_TRANSITIONS.get(current_status, set())


class SubscriptionRepository:
    """Repository for subscription lifecycle."""

    def __init__(self, session: AsyncSession) -> None:
        self.session: AsyncSession = session

    async def create(self, subscription: Subscription) -> AppResult[Subscription]:
        try:
            self.session.add(subscription)
            await self.session.flush()
            return Success(subscription)
        except IntegrityError as exc:
            await self.session.rollback()
            return Failure(
                ConflictAppError(
                    code="DUPLICATE_SUBSCRIPTION",
                    message="An active subscription already exists for this user and plan",
                    details={
                        "user_id": subscription.user_id,
                        "plan_id": str(subscription.plan_id),
                        "error": str(exc),
                    },
                    source="subscription_repository",
                )
            )
        except SQLAlchemyError as exc:
            await self.session.rollback()
            return Failure(
                InfrastructureAppError(
                    code=ErrorCode.DATABASE_ERROR,
                    retryable=False,
                    message="Database error while creating subscription",
                    details={"error": str(exc)},
                    source="subscription_repository",
                )
            )

    async def find_by_id(self, subscription_id: str | UUID) -> AppResult[Subscription | None]:
        try:
            statement: Select[tuple[Subscription]] = select(Subscription).where(
                Subscription.id == subscription_id,
                Subscription.deleted_at.is_(None),
            )
            result = await self.session.execute(statement)
            subscription: Subscription | None = result.scalar_one_or_none()
            if subscription is None:
                return Failure(
                    inner_value=NotFoundAppError(
                        code="SUBSCRIPTION_NOT_FOUND",
                        message="Subscription not found",
                        details={"subscription_id": str(subscription_id)},
                        source="subscription_repository",
                    )
                )
            return Success(subscription)
        except SQLAlchemyError as exc:
            await self.session.rollback()
            return Failure(
                inner_value=InfrastructureAppError(
                    code=ErrorCode.DATABASE_ERROR,
                    retryable=False,
                    message="Database error while fetching subscription",
                    details={"subscription_id": str(subscription_id), "error": str(exc)},
                    source="subscription_repository",
                )
            )

    async def find_by_razorpay_id(
        self, razorpay_subscription_id: str
    ) -> AppResult[Subscription | None]:
        try:
            statement: Select[tuple[Subscription]] = select(Subscription).where(
                Subscription.razorpay_subscription_id == razorpay_subscription_id,
                Subscription.deleted_at.is_(None),
            )
            result = await self.session.execute(statement)
            subscription: Subscription | None = result.scalar_one_or_none()
            if subscription is None:
                return Failure(
                    inner_value=NotFoundAppError(
                        code="SUBSCRIPTION_NOT_FOUND",
                        message="Subscription not found for Razorpay ID",
                        details={"razorpay_subscription_id": razorpay_subscription_id},
                        source="subscription_repository",
                    )
                )
            return Success(inner_value=subscription)
        except SQLAlchemyError as exc:
            await self.session.rollback()
            return Failure(
                inner_value=InfrastructureAppError(
                    code=ErrorCode.DATABASE_ERROR,
                    retryable=False,
                    message="Database error while fetching subscription by Razorpay ID",
                    details={
                        "razorpay_subscription_id": razorpay_subscription_id,
                        "error": str(exc),
                    },
                    source="subscription_repository",
                )
            )

    async def find_by_user_and_plan(
        self, user_id: str, plan_id: str | UUID
    ) -> AppResult[Subscription | None]:
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
            await self.session.rollback()
            return Failure(
                InfrastructureAppError(
                    code=ErrorCode.DATABASE_ERROR,
                    retryable=False,
                    message="Database error while finding subscription by user and plan",
                    details={"user_id": user_id, "plan_id": str(plan_id), "error": str(exc)},
                    source="subscription_repository",
                )
            )

    async def list_by_user(
        self,
        user_id: str,
        *,
        status: SubscriptionStatus | None = None,
        plan_id: str | UUID | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> AppResult[tuple[list[Subscription], int]]:
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
            await self.session.rollback()
            return Failure(
                InfrastructureAppError(
                    code=ErrorCode.DATABASE_ERROR,
                    retryable=False,
                    message="Database error while listing subscriptions",
                    details={"user_id": user_id, "error": str(exc)},
                    source="subscription_repository",
                )
            )

    async def update_with_lock(
        self,
        subscription: Subscription,
        expected_version: int,
        *,
        values: dict[str, object],
    ) -> AppResult[Subscription]:
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
                    ConflictAppError(
                        code="VERSION_CONFLICT",
                        message="Subscription was modified concurrently; refetch and retry",
                        details={
                            "subscription_id": str(subscription.id),
                            "expected_version": expected_version,
                        },
                        source="subscription_repository",
                    )
                )
            return Success(updated)
        except SQLAlchemyError as exc:
            await self.session.rollback()
            return Failure(
                InfrastructureAppError(
                    code=ErrorCode.DATABASE_ERROR,
                    retryable=False,
                    message="Database error while updating subscription",
                    details={"subscription_id": str(subscription.id), "error": str(exc)},
                    source="subscription_repository",
                )
            )

    async def update_status(
        self,
        subscription: Subscription,
        new_status: SubscriptionStatus,
        *,
        expected_version: int,
        extra_values: dict[str, object] | None = None,
    ) -> AppResult[Subscription]:
        """Update status with state-machine validation and optimistic locking."""
        if not validate_transition(subscription.status, new_status):
            return Failure(
                ValidationAppError(
                    code="INVALID_STATE_TRANSITION",
                    message=f"Invalid subscription state transition: {subscription.status} -> {new_status}",
                    details={
                        "subscription_id": str(subscription.id),
                        "current": subscription.status,
                        "target": new_status.value,
                    },
                    source="subscription_repository",
                )
            )
        values: dict[str, object] = {"status": new_status.value}
        if extra_values:
            values.update(extra_values)
        return await self.update_with_lock(subscription, expected_version, values=values)

    async def increment_retry_count(
        self, subscription: Subscription, *, expected_version: int
    ) -> AppResult[Subscription]:
        return await self.update_with_lock(
            subscription,
            expected_version,
            values={"retry_count": Subscription.retry_count + 1},
        )

    async def reset_retry_count(
        self, subscription: Subscription, *, expected_version: int
    ) -> AppResult[Subscription]:
        return await self.update_with_lock(
            subscription, expected_version, values={"retry_count": 0}
        )
