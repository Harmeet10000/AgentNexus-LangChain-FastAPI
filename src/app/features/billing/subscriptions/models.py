"""Subscription and subscription-status SQLAlchemy models."""

from datetime import UTC, datetime
from enum import StrEnum
from uuid import UUID, uuid4

from sqlalchemy import (
    BigInteger,
    Boolean,
    CheckConstraint,
    DateTime,
    ForeignKey,
    Index,
    String,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.dialects.postgresql import UUID as PGUUID
from sqlalchemy.orm import Mapped, mapped_column, synonym

from database.base import Base


class SubscriptionStatus(StrEnum):
    CREATED = "created"
    AUTHENTICATED = "authenticated"
    ACTIVE = "active"
    PAST_DUE = "past_due"
    HALTED = "halted"
    CANCELLED = "cancelled"
    PAUSED = "paused"
    EXPIRED = "expired"


class RemoteSyncState(StrEnum):
    """Durable state of the local desired state versus the provider."""

    SYNCED = "synced"
    PENDING = "pending"
    RETRYABLE_FAILURE = "retryable_failure"
    PERMANENT_FAILURE = "permanent_failure"


class ProviderOperationState(StrEnum):
    """States that keep provider mutations safe to reconcile."""

    PENDING = "pending"
    SUCCEEDED = "succeeded"
    UNKNOWN = "unknown"
    RECONCILIATION = "reconciliation"
    MANUAL_REVIEW = "manual_review"
    FAILED = "failed"


class Subscription(Base):
    """User subscription to a billing plan.

    ``version`` is the optimistic-locking field: every update runs
    ``UPDATE ... WHERE id = :id AND version = :expected`` and bumps the
    version, so concurrent writers cannot silently lose updates.
    """

    __tablename__ = "subscriptions"
    __table_args__ = (
        CheckConstraint(
            "remote_sync IN ('synced', 'pending', 'retryable_failure', 'permanent_failure')",
            name="ck_subscriptions_remote_sync",
        ),
        Index("ix_subscriptions_user_id", "user_id"),
        Index("ix_subscriptions_razorpay_subscription_id", "razorpay_subscription_id"),
        Index("ix_subscriptions_plan_id", "plan_id"),
        Index("ix_subscriptions_observed_plan_id", "observed_plan_id"),
        Index("ix_subscriptions_remote_sync", "remote_sync"),
        Index("ix_subscriptions_id_version", "id", "version"),
        Index(
            "uq_subscriptions_user_plan_active",
            "user_id",
            "plan_id",
            unique=True,
            postgresql_where=text("deleted_at IS NULL AND status NOT IN ('cancelled', 'expired')"),
        ),
    )

    id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True), primary_key=True, default=uuid4)
    user_id: Mapped[str] = mapped_column(String(length=255), nullable=False)
    plan_id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True), ForeignKey(column="plans.id"), nullable=False
    )
    # ``plan_id`` is the historical public field and remains the desired local
    # plan.  The synonym makes that invariant explicit to new coordinators
    # without breaking existing service/repository callers.
    desired_plan_id = synonym("plan_id")
    observed_plan_id: Mapped[UUID | None] = mapped_column(PGUUID(as_uuid=True), nullable=True)
    provider_observed_plan_id = synonym("observed_plan_id")
    razorpay_subscription_id: Mapped[str | None] = mapped_column(String(length=64), nullable=True)
    razorpay_customer_id: Mapped[str | None] = mapped_column(String(length=64), nullable=True)

    status: Mapped[str] = mapped_column(String(length=16), nullable=False, default="created")

    current_period_start: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    current_period_end: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    trial_end: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    cancel_at_period_end: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    cancellation_requested_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    entitlement_until: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    cancelled_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    ended_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    pause_start: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    pause_end: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    retry_count: Mapped[int] = mapped_column(BigInteger, nullable=False, default=0)
    max_retries: Mapped[int] = mapped_column(BigInteger, nullable=False, default=4)

    currency_display: Mapped[str] = mapped_column(String(length=3), nullable=False, default="INR")
    trial_extension_count: Mapped[int] = mapped_column(BigInteger, nullable=False, default=0)
    deleted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    version: Mapped[int] = mapped_column(BigInteger, nullable=False, default=0)
    lifecycle_version = synonym("version")

    remote_sync: Mapped[str] = mapped_column(
        String(length=32),
        nullable=False,
        default=RemoteSyncState.PENDING.value,
        server_default=RemoteSyncState.PENDING.value,
    )
    remote_sync_state = synonym("remote_sync")
    cancellation_intent_requested_at = synonym("cancellation_requested_at")

    metadata_: Mapped[dict[str, object]] = mapped_column(JSONB, default=dict, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(tz=UTC),
        nullable=False,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(tz=UTC),
        onupdate=lambda: datetime.now(tz=UTC),
        nullable=False,
    )


class ProviderOperationRecord(Base):
    """Durable ledger entry for one provider operation attempt.

    A mutation whose network outcome is unknown must remain represented here;
    the coordinator can then search by ``correlation_id``/``idempotency_key``
    before issuing another mutation.  Request and observation payloads are
    intentionally operation-specific normalized facts, not a replacement for
    queryable lifecycle columns on :class:`Subscription`.
    """

    __tablename__ = "provider_operation_records"
    __table_args__ = (
        CheckConstraint(
            "state IN ('pending', 'succeeded', 'unknown', 'reconciliation', "
            "'manual_review', 'failed')",
            name="ck_provider_operation_records_state",
        ),
        Index("ix_provider_operation_records_subscription_id", "subscription_id"),
        Index("ix_provider_operation_records_correlation_id", "correlation_id"),
        Index("ix_provider_operation_records_state", "state"),
    )

    id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True), primary_key=True, default=uuid4)
    subscription_id: Mapped[UUID | None] = mapped_column(
        PGUUID(as_uuid=True),
        # Operation history is the audit/reconciliation record of what was
        # sent to the provider.  Soft-deleting a subscription must not erase
        # that evidence; retain the row and detach it when a hard delete is
        # unavoidable.
        ForeignKey(column="subscriptions.id", ondelete="SET NULL"),
        nullable=True,
    )
    correlation_id: Mapped[str] = mapped_column(String(length=128), nullable=False)
    idempotency_key: Mapped[str] = mapped_column(String(length=128), nullable=False, unique=True)
    operation_kind: Mapped[str] = mapped_column(String(length=64), nullable=False)
    state: Mapped[str] = mapped_column(
        String(length=32),
        nullable=False,
        default=ProviderOperationState.PENDING.value,
        server_default=ProviderOperationState.PENDING.value,
    )
    provider_operation_state = synonym("state")
    attempts: Mapped[int] = mapped_column(BigInteger, nullable=False, default=0, server_default="0")
    provider_resource_id: Mapped[str | None] = mapped_column(String(length=128), nullable=True)
    error_code: Mapped[str | None] = mapped_column(String(length=64), nullable=True)
    error_message: Mapped[str | None] = mapped_column(String(length=1000), nullable=True)
    sanitized_error = synonym("error_message")
    request_data: Mapped[dict[str, object]] = mapped_column(
        JSONB, nullable=False, default=dict, server_default=text("'{}'::jsonb")
    )
    observation_data: Mapped[dict[str, object]] = mapped_column(
        JSONB, nullable=False, default=dict, server_default=text("'{}'::jsonb")
    )
    normalized_request = synonym("request_data")
    normalized_observation = synonym("observation_data")
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    last_attempt_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(tz=UTC), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(tz=UTC),
        onupdate=lambda: datetime.now(tz=UTC),
        nullable=False,
    )
