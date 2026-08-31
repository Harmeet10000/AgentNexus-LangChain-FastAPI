"""Subscriptions feature typed errors — per-feature closed union."""

from __future__ import annotations

from enum import StrEnum
from typing import TYPE_CHECKING

from returns.result import Result

from app.shared.result.errors import ErrorKind, FeatureError

if TYPE_CHECKING:
    from typing import ClassVar


class SubscriptionCode(StrEnum):
    SUBSCRIPTION_NOT_FOUND = "SUBSCRIPTION_NOT_FOUND"
    DUPLICATE_SUBSCRIPTION = "DUPLICATE_SUBSCRIPTION"
    VERSION_CONFLICT = "VERSION_CONFLICT"
    INVALID_STATE_TRANSITION = "INVALID_STATE_TRANSITION"
    PLAN_NOT_FOUND = "PLAN_NOT_FOUND"
    DATABASE_ERROR = "DATABASE_ERROR"
    VALIDATION_ERROR = "VALIDATION_ERROR"


class SubscriptionNotFoundError(FeatureError):
    kind: ClassVar[ErrorKind] = ErrorKind.NOT_FOUND
    code: ClassVar[SubscriptionCode] = SubscriptionCode.SUBSCRIPTION_NOT_FOUND
    retryable: ClassVar[bool] = False

    subscription_id: str | None = None


class SubscriptionDuplicateError(FeatureError):
    kind: ClassVar[ErrorKind] = ErrorKind.CONFLICT
    code: ClassVar[SubscriptionCode] = SubscriptionCode.DUPLICATE_SUBSCRIPTION
    retryable: ClassVar[bool] = False

    user_id: str | None = None
    plan_id: str | None = None


class SubscriptionVersionConflictError(FeatureError):
    kind: ClassVar[ErrorKind] = ErrorKind.CONFLICT
    code: ClassVar[SubscriptionCode] = SubscriptionCode.VERSION_CONFLICT
    retryable: ClassVar[bool] = False

    subscription_id: str | None = None
    expected_version: int | None = None


class SubscriptionInvalidTransitionError(FeatureError):
    kind: ClassVar[ErrorKind] = ErrorKind.VALIDATION
    code: ClassVar[SubscriptionCode] = SubscriptionCode.INVALID_STATE_TRANSITION
    retryable: ClassVar[bool] = False

    subscription_id: str | None = None
    current: str | None = None
    target: str | None = None


class SubscriptionPlanNotFoundError(FeatureError):
    kind: ClassVar[ErrorKind] = ErrorKind.NOT_FOUND
    code: ClassVar[SubscriptionCode] = SubscriptionCode.PLAN_NOT_FOUND
    retryable: ClassVar[bool] = False

    plan_id: str | None = None


class SubscriptionInfrastructureError(FeatureError):
    kind: ClassVar[ErrorKind] = ErrorKind.INFRASTRUCTURE
    code: ClassVar[SubscriptionCode] = SubscriptionCode.DATABASE_ERROR
    retryable: ClassVar[bool] = False

    operation: str | None = None


class SubscriptionTransientInfrastructureError(FeatureError):
    kind: ClassVar[ErrorKind] = ErrorKind.INFRASTRUCTURE
    code: ClassVar[SubscriptionCode] = SubscriptionCode.DATABASE_ERROR
    retryable: ClassVar[bool] = True

    operation: str | None = None


class SubscriptionValidationError(FeatureError):
    kind: ClassVar[ErrorKind] = ErrorKind.VALIDATION
    code: ClassVar[SubscriptionCode] = SubscriptionCode.VALIDATION_ERROR
    retryable: ClassVar[bool] = False


type SubscriptionError = (
    SubscriptionNotFoundError
    | SubscriptionDuplicateError
    | SubscriptionVersionConflictError
    | SubscriptionInvalidTransitionError
    | SubscriptionPlanNotFoundError
    | SubscriptionInfrastructureError
    | SubscriptionTransientInfrastructureError
    | SubscriptionValidationError
)

type SubscriptionResult[T] = Result[T, SubscriptionError]
