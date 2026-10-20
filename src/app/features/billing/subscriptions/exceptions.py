"""Subscriptions feature typed errors — per-feature closed union."""

from __future__ import annotations

from enum import StrEnum
from typing import TYPE_CHECKING, assert_never

from returns.result import Result

from app.kernel.errors import ErrorKind, FeatureError
from app.kernel.result import http_status_for_kind

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
    REFUND_RECONCILIATION_REQUIRED = "REFUND_RECONCILIATION_REQUIRED"


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


class SubscriptionRefundError(FeatureError):
    """Cancellation succeeded locally but its economic refund is unresolved."""

    kind: ClassVar[ErrorKind] = ErrorKind.EXTERNAL_SERVICE
    code: ClassVar[SubscriptionCode] = SubscriptionCode.REFUND_RECONCILIATION_REQUIRED
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
    | SubscriptionRefundError
)

type SubscriptionResult[T] = Result[T, SubscriptionError]


def subscription_error_to_http_status(error: SubscriptionError) -> int:
    match error:
        case SubscriptionNotFoundError() | SubscriptionPlanNotFoundError():
            return http_status_for_kind(error.kind)
        case SubscriptionDuplicateError() | SubscriptionVersionConflictError():
            return http_status_for_kind(error.kind)
        case SubscriptionInvalidTransitionError() | SubscriptionValidationError():
            return http_status_for_kind(error.kind)
        case SubscriptionInfrastructureError() | SubscriptionTransientInfrastructureError():
            return http_status_for_kind(error.kind, retryable=error.retryable)
        case SubscriptionRefundError():
            return http_status_for_kind(error.kind)
        case _ as unreachable:
            assert_never(unreachable)
