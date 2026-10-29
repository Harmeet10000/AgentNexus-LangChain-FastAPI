"""Provider-neutral billing operations and typed results.

Razorpay is an external observer, not a transactional participant.  The API
does not document request idempotency for the mutation endpoints used here, so
every mutation carries a durable local correlation ID.  A transport failure
after a mutation has been sent is represented by a typed reconciliation error;
callers must read/search before deciding whether another write is safe.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime
from enum import StrEnum
from typing import ClassVar, Literal, Protocol, override

from pydantic import BaseModel, ConfigDict, Field
from returns.result import Result


def _invalid(message: str) -> None:
    raise ValueError(message)


def _require_text(value: str, *, field_name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        _invalid(f"{field_name} must not be empty")


def _require_notes(notes: Mapping[str, str]) -> None:
    if not isinstance(notes, Mapping):
        _invalid("notes must be a mapping")
    if len(notes) > 15:
        _invalid("notes must contain at most 15 entries")
    if any(not isinstance(key, str) or not isinstance(value, str) for key, value in notes.items()):
        _invalid("notes keys and values must be strings")
    if any(len(key) > 256 or len(value) > 256 for key, value in notes.items()):
        _invalid("notes keys and values must be at most 256 characters")


class ProviderOperationKind(StrEnum):
    """Closed set of operations supported by the billing provider port."""

    CREATE_CUSTOMER = "create_customer"
    FIND_CUSTOMER_BY_EMAIL = "find_customer_by_email"
    CREATE_SUBSCRIPTION = "create_subscription"
    CANCEL_SUBSCRIPTION = "cancel_subscription"
    PAUSE_SUBSCRIPTION = "pause_subscription"
    RESUME_SUBSCRIPTION = "resume_subscription"
    CHANGE_SUBSCRIPTION_PLAN = "change_subscription_plan"
    OBSERVE_SUBSCRIPTION = "observe_subscription"
    FIND_SUBSCRIPTION_BY_CORRELATION = "find_subscription_by_correlation"
    LIST_PAYMENTS = "list_payments"
    CREATE_PAYMENT_LINK = "create_payment_link"
    FIND_PAYMENT_LINK_BY_CORRELATION = "find_payment_link_by_correlation"
    CREATE_REFUND = "create_refund"
    FIND_REFUND_BY_CORRELATION = "find_refund_by_correlation"
    CREATE_PLAN = "create_plan"
    FIND_PLAN_BY_CORRELATION = "find_plan_by_correlation"


class CreateCustomerOperation(BaseModel):
    model_config = ConfigDict(frozen=True)

    email: str
    contact: str | None
    name: str | None
    correlation_id: str
    notes: Mapping[str, str] = Field(default_factory=dict)

    kind: ClassVar[ProviderOperationKind] = ProviderOperationKind.CREATE_CUSTOMER

    @override
    def model_post_init(self, __context: object) -> None:
        _require_text(self.email, field_name="email")
        _require_text(self.correlation_id, field_name="correlation_id")
        if self.contact is not None:
            _require_text(self.contact, field_name="contact")
        if self.name is not None:
            _require_text(self.name, field_name="name")
        _require_notes(self.notes)


class FindCustomerByEmailOperation(BaseModel):
    model_config = ConfigDict(frozen=True)

    email: str
    correlation_id: str

    kind: ClassVar[ProviderOperationKind] = ProviderOperationKind.FIND_CUSTOMER_BY_EMAIL

    @override
    def model_post_init(self, __context: object) -> None:
        _require_text(self.email, field_name="email")
        _require_text(self.correlation_id, field_name="correlation_id")


class CreateSubscriptionOperation(BaseModel):
    model_config = ConfigDict(frozen=True)

    plan_id: str
    customer_id: str
    total_count: int
    quantity: int
    customer_notify: bool
    correlation_id: str
    notes: Mapping[str, str] = Field(default_factory=dict)

    kind: ClassVar[ProviderOperationKind] = ProviderOperationKind.CREATE_SUBSCRIPTION

    @override
    def model_post_init(self, __context: object) -> None:
        _require_text(self.plan_id, field_name="plan_id")
        _require_text(self.customer_id, field_name="customer_id")
        _require_text(self.correlation_id, field_name="correlation_id")
        if isinstance(self.total_count, bool) or not isinstance(self.total_count, int):
            _invalid("total_count must be an integer")
        if self.total_count < 0:
            _invalid("total_count must be non-negative")
        if isinstance(self.quantity, bool) or not isinstance(self.quantity, int):
            _invalid("quantity must be an integer")
        if self.quantity <= 0:
            _invalid("quantity must be positive")
        _require_notes(self.notes)


class CancelSubscriptionOperation(BaseModel):
    model_config = ConfigDict(frozen=True)

    provider_subscription_id: str
    cancel_at_cycle_end: bool
    correlation_id: str

    kind: ClassVar[ProviderOperationKind] = ProviderOperationKind.CANCEL_SUBSCRIPTION

    @override
    def model_post_init(self, __context: object) -> None:
        _require_text(self.provider_subscription_id, field_name="provider_subscription_id")
        _require_text(self.correlation_id, field_name="correlation_id")
        if not isinstance(self.cancel_at_cycle_end, bool):
            _invalid("cancel_at_cycle_end must be a boolean")


class PauseSubscriptionOperation(BaseModel):
    model_config = ConfigDict(frozen=True)

    """Pause immediately; duration is a local scheduling fact.

    Razorpay's documented pause request accepts only ``pause_at=now``.  A
    finite duration therefore belongs to the local desired state/coordinator,
    which must later issue a separate resume operation.
    """

    provider_subscription_id: str
    correlation_id: str
    duration_days: int | None = None
    pause_at: Literal["now"] = "now"

    kind: ClassVar[ProviderOperationKind] = ProviderOperationKind.PAUSE_SUBSCRIPTION

    @override
    def model_post_init(self, __context: object) -> None:
        _require_text(self.provider_subscription_id, field_name="provider_subscription_id")
        _require_text(self.correlation_id, field_name="correlation_id")
        if self.duration_days is not None and (
            isinstance(self.duration_days, bool)
            or not isinstance(self.duration_days, int)
            or self.duration_days <= 0
            or self.duration_days > 365
        ):
            _invalid("duration_days must be positive and at most 365 when provided")
        if self.pause_at != "now":
            _invalid("pause_at must be 'now'")


class ResumeSubscriptionOperation(BaseModel):
    model_config = ConfigDict(frozen=True)

    provider_subscription_id: str
    correlation_id: str
    resume_at: Literal["now"] = "now"

    kind: ClassVar[ProviderOperationKind] = ProviderOperationKind.RESUME_SUBSCRIPTION

    @override
    def model_post_init(self, __context: object) -> None:
        _require_text(self.provider_subscription_id, field_name="provider_subscription_id")
        _require_text(self.correlation_id, field_name="correlation_id")
        if self.resume_at != "now":
            _invalid("resume_at must be 'now'")


class ChangeSubscriptionPlanOperation(BaseModel):
    model_config = ConfigDict(frozen=True)

    provider_subscription_id: str
    plan_id: str
    correlation_id: str
    schedule_at: Literal["now", "cycle_end"] = "now"

    kind: ClassVar[ProviderOperationKind] = ProviderOperationKind.CHANGE_SUBSCRIPTION_PLAN

    @override
    def model_post_init(self, __context: object) -> None:
        _require_text(self.provider_subscription_id, field_name="provider_subscription_id")
        _require_text(self.plan_id, field_name="plan_id")
        _require_text(self.correlation_id, field_name="correlation_id")
        if self.schedule_at not in {"now", "cycle_end"}:
            _invalid("schedule_at must be 'now' or 'cycle_end'")


class ObserveSubscriptionOperation(BaseModel):
    model_config = ConfigDict(frozen=True)

    provider_subscription_id: str
    correlation_id: str

    kind: ClassVar[ProviderOperationKind] = ProviderOperationKind.OBSERVE_SUBSCRIPTION

    @override
    def model_post_init(self, __context: object) -> None:
        _require_text(self.provider_subscription_id, field_name="provider_subscription_id")
        _require_text(self.correlation_id, field_name="correlation_id")


class FindSubscriptionByCorrelationOperation(BaseModel):
    model_config = ConfigDict(frozen=True)

    correlation_id: str

    kind: ClassVar[ProviderOperationKind] = ProviderOperationKind.FIND_SUBSCRIPTION_BY_CORRELATION

    @override
    def model_post_init(self, __context: object) -> None:
        _require_text(self.correlation_id, field_name="correlation_id")


class ListPaymentsOperation(BaseModel):
    model_config = ConfigDict(frozen=True)

    """Read a bounded provider payment window for reconciliation.

    The provider port exposes the observation as one typed operation so jobs
    cannot accidentally turn a reconciliation sweep into a remote mutation.
    Epoch fields match Razorpay's wire contract and are intentionally optional
    to support a full bounded list when a caller has no lookback window.
    """

    correlation_id: str
    from_timestamp: int | None = None
    to_timestamp: int | None = None
    count: int = 100

    kind: ClassVar[ProviderOperationKind] = ProviderOperationKind.LIST_PAYMENTS

    @override
    def model_post_init(self, __context: object) -> None:
        _require_text(self.correlation_id, field_name="correlation_id")
        if self.from_timestamp is not None and (
            isinstance(self.from_timestamp, bool) or not isinstance(self.from_timestamp, int)
        ):
            _invalid("from_timestamp must be an integer when provided")
        if self.to_timestamp is not None and (
            isinstance(self.to_timestamp, bool) or not isinstance(self.to_timestamp, int)
        ):
            _invalid("to_timestamp must be an integer when provided")
        if (
            isinstance(self.count, bool)
            or not isinstance(self.count, int)
            or self.count <= 0
            or self.count > 100
        ):
            _invalid("count must be positive and at most 100")
        if (
            self.from_timestamp is not None
            and self.to_timestamp is not None
            and self.from_timestamp > self.to_timestamp
        ):
            _invalid("from_timestamp must not be after to_timestamp")


class CreatePaymentLinkOperation(BaseModel):
    model_config = ConfigDict(frozen=True)

    amount: int
    currency: str
    description: str
    correlation_id: str
    customer_id: str | None = None
    notes: Mapping[str, str] = Field(default_factory=dict)

    kind: ClassVar[ProviderOperationKind] = ProviderOperationKind.CREATE_PAYMENT_LINK

    @override
    def model_post_init(self, __context: object) -> None:
        if isinstance(self.amount, bool) or not isinstance(self.amount, int):
            _invalid("amount must be an integer")
        if self.amount <= 0:
            _invalid("amount must be positive")
        if (
            not isinstance(self.currency, str)
            or len(self.currency) != 3
            or not self.currency.isascii()
            or not self.currency.isalpha()
            or not self.currency.isupper()
        ):
            _invalid("currency must be a three-letter code")
        _require_text(self.description, field_name="description")
        _require_text(self.correlation_id, field_name="correlation_id")
        _require_notes(self.notes)


class FindPaymentLinkByCorrelationOperation(BaseModel):
    model_config = ConfigDict(frozen=True)

    correlation_id: str

    kind: ClassVar[ProviderOperationKind] = ProviderOperationKind.FIND_PAYMENT_LINK_BY_CORRELATION

    @override
    def model_post_init(self, __context: object) -> None:
        _require_text(self.correlation_id, field_name="correlation_id")


class CreateRefundOperation(BaseModel):
    model_config = ConfigDict(frozen=True)

    payment_id: str
    amount: int
    correlation_id: str
    notes: Mapping[str, str] = Field(default_factory=dict)

    kind: ClassVar[ProviderOperationKind] = ProviderOperationKind.CREATE_REFUND

    @override
    def model_post_init(self, __context: object) -> None:
        _require_text(self.payment_id, field_name="payment_id")
        if isinstance(self.amount, bool) or not isinstance(self.amount, int):
            _invalid("amount must be an integer")
        if self.amount <= 0:
            _invalid("amount must be positive")
        _require_text(self.correlation_id, field_name="correlation_id")
        _require_notes(self.notes)


class FindRefundByCorrelationOperation(BaseModel):
    model_config = ConfigDict(frozen=True)

    payment_id: str
    correlation_id: str

    kind: ClassVar[ProviderOperationKind] = ProviderOperationKind.FIND_REFUND_BY_CORRELATION

    @override
    def model_post_init(self, __context: object) -> None:
        _require_text(self.payment_id, field_name="payment_id")
        _require_text(self.correlation_id, field_name="correlation_id")


class CreatePlanOperation(BaseModel):
    """Create an immutable provider plan under a durable correlation key."""

    model_config = ConfigDict(frozen=True)

    name: str
    amount: int
    currency: str
    period: str
    interval: int
    correlation_id: str
    notes: Mapping[str, str] = Field(default_factory=dict)

    kind: ClassVar[ProviderOperationKind] = ProviderOperationKind.CREATE_PLAN

    @override
    def model_post_init(self, __context: object) -> None:
        _require_text(self.name, field_name="name")
        if isinstance(self.amount, bool) or not isinstance(self.amount, int) or self.amount <= 0:
            _invalid("amount must be a positive integer")
        if (
            not isinstance(self.currency, str)
            or len(self.currency) != 3
            or not self.currency.isascii()
            or not self.currency.isalpha()
            or not self.currency.isupper()
        ):
            _invalid("currency must be a three-letter code")
        _require_text(self.period, field_name="period")
        if (
            isinstance(self.interval, bool)
            or not isinstance(self.interval, int)
            or self.interval <= 0
        ):
            _invalid("interval must be a positive integer")
        _require_text(self.correlation_id, field_name="correlation_id")
        _require_notes(self.notes)


class FindPlanByCorrelationOperation(BaseModel):
    model_config = ConfigDict(frozen=True)

    correlation_id: str

    kind: ClassVar[ProviderOperationKind] = ProviderOperationKind.FIND_PLAN_BY_CORRELATION

    @override
    def model_post_init(self, __context: object) -> None:
        _require_text(self.correlation_id, field_name="correlation_id")


type ProviderOperation = (
    CreateCustomerOperation
    | FindCustomerByEmailOperation
    | CreateSubscriptionOperation
    | CancelSubscriptionOperation
    | PauseSubscriptionOperation
    | ResumeSubscriptionOperation
    | ChangeSubscriptionPlanOperation
    | ObserveSubscriptionOperation
    | FindSubscriptionByCorrelationOperation
    | ListPaymentsOperation
    | CreatePaymentLinkOperation
    | FindPaymentLinkByCorrelationOperation
    | CreateRefundOperation
    | FindRefundByCorrelationOperation
    | CreatePlanOperation
    | FindPlanByCorrelationOperation
)


class ProviderErrorCode(StrEnum):
    INVALID_REQUEST = "invalid_request"
    AUTHENTICATION = "authentication"
    RATE_LIMITED = "rate_limited"
    UNAVAILABLE = "unavailable"
    NOT_FOUND = "not_found"
    OUTCOME_UNKNOWN = "outcome_unknown"
    MANUAL_REVIEW_REQUIRED = "manual_review_required"
    RECONCILIATION_REQUIRED = "reconciliation_required"
    UNEXPECTED = "unexpected"


class ProviderInvalidRequestError(BaseModel):
    model_config = ConfigDict(frozen=True)

    message: str = "The billing provider rejected the request"
    operation: ProviderOperationKind | None = None
    correlation_id: str | None = None

    code: ClassVar[ProviderErrorCode] = ProviderErrorCode.INVALID_REQUEST
    retryable: ClassVar[bool] = False
    requires_reconciliation: ClassVar[bool] = False


class ProviderAuthenticationError(BaseModel):
    model_config = ConfigDict(frozen=True)

    message: str = "The billing provider credentials were rejected"
    operation: ProviderOperationKind | None = None
    correlation_id: str | None = None

    code: ClassVar[ProviderErrorCode] = ProviderErrorCode.AUTHENTICATION
    retryable: ClassVar[bool] = False
    requires_reconciliation: ClassVar[bool] = False


class ProviderRateLimitedError(BaseModel):
    model_config = ConfigDict(frozen=True)

    message: str = "The billing provider rate limit was reached"
    operation: ProviderOperationKind | None = None
    correlation_id: str | None = None

    code: ClassVar[ProviderErrorCode] = ProviderErrorCode.RATE_LIMITED
    retryable: ClassVar[bool] = True
    requires_reconciliation: ClassVar[bool] = False


class ProviderUnavailableError(BaseModel):
    model_config = ConfigDict(frozen=True)

    message: str = "The billing provider is temporarily unavailable"
    operation: ProviderOperationKind | None = None
    correlation_id: str | None = None

    code: ClassVar[ProviderErrorCode] = ProviderErrorCode.UNAVAILABLE
    retryable: ClassVar[bool] = True
    requires_reconciliation: ClassVar[bool] = False


class ProviderNotFoundError(BaseModel):
    model_config = ConfigDict(frozen=True)

    message: str = "The billing provider resource was not found"
    operation: ProviderOperationKind | None = None
    correlation_id: str | None = None

    code: ClassVar[ProviderErrorCode] = ProviderErrorCode.NOT_FOUND
    retryable: ClassVar[bool] = False
    requires_reconciliation: ClassVar[bool] = False


class ProviderOutcomeUnknownError(BaseModel):
    """A mutation may have committed and must be reconciled before retry."""

    model_config = ConfigDict(frozen=True)

    operation: ProviderOperationKind
    correlation_id: str
    message: str = "The provider outcome is unknown; reconcile before retrying"

    code: ClassVar[ProviderErrorCode] = ProviderErrorCode.OUTCOME_UNKNOWN
    retryable: ClassVar[bool] = False
    requires_reconciliation: ClassVar[bool] = True


class ProviderManualReviewRequiredError(BaseModel):
    """The provider cannot deterministically resolve a correlation search."""

    model_config = ConfigDict(frozen=True)

    operation: ProviderOperationKind
    correlation_id: str
    message: str = "Provider correlation could not be resolved; manual review is required"

    code: ClassVar[ProviderErrorCode] = ProviderErrorCode.MANUAL_REVIEW_REQUIRED
    retryable: ClassVar[bool] = False
    requires_reconciliation: ClassVar[bool] = True


class ProviderReconciliationRequiredError(BaseModel):
    """A provider read returned an unusable fact; reconcile before mutating."""

    model_config = ConfigDict(frozen=True)

    operation: ProviderOperationKind
    correlation_id: str
    message: str = "Provider state requires reconciliation before continuing"

    code: ClassVar[ProviderErrorCode] = ProviderErrorCode.RECONCILIATION_REQUIRED
    retryable: ClassVar[bool] = False
    requires_reconciliation: ClassVar[bool] = True


class ProviderUnexpectedError(BaseModel):
    model_config = ConfigDict(frozen=True)

    message: str = "The billing provider returned an unusable response"
    operation: ProviderOperationKind | None = None
    correlation_id: str | None = None

    code: ClassVar[ProviderErrorCode] = ProviderErrorCode.UNEXPECTED
    retryable: ClassVar[bool] = False
    requires_reconciliation: ClassVar[bool] = False


type ProviderError = (
    ProviderInvalidRequestError
    | ProviderAuthenticationError
    | ProviderRateLimitedError
    | ProviderUnavailableError
    | ProviderNotFoundError
    | ProviderOutcomeUnknownError
    | ProviderManualReviewRequiredError
    | ProviderReconciliationRequiredError
    | ProviderUnexpectedError
)


class ProviderSubscriptionState(BaseModel):
    model_config = ConfigDict(frozen=True)

    provider_subscription_id: str
    status: str
    current_period_start: datetime | None = None
    current_period_end: datetime | None = None
    ended_at: datetime | None = None
    payment_url: str | None = None
    cancel_at_cycle_end: bool | None = None
    plan_id: str | None = None

    @override
    def model_post_init(self, __context: object) -> None:
        _require_text(self.provider_subscription_id, field_name="provider_subscription_id")
        _require_text(self.status, field_name="status")
        if self.status not in {
            "created",
            "authenticated",
            "active",
            "pending",
            "paused",
            "halted",
            "cancelled",
            "completed",
            "expired",
        }:
            _invalid("status is not a supported Razorpay subscription status")
        if self.plan_id is not None:
            _require_text(self.plan_id, field_name="plan_id")

    @property
    def grants_paid_entitlement(self) -> bool:
        return self.status == "active"


class ProviderCustomer(BaseModel):
    model_config = ConfigDict(frozen=True)

    provider_customer_id: str
    email: str

    @override
    def model_post_init(self, __context: object) -> None:
        _require_text(self.provider_customer_id, field_name="provider_customer_id")
        _require_text(self.email, field_name="email")


class ProviderPaymentLink(BaseModel):
    model_config = ConfigDict(frozen=True)

    provider_payment_link_id: str
    short_url: str | None = None
    status: str | None = None

    @override
    def model_post_init(self, __context: object) -> None:
        _require_text(self.provider_payment_link_id, field_name="provider_payment_link_id")
        if self.short_url is not None:
            _require_text(self.short_url, field_name="short_url")
        if self.status is not None and self.status not in {
            "created",
            "partially_paid",
            "expired",
            "cancelled",
            "paid",
            # Legacy invoice-backed payment-link responses use ``issued``.
            "issued",
        }:
            _invalid("status is not a supported Razorpay payment-link status")


class ProviderPlan(BaseModel):
    model_config = ConfigDict(frozen=True)

    provider_plan_id: str
    name: str | None = None

    @override
    def model_post_init(self, __context: object) -> None:
        _require_text(self.provider_plan_id, field_name="provider_plan_id")


class ProviderReceipt(BaseModel):
    model_config = ConfigDict(frozen=True)

    provider_refund_id: str
    payment_id: str
    amount: int

    @override
    def model_post_init(self, __context: object) -> None:
        _require_text(self.provider_refund_id, field_name="provider_refund_id")
        _require_text(self.payment_id, field_name="payment_id")
        if isinstance(self.amount, bool) or not isinstance(self.amount, int) or self.amount <= 0:
            _invalid("amount must be positive")


class ProviderPaymentState(BaseModel):
    """One normalized payment fact observed from the billing provider."""

    model_config = ConfigDict(frozen=True)

    provider_payment_id: str
    status: str
    amount: int
    currency: str
    subscription_id: str | None = None
    method: str | None = None
    created_at: datetime | None = None
    captured_at: datetime | None = None

    @override
    def model_post_init(self, __context: object) -> None:
        _require_text(self.provider_payment_id, field_name="provider_payment_id")
        _require_text(self.status, field_name="status")
        _require_text(self.currency, field_name="currency")
        if isinstance(self.amount, bool) or not isinstance(self.amount, int) or self.amount <= 0:
            _invalid("amount must be a positive integer")
        if self.subscription_id is not None:
            _require_text(self.subscription_id, field_name="subscription_id")
        if self.method is not None:
            _require_text(self.method, field_name="method")


class ProviderPaymentBatch(BaseModel):
    """Bounded, immutable result of a payment-list observation."""

    model_config = ConfigDict(frozen=True)

    payments: tuple[ProviderPaymentState, ...]


type ProviderValue = (
    ProviderSubscriptionState
    | ProviderCustomer
    | ProviderPaymentLink
    | ProviderReceipt
    | ProviderPlan
    | ProviderPaymentState
    | ProviderPaymentBatch
)


class ProviderSuccess[T: ProviderValue](BaseModel):
    model_config = ConfigDict(frozen=True)

    operation: ProviderOperationKind
    correlation_id: str
    value: T

    @override
    def model_post_init(self, __context: object) -> None:
        _require_text(self.correlation_id, field_name="correlation_id")


type ProviderResult[T: ProviderValue] = Result[ProviderSuccess[T], ProviderError]


class BillingProvider(Protocol):
    """Provider seam whose expected failures are ``returns.Result`` values."""

    async def execute(self, operation: ProviderOperation) -> ProviderResult[ProviderValue]:
        """Execute one operation without claiming cross-system atomicity."""


type InMemoryScript = dict[ProviderOperationKind, tuple[ProviderResult[ProviderValue], ...]]
