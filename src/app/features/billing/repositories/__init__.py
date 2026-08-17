"""Billing repository layer exports."""

from pydantic import BaseModel, ConfigDict
from sqlalchemy.ext.asyncio import AsyncSession

from .audit_repository import AuditLogRepository
from .invoice_repository import InvoiceRepository
from .payment_repository import PaymentRepository
from .plan_repository import PlanRepository
from .subscription_repository import (
    SubscriptionRepository,
    validate_transition,
)
from .webhook_repository import WebhookEventRepository


class BillingRepositories(BaseModel):
    """Shared repository context passed to billing services (single session)."""

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    session: AsyncSession
    plans: PlanRepository
    subscriptions: SubscriptionRepository
    payments: PaymentRepository
    invoices: InvoiceRepository
    webhooks: WebhookEventRepository
    audit: AuditLogRepository


__all__ = [
    "AuditLogRepository",
    "BillingRepositories",
    "InvoiceRepository",
    "PaymentRepository",
    "PlanRepository",
    "SubscriptionRepository",
    "WebhookEventRepository",
    "validate_transition",
]
