"""Dependency wiring for the webhooks feature."""

from typing import Annotated

from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.connections import get_postgres_db
from app.features.audit.repository import AuditLogRepository
from app.features.billing.credits.dependencies import ApplyCreditsToInvoiceDep
from app.features.billing.invoices.dependencies import get_invoice_service
from app.features.billing.invoices.service import InvoiceService
from app.features.billing.payments.dependencies import get_payment_service
from app.features.billing.payments.service import PaymentService
from app.features.billing.plans.repository import PlanRepository
from app.features.billing.providers import BillingProvider
from app.features.billing.subscriptions.coordinator import SubscriptionLifecycleCoordinator
from app.features.billing.subscriptions.dependencies import get_billing_provider
from app.features.billing.subscriptions.repository import (
    SqlAlchemyLifecyclePersistence,
    SubscriptionRepository,
)

from .repository import WebhookEventRepository
from .service import WebhookService


async def get_webhook_service(
    session: Annotated[AsyncSession, Depends(get_postgres_db)],
    payment_service: Annotated[PaymentService, Depends(get_payment_service)],
    invoice_service: Annotated[InvoiceService, Depends(get_invoice_service)],
    provider: Annotated[BillingProvider, Depends(get_billing_provider)],
    credit_apply: ApplyCreditsToInvoiceDep,
) -> WebhookService:
    return WebhookService(
        webhooks=WebhookEventRepository(session),
        subscriptions=SubscriptionRepository(session),
        plans=PlanRepository(session),
        audit=AuditLogRepository(session),
        payment_service=payment_service,
        invoice_service=invoice_service,
        credit_apply=credit_apply,
        lifecycle_coordinator=SubscriptionLifecycleCoordinator(
            SqlAlchemyLifecyclePersistence(session), provider
        ),
    )


WebhookServiceDep = Annotated[WebhookService, Depends(get_webhook_service)]
