"""Dependency wiring for the webhooks feature."""

from typing import Annotated

from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.connections import get_postgres_db
from app.features.audit.repository import AuditLogRepository
from app.features.invoices.dependencies import get_invoice_service
from app.features.invoices.service import InvoiceService
from app.features.payments.dependencies import get_payment_service
from app.features.payments.service import PaymentService
from app.features.plans.repository import PlanRepository
from app.features.subscriptions.repository import SubscriptionRepository

from .repository import WebhookEventRepository
from .service import WebhookService


async def get_webhook_service(
    session: Annotated[AsyncSession, Depends(get_postgres_db)],
    payment_service: Annotated[PaymentService, Depends(get_payment_service)],
    invoice_service: Annotated[InvoiceService, Depends(get_invoice_service)],
) -> WebhookService:
    return WebhookService(
        webhooks=WebhookEventRepository(session),
        subscriptions=SubscriptionRepository(session),
        plans=PlanRepository(session),
        audit=AuditLogRepository(session),
        payment_service=payment_service,
        invoice_service=invoice_service,
    )


WebhookServiceDep = Annotated[WebhookService, Depends(get_webhook_service)]
