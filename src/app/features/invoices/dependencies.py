"""Dependency wiring for the invoices feature."""

from typing import Annotated

from fastapi import Depends, Request
from sqlalchemy.ext.asyncio import AsyncSession

from app.connections import get_postgres_db
from app.features.audit.repository import AuditLogRepository
from app.features.payments.repository import PaymentRepository
from app.features.plans.repository import PlanRepository
from app.features.subscriptions.repository import SubscriptionRepository
from app.shared.services.storage import StorageService

from .repository import InvoiceRepository
from .service import InvoiceService


def get_storage(request: Request) -> StorageService | None:
    return getattr(request.app.state, "object_store", None)


async def get_invoice_service(
    session: Annotated[AsyncSession, Depends(get_postgres_db)],
    storage: Annotated[StorageService | None, Depends(get_storage)],
) -> InvoiceService:
    return InvoiceService(
        session,
        InvoiceRepository(session),
        SubscriptionRepository(session),
        PlanRepository(session),
        PaymentRepository(session),
        AuditLogRepository(session),
        storage=storage,
    )


InvoiceServiceDep = Annotated[InvoiceService, Depends(get_invoice_service)]
