"""Dependency wiring for the credits feature."""

from typing import Annotated

from fastapi import Depends, Request
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.connections import get_postgres_db
from app.features.audit.repository import AuditLogRepository
from app.features.billing.invoices.repository import InvoiceRepository

from .apply import ApplyCreditsToInvoice
from .repository import ConsumptionRepository, CreditRepository
from .service import CreditService


async def get_credit_service(
    session: Annotated[AsyncSession, Depends(get_postgres_db)],
) -> CreditService:
    return CreditService(
        session,
        credit_repo=CreditRepository(session),
        consumptions=ConsumptionRepository(session),
        audit=AuditLogRepository(session),
    )


def get_db_session_local(request: Request) -> async_sessionmaker[AsyncSession]:
    """Return the application-owned maker, never a request-scoped session."""
    return request.app.state.db_session_local


async def get_apply_credits_operation(
    session_factory: Annotated[async_sessionmaker[AsyncSession], Depends(get_db_session_local)],
) -> ApplyCreditsToInvoice:
    """Provides a session factory so the operation can own its transaction."""
    return ApplyCreditsToInvoice(
        session_factory,
        credit_repo_factory=CreditRepository,
        consumptions_factory=ConsumptionRepository,
        invoices_factory=InvoiceRepository,
        audit_factory=AuditLogRepository,
    )


CreditServiceDep = Annotated[CreditService, Depends(get_credit_service)]
ApplyCreditsToInvoiceDep = Annotated[ApplyCreditsToInvoice, Depends(get_apply_credits_operation)]
