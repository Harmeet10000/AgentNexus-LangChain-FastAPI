"""Dependency wiring for the payments feature."""

from typing import Annotated

from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.connections import get_postgres_db
from app.features.audit.repository import AuditLogRepository
from app.features.billing.providers import BillingProvider
from app.features.billing.providers.dependencies import get_lifespan_billing_provider
from app.features.billing.subscriptions.coordinator import SubscriptionLifecycleCoordinator
from app.features.billing.subscriptions.repository import SqlAlchemyLifecyclePersistence

from .repository import PaymentRepository
from .service import PaymentService


def get_payment_repo(
    session: Annotated[AsyncSession, Depends(get_postgres_db)],
) -> PaymentRepository:
    return PaymentRepository(session)


def get_audit_repo(
    session: Annotated[AsyncSession, Depends(get_postgres_db)],
) -> AuditLogRepository:
    return AuditLogRepository(session)


async def get_payment_service(
    session: Annotated[AsyncSession, Depends(get_postgres_db)],
    provider: Annotated[BillingProvider, Depends(get_lifespan_billing_provider)],
) -> PaymentService:
    coordinator = SubscriptionLifecycleCoordinator(
        SqlAlchemyLifecyclePersistence(session), provider
    )
    return PaymentService(
        PaymentRepository(session),
        AuditLogRepository(session),
        lifecycle_coordinator=coordinator,
    )


PaymentServiceDep = Annotated[PaymentService, Depends(get_payment_service)]
