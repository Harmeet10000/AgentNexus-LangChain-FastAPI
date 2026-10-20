"""Dependency wiring for the subscriptions feature."""

from typing import Annotated

from fastapi import Depends, Request
from sqlalchemy.ext.asyncio import AsyncSession

from app.connections import get_postgres_db
from app.features.audit.repository import AuditLogRepository
from app.features.billing.credits.dependencies import get_credit_service
from app.features.billing.credits.service import CreditService
from app.features.billing.payments.dependencies import get_payment_service
from app.features.billing.payments.repository import PaymentRepository
from app.features.billing.payments.service import PaymentService
from app.features.billing.plans.repository import PlanRepository
from app.features.billing.providers import BillingProvider

from .coordinator import SubscriptionLifecycleCoordinator
from .repository import SqlAlchemyLifecyclePersistence, SubscriptionRepository
from .service import SubscriptionService


def get_subscription_repo(
    session: Annotated[AsyncSession, Depends(get_postgres_db)],
) -> SubscriptionRepository:
    return SubscriptionRepository(session)


def get_billing_provider(request: Request) -> BillingProvider:
    """Return the process-scoped provider owned by the application lifespan."""

    provider = getattr(request.app.state, "billing_provider", None)
    if provider is None:
        message = "Billing provider is not initialized"
        raise RuntimeError(message)
    return provider


async def get_subscription_service(
    session: Annotated[AsyncSession, Depends(get_postgres_db)],
    payment_service: Annotated[PaymentService, Depends(get_payment_service)],
    provider: Annotated[BillingProvider, Depends(get_billing_provider)],
    credit_service: Annotated[CreditService, Depends(get_credit_service)],
) -> SubscriptionService:
    audit = AuditLogRepository(session)
    lifecycle_persistence = SqlAlchemyLifecyclePersistence(session, audit=audit)
    return SubscriptionService(
        session,
        SubscriptionRepository(session),
        PlanRepository(session),
        PaymentRepository(session),
        audit,
        payment_service=payment_service,
        lifecycle_coordinator=SubscriptionLifecycleCoordinator(
            lifecycle_persistence,
            provider,
        ),
        credit_service=credit_service,
    )


SubscriptionServiceDep = Annotated[SubscriptionService, Depends(get_subscription_service)]
