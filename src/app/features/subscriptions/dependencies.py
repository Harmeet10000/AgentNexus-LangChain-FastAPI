"""Dependency wiring for the subscriptions feature."""

from typing import Annotated

from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.connections import get_postgres_db
from app.features.audit.repository import AuditLogRepository
from app.features.payments.clients.razorpay_client import RazorpayClient
from app.features.payments.dependencies import get_payment_service
from app.features.payments.repository import PaymentRepository
from app.features.payments.service import PaymentService
from app.features.plans.repository import PlanRepository

from .repository import SubscriptionRepository
from .service import SubscriptionService


def get_subscription_repo(
    session: Annotated[AsyncSession, Depends(get_postgres_db)],
) -> SubscriptionRepository:
    return SubscriptionRepository(session)


async def get_subscription_service(
    session: Annotated[AsyncSession, Depends(get_postgres_db)],
    payment_service: Annotated[PaymentService, Depends(get_payment_service)],
) -> SubscriptionService:
    return SubscriptionService(
        session,
        SubscriptionRepository(session),
        PlanRepository(session),
        PaymentRepository(session),
        AuditLogRepository(session),
        payment_service=payment_service,
        razorpay=RazorpayClient(),
    )


SubscriptionServiceDep = Annotated[SubscriptionService, Depends(get_subscription_service)]
