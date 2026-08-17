"""Dependency wiring for the dunning feature."""

from typing import Annotated

from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.connections import get_postgres_db
from app.features.audit.repository import AuditLogRepository
from app.features.payments.clients.razorpay_client import RazorpayClient
from app.features.plans.repository import PlanRepository
from app.features.subscriptions.repository import SubscriptionRepository

from .service import DunningService


async def get_dunning_service(
    session: Annotated[AsyncSession, Depends(get_postgres_db)],
) -> DunningService:
    return DunningService(
        session,
        SubscriptionRepository(session),
        PlanRepository(session),
        AuditLogRepository(session),
        razorpay=RazorpayClient(),
    )


DunningServiceDep = Annotated[DunningService, Depends(get_dunning_service)]
