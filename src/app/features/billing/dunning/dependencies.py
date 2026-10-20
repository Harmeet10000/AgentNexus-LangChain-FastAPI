"""Dependency wiring for the dunning feature."""

from typing import Annotated

from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.connections import get_postgres_db
from app.features.audit.repository import AuditLogRepository
from app.features.billing.plans.repository import PlanRepository
from app.features.billing.providers import BillingProvider
from app.features.billing.providers.dependencies import get_lifespan_billing_provider
from app.features.billing.subscriptions.coordinator import SubscriptionLifecycleCoordinator
from app.features.billing.subscriptions.repository import (
    SqlAlchemyLifecyclePersistence,
    SubscriptionRepository,
)

from .service import DunningService


async def get_dunning_service(
    session: Annotated[AsyncSession, Depends(get_postgres_db)],
    provider: Annotated[BillingProvider, Depends(get_lifespan_billing_provider)],
) -> DunningService:
    coordinator = SubscriptionLifecycleCoordinator(
        SqlAlchemyLifecyclePersistence(session), provider
    )
    return DunningService(
        session,
        subscriptions=SubscriptionRepository(session),
        plans=PlanRepository(session),
        audit=AuditLogRepository(session),
        provider=provider,
        lifecycle_coordinator=coordinator,
    )


DunningServiceDep = Annotated[DunningService, Depends(get_dunning_service)]
