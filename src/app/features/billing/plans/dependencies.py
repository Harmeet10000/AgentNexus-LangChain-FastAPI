"""Dependency wiring for the plans feature."""

from typing import Annotated

from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.connections import get_postgres_db
from app.features.audit.module import build_audit_recorder
from app.features.audit.public import AuditRecorder
from app.features.billing.providers import BillingProvider
from app.features.billing.providers.dependencies import get_lifespan_billing_provider
from app.features.billing.subscriptions.coordinator import SubscriptionLifecycleCoordinator
from app.features.billing.subscriptions.repository import SqlAlchemyLifecyclePersistence

from .repository import PlanRepository
from .service import PlanService


def get_plan_repo(
    session: Annotated[AsyncSession, Depends(get_postgres_db)],
) -> PlanRepository:
    return PlanRepository(session)


def get_audit_repo(
    session: Annotated[AsyncSession, Depends(get_postgres_db)],
) -> AuditRecorder:
    return build_audit_recorder(session)


async def get_plan_service(
    session: Annotated[AsyncSession, Depends(get_postgres_db)],
    provider: Annotated[BillingProvider, Depends(get_lifespan_billing_provider)],
) -> PlanService:
    coordinator = SubscriptionLifecycleCoordinator(
        SqlAlchemyLifecyclePersistence(session), provider
    )
    return PlanService(
        session,
        PlanRepository(session),
        build_audit_recorder(session),
        lifecycle_coordinator=coordinator,
    )


PlanServiceDep = Annotated[PlanService, Depends(get_plan_service)]
