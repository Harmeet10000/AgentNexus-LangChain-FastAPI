"""Dependency wiring for the plans feature."""

from typing import Annotated

from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.connections import get_postgres_db
from app.features.audit.repository import AuditLogRepository
from app.features.payments.clients.razorpay_client import RazorpayClient

from .repository import PlanRepository
from .service import PlanService


def get_plan_repo(
    session: Annotated[AsyncSession, Depends(get_postgres_db)],
) -> PlanRepository:
    return PlanRepository(session)


def get_audit_repo(
    session: Annotated[AsyncSession, Depends(get_postgres_db)],
) -> AuditLogRepository:
    return AuditLogRepository(session)


def get_razorpay_client() -> RazorpayClient:
    return RazorpayClient()


async def get_plan_service(
    session: Annotated[AsyncSession, Depends(get_postgres_db)],
    razorpay: Annotated[RazorpayClient, Depends(get_razorpay_client)],
) -> PlanService:
    return PlanService(
        session,
        PlanRepository(session),
        AuditLogRepository(session),
        razorpay=razorpay,
    )


PlanServiceDep = Annotated[PlanService, Depends(get_plan_service)]
