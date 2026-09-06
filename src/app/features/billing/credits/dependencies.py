"""Dependency wiring for the credits feature."""

from typing import Annotated

from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.connections import get_postgres_db
from app.features.audit.repository import AuditLogRepository

from .repositories.consumption_repository import ConsumptionRepository
from .repositories.credit_repository import CreditRepository
from .services.credit_service import CreditService


async def get_credit_service(
    session: Annotated[AsyncSession, Depends(get_postgres_db)],
) -> CreditService:
    return CreditService(
        session,
        credit_repo=CreditRepository(session),
        consumptions=ConsumptionRepository(session),
        audit=AuditLogRepository(session),
    )


CreditServiceDep = Annotated[CreditService, Depends(get_credit_service)]
