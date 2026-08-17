"""Dependency wiring for the payments feature."""

from typing import Annotated

from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.connections import get_postgres_db
from app.features.audit.repository import AuditLogRepository

from .clients.razorpay_client import RazorpayClient
from .repository import PaymentRepository
from .service import PaymentService


def get_razorpay_client() -> RazorpayClient:
    return RazorpayClient()


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
    razorpay: Annotated[RazorpayClient, Depends(get_razorpay_client)],
) -> PaymentService:
    return PaymentService(
        PaymentRepository(session),
        AuditLogRepository(session),
        razorpay=razorpay,
    )


PaymentServiceDep = Annotated[PaymentService, Depends(get_payment_service)]
RazorpayClientDep = Annotated[RazorpayClient, Depends(get_razorpay_client)]
