"""Dependency wiring for the billing feature."""

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated

from fastapi import Depends, Request

from app.connections import get_postgres_db

from .clients.razorpay_client import RazorpayClient
from .repositories import (
    AuditLogRepository,
    BillingRepositories,
    InvoiceRepository,
    PaymentRepository,
    PlanRepository,
    SubscriptionRepository,
    WebhookEventRepository,
)
from .services import (
    DunningService,
    InvoiceService,
    PaymentService,
    PlanService,
    SubscriptionService,
    WebhookService,
)

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

    from app.shared.services.storage import StorageService


async def get_billing_repos(
    session: Annotated[AsyncSession, Depends(get_postgres_db)],
) -> BillingRepositories:
    return BillingRepositories(
        session=session,
        plans=PlanRepository(session),
        subscriptions=SubscriptionRepository(session),
        payments=PaymentRepository(session),
        invoices=InvoiceRepository(session),
        webhooks=WebhookEventRepository(session),
        audit=AuditLogRepository(session),
    )


def get_razorpay_client() -> RazorpayClient:
    return RazorpayClient()


def get_storage(request: Request) -> StorageService | None:
    return getattr(request.app.state, "object_store", None)


async def get_plan_service(
    repos: Annotated[BillingRepositories, Depends(get_billing_repos)],
    razorpay: Annotated[RazorpayClient, Depends(get_razorpay_client)],
) -> PlanService:
    return PlanService(repos, razorpay=razorpay)


async def get_subscription_service(
    repos: Annotated[BillingRepositories, Depends(get_billing_repos)],
    razorpay: Annotated[RazorpayClient, Depends(get_razorpay_client)],
) -> SubscriptionService:
    return SubscriptionService(repos, razorpay=razorpay)


async def get_payment_service(
    repos: Annotated[BillingRepositories, Depends(get_billing_repos)],
    razorpay: Annotated[RazorpayClient, Depends(get_razorpay_client)],
) -> PaymentService:
    return PaymentService(repos, razorpay=razorpay)


async def get_invoice_service(
    repos: Annotated[BillingRepositories, Depends(get_billing_repos)],
    storage: Annotated[StorageService | None, Depends(get_storage)],
) -> InvoiceService:
    return InvoiceService(repos, storage=storage)


async def get_webhook_service(
    repos: Annotated[BillingRepositories, Depends(get_billing_repos)],
) -> WebhookService:
    return WebhookService(repos)


async def get_dunning_service(
    repos: Annotated[BillingRepositories, Depends(get_billing_repos)],
) -> DunningService:
    return DunningService(repos)


BillingReposDep = Annotated[BillingRepositories, Depends(get_billing_repos)]
PlanServiceDep = Annotated[PlanService, Depends(get_plan_service)]
SubscriptionServiceDep = Annotated[SubscriptionService, Depends(get_subscription_service)]
PaymentServiceDep = Annotated[PaymentService, Depends(get_payment_service)]
InvoiceServiceDep = Annotated[InvoiceService, Depends(get_invoice_service)]
WebhookServiceDep = Annotated[WebhookService, Depends(get_webhook_service)]
DunningServiceDep = Annotated[DunningService, Depends(get_dunning_service)]
RazorpayClientDep = Annotated[RazorpayClient, Depends(get_razorpay_client)]

__all__ = [
    "BillingReposDep",
    "BillingRepositories",
    "DunningServiceDep",
    "InvoiceServiceDep",
    "PaymentServiceDep",
    "PlanServiceDep",
    "RazorpayClientDep",
    "SubscriptionServiceDep",
    "WebhookServiceDep",
    "get_billing_repos",
    "get_razorpay_client",
    "get_storage",
]
