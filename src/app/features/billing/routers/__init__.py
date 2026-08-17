"""Billing API router aggregation."""

from fastapi import APIRouter

from .admin_router import router as admin_router
from .invoices_router import router as invoices_router
from .payments_router import router as payments_router
from .plans_router import router as plans_router
from .subscriptions_router import router as subscriptions_router
from .webhooks_router import router as webhooks_router

billing_router = APIRouter(tags=["billing"])
billing_router.include_router(plans_router)
billing_router.include_router(subscriptions_router)
billing_router.include_router(payments_router)
billing_router.include_router(invoices_router)
billing_router.include_router(webhooks_router)
billing_router.include_router(admin_router)

__all__ = ["billing_router"]
