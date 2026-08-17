# src/app/api/versions.py

from fastapi import APIRouter

from app.api.strict_envelope import StrictEnvelopeAPIRouter
from app.features.dunning.router import router as dunning_router
from app.features.health.router import router as health_router
from app.features.invoices.router import router as invoices_router
from app.features.payments.router import router as payments_router
from app.features.plans.router import router as plans_router
from app.features.subscriptions.router import router as subscriptions_router
from app.features.webhooks.router import router as webhooks_router

billing_router = APIRouter(tags=["billing"])
for _router in (
    plans_router,
    subscriptions_router,
    payments_router,
    invoices_router,
    webhooks_router,
    dunning_router,
):
    billing_router.include_router(_router)

v2_router = StrictEnvelopeAPIRouter(prefix="/api/v2")
# v2_router.include_router(auth_router)
v2_router.include_router(health_router)
v2_router.include_router(billing_router)
