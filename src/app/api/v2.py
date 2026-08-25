# src/app/api/versions.py

from fastapi import APIRouter

from app.api.strict_envelope import StrictEnvelopeAPIRouter
from app.features.agent_saul.router import router as agent_saul_router
from app.features.auth.router import router as auth_router
from app.features.credits.routers.credit_admin_router import router as credit_admin_router
from app.features.credits.routers.credit_internal_router import router as credit_internal_router
from app.features.documents.router import router as documents_router
from app.features.dunning.router import router as dunning_router
from app.features.health.router import router as health_router
from app.features.invoices.router import router as invoices_router
from app.features.payments.router import router as payments_router
from app.features.plans.router import router as plans_router
from app.features.profile.router import router as profile_router
from app.features.subscriptions.router import router as subscriptions_router
from app.features.users.router import router as users_router
from app.features.webhooks.router import router as webhooks_router

billing_router = APIRouter(tags=["billing"])
for _router in (
    plans_router,
    subscriptions_router,
    payments_router,
    invoices_router,
    webhooks_router,
    dunning_router,
    credit_admin_router,
    credit_internal_router,
):
    billing_router.include_router(_router)

v2_router = StrictEnvelopeAPIRouter(prefix="/api/v2")
v2_router.include_router(auth_router)
v2_router.include_router(health_router)
v2_router.include_router(users_router)
v2_router.include_router(profile_router)
v2_router.include_router(documents_router)
v2_router.include_router(agent_saul_router)
v2_router.include_router(billing_router)
