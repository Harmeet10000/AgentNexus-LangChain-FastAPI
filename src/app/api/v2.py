# src/app/api/versions.py

from app.api.strict_envelope import StrictEnvelopeAPIRouter
from app.features.health.router import router as health_router

v2_router = StrictEnvelopeAPIRouter(prefix="/api/v2")
# v2_router.include_router(auth_router)
v2_router.include_router(health_router)
