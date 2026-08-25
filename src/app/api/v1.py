# src/app/api/versions.py

from app.api.strict_envelope import StrictEnvelopeAPIRouter
from app.features.agent_saul.router import router as agent_saul_router
from app.features.auth import router as auth_router
from app.features.documents.router import router as documents_router
from app.features.health.router import router as health_router
from app.features.profile.router import router as profile_router
from app.features.users.router import router as users_router

v1_router = StrictEnvelopeAPIRouter(prefix="/api/v1", deprecated=True)
v1_router.include_router(auth_router)
v1_router.include_router(health_router)
v1_router.include_router(users_router)
v1_router.include_router(profile_router)
v1_router.include_router(documents_router)
v1_router.include_router(agent_saul_router)
