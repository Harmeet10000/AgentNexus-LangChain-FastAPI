# src/app/api/versions.py

from app.api.strict_envelope import StrictEnvelopeAPIRouter
from app.features.agent_saul import router as agent_saul_router
from app.features.auth import router as auth_router
from app.features.health import router as health_router
from app.features.ingestion import router as ingestion_router
from app.features.profile import router as profile_router
from app.features.search import router as search_router
from app.features.users import router as users_router

v1_router = StrictEnvelopeAPIRouter(prefix="/api/v1")
v1_router.include_router(auth_router)
v1_router.include_router(health_router)
v1_router.include_router(users_router)
v1_router.include_router(profile_router)
v1_router.include_router(search_router)
v1_router.include_router(ingestion_router)
v1_router.include_router(agent_saul_router)
