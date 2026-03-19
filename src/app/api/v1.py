# src/app/api/versions.py

from app.api.strict_envelope import StrictEnvelopeAPIRouter
from app.features.auth.router import router as auth_router

# from app.features.crawler.router import router as crawler_router
from app.features.health.router import router as health_router
from app.features.profile.router import router as profile_router

# from app.features.search.router import router as search_router
from app.features.users.router import router as users_router

# from app.features.agents.router import router as agent_router

v1_router = StrictEnvelopeAPIRouter(prefix="/api/v1")
v1_router.include_router(auth_router)
v1_router.include_router(health_router)
v1_router.include_router(users_router)
v1_router.include_router(profile_router)

# v1_router.include_router(search_router)
# v1_router.include_router(crawler_router)
# v1_router.include_router(agent_router, prefix="/agents")
