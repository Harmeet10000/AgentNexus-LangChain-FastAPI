"""Main API router configuration."""

from fastapi import APIRouter
from features.chat.api.routes import router as chat_router
from features.rag.api.routes import router as rag_router
from features.documents.api.routes import router as documents_router
from features.crawl.api.routes import router as crawl_router
from features.workflows.api.routes import router as workflows_router
# TODO: Import remaining feature routers as they are migrated
# from features.mcp_agents.api.routes import router as mcp_agents_router

api_router = APIRouter()

# Health check endpoint in main router
@api_router.get("/health")
async def health_check():
    """API health check endpoint."""
    return {"status": "healthy", "service": "api"}

# Include feature routers
api_router.include_router(chat_router)
api_router.include_router(rag_router)
api_router.include_router(documents_router)
api_router.include_router(crawl_router)
api_router.include_router(workflows_router)

# TODO: Include remaining routers as features are migrated
# from features.mcp_agents.api.routes import router as mcp_agents_router
