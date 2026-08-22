"""Credits routers."""

from .credit_admin_router import router as admin_router
from .credit_internal_router import router as internal_router

__all__ = ["admin_router", "internal_router"]
