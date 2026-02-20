# src/app/api/versions.py
from fastapi import APIRouter

v2_router = APIRouter(prefix="/api/v2")
# v2_router.include_router(auth_router)
# v2_router.include_router(health_router)
