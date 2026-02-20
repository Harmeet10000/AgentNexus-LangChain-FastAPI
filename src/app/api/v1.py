# src/app/api/versions.py
from fastapi import APIRouter

from app.features.auth.router import router as auth_router
from app.features.health.router import router as health_router

v1_router = APIRouter(prefix="/api/v1")
v1_router.include_router(auth_router)
v1_router.include_router(health_router)
