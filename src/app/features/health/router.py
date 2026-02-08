from fastapi import APIRouter, Request

from .handler import health_check, self_info

router = APIRouter(prefix="/api/v1/health", tags=["health"])


@router.get("/self", response_model=None)
async def get_self(request: Request) -> object:
    return await self_info(request)


@router.get("/", response_model=None)
async def get_health(request: Request) -> object:
    return await health_check(request)
