"""Health feature API router."""
from fastapi import APIRouter, Depends, Request, Response

from app.shared.response_type import APIResponse
from app.utils import http_response

from .dependencies import get_health_service
from .dto import HealthDataDTO, SelfInfoDTO
from .service import HealthService

router = APIRouter(prefix="/health", tags=["health"])


@router.get("/self", response_model=APIResponse[SelfInfoDTO])
async def get_self(
    request: Request,
    service: HealthService = Depends(get_health_service),
) -> APIResponse[SelfInfoDTO]:
    self_info = await service.get_self_info(
        server_name=request.app.title or "unknown",
        server_version=request.app.version or "unknown",
        client_host=request.client.host if request.client else "unknown",
    )
    return http_response(
        message="Server information retrieved",
        data=self_info,
        status_code=200,
    )


@router.get("/", response_model=APIResponse[HealthDataDTO])
async def get_health(
    response: Response,
    service: HealthService = Depends(get_health_service),
) -> APIResponse[HealthDataDTO]:
    result = await service.get_health()
    response.status_code = result.status_code
    return http_response(
        message=result.message,
        data=result.data,
        status_code=result.status_code,
    )
