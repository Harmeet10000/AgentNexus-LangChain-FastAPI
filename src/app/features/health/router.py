"""Health feature API router."""

import asyncio
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Annotated

from fastapi import APIRouter, Depends, FastAPI, Request, Response, status
from returns.result import Success

from app.config import get_settings
from app.shared.result import (
    APIResponse,
    DependencyHealth,
    HealthResponse,
    HealthStatus,
    render_result,
)
from app.utils import http_response

from .dependencies import get_health_service
from .dto import HealthDataDTO, SelfInfoDTO
from .health_check import ALL_PROBES
from .service import HealthService

if TYPE_CHECKING:
    from typing import Literal

    from .dto import HealthResultDTO

router = APIRouter(prefix="/health", tags=["health"])
deep_health_router = APIRouter(tags=["Monitoring"])
_DEEP_PROBE_TIMEOUT_S = 3.0
_DEEP_PROBE_NAMES = ("postgres", "redis", "mongodb", "neo4j", "neo4j-plugins", "graphiti", "cognee")


async def _run_probe(
    probe: Callable[[FastAPI], Awaitable[DependencyHealth]], app: FastAPI, component: str
) -> DependencyHealth:
    """Run a readiness probe with one uniform request-level timeout."""
    try:
        async with asyncio.timeout(_DEEP_PROBE_TIMEOUT_S):
            return await probe(app)
    except TimeoutError:
        return DependencyHealth.fail(component, "probe timed out")
    except Exception as exc:  # noqa: BLE001 — readiness must fail closed
        return DependencyHealth.fail(component, type(exc).__name__)


@deep_health_router.get("/health")
async def get_deep_health(request: Request, response: Response) -> APIResponse[HealthResponse]:
    """Deep health check that probes all critical dependencies in parallel."""
    settings = get_settings()
    dependencies = await asyncio.gather(
        *[
            _run_probe(probe, request.app, component)
            for probe, component in zip(ALL_PROBES, _DEEP_PROBE_NAMES, strict=True)
        ]
    )

    failed = sum(1 for dependency in dependencies if dependency.status == HealthStatus.UNHEALTHY)
    if failed >= 3:
        overall: Literal[HealthStatus.UNHEALTHY] = HealthStatus.UNHEALTHY
    elif failed >= 1:
        overall: Literal[HealthStatus.DEGRADED] = HealthStatus.DEGRADED
    else:
        overall: Literal[HealthStatus.HEALTHY] = HealthStatus.HEALTHY

    body = HealthResponse(
        status=overall,
        version=settings.APP_VERSION,
        git_sha=settings.GIT_SHA,
        build_date=settings.BUILD_DATE,
        dependencies=dependencies,
    )
    code: Literal[503, 200] = (
        status.HTTP_503_SERVICE_UNAVAILABLE
        if overall == HealthStatus.UNHEALTHY
        else status.HTTP_200_OK
    )
    return render_result(Success(body), response, message="Deep health check", success_status=code)


@router.get("/self", operation_id="health_self")
@router.get("/live", operation_id="health_live")
async def get_self(
    request: Request,
    service: Annotated[HealthService, Depends(get_health_service)],
) -> APIResponse[SelfInfoDTO]:
    self_info: SelfInfoDTO = await service.get_self_info(
        server_name=request.app.title or "unknown",
        server_version=request.app.version or "unknown",
        client_host=request.client.host if request.client else "unknown",
    )
    return http_response(
        message="Server information retrieved",
        data=self_info,
        status_code=200,
    )


@router.get("/", operation_id="health_readiness")
@router.get("/ready", operation_id="health_ready")
async def get_health(
    response: Response,
    service: Annotated[HealthService, Depends(get_health_service)],
) -> APIResponse[HealthDataDTO]:
    result: HealthResultDTO = await service.get_health()
    response.status_code = result.status_code
    return http_response(
        message=result.message,
        data=result.data,
        status_code=result.status_code,
    )
