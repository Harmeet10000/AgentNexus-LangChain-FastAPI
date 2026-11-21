"""HTTP routes for the health feature."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Annotated  # noqa: TC003 — FastAPI resolves this metadata

from fastapi import APIRouter, Depends, Request, Response, status
from returns.result import Success

from app.config import get_settings
from app.features.health.application.dto.aggregate import (
    HealthDataDTO,  # noqa: TC001 — FastAPI response annotation
)
from app.features.health.application.dto.result import (
    DependencyHealth,
    HealthResponse,
    HealthStatus,
)
from app.features.health.application.service import (
    HealthService,  # noqa: TC001 — FastAPI dependency annotation
)
from app.shared.result.render import (
    APIResponse,  # noqa: TC001 — FastAPI response annotation
    render_result,
)

from ....adapters.outbound.mongodb.probe import check_mongodb
from ....adapters.outbound.neo4j.probe import (
    check_cognee,
    check_graphiti,
    check_neo4j,
    check_neo4j_plugins,
)
from ....adapters.outbound.postgres.probe import check_postgres
from ....adapters.outbound.redis.probe import check_redis
from ....module import get_health_service
from .build_info import resolve_git_sha
from .dto import (
    SelfInfoDTO,  # noqa: TC001 — FastAPI response annotation
)
from .presenters import present_health, present_self_info

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable
    from typing import Literal

    from app.features.health.application.dto.aggregate import HealthResultDTO

router = APIRouter(prefix="/health", tags=["health"])
deep_health_router = APIRouter(tags=["Monitoring"])
_DEEP_PROBE_TIMEOUT_S = 9.0
_DEEP_PROBE_NAMES = (
    "postgres",
    "redis",
    "mongodb",
    "neo4j",
    "neo4j-plugins",
    "graphiti",
    "cognee",
)
ALL_PROBES: tuple[Callable[[object], Awaitable[DependencyHealth]], ...] = (
    check_postgres,
    check_redis,
    check_mongodb,
    check_neo4j,
    check_neo4j_plugins,
    check_graphiti,
    check_cognee,
)


async def _run_probe(
    probe: Callable[[object], Awaitable[DependencyHealth]], app: object, component: str
) -> DependencyHealth:
    """Apply one uniform request-level timeout and fail closed."""
    try:
        async with asyncio.timeout(_DEEP_PROBE_TIMEOUT_S):
            return await probe(app)
    except TimeoutError as exc:
        exc.add_note(f"component={component}, operation=deep_probe")
        return DependencyHealth.fail(component, "probe timed out")
    except Exception as exc:  # noqa: BLE001 — readiness must fail closed
        exc.add_note(f"component={component}, operation=deep_probe")
        return DependencyHealth.fail(component, type(exc).__name__)


@deep_health_router.get("/health")
async def get_deep_health(request: Request, response: Response) -> APIResponse[HealthResponse]:
    """Deep health check that probes critical dependencies in parallel."""
    settings = get_settings()
    dependencies: list[DependencyHealth] = await asyncio.gather(
        *[
            _run_probe(probe, request.app, component)
            for probe, component in zip(ALL_PROBES, _DEEP_PROBE_NAMES, strict=True)
        ]
    )
    failed = sum(1 for dependency in dependencies if dependency.status == HealthStatus.UNHEALTHY)
    if failed >= 3:
        overall: Literal[HealthStatus.UNHEALTHY] = HealthStatus.UNHEALTHY
    elif failed >= 1:
        overall = HealthStatus.DEGRADED
    else:
        overall = HealthStatus.HEALTHY
    body = HealthResponse(
        status=overall,
        version=settings.APP_VERSION,
        git_sha=resolve_git_sha(),
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
    self_info = await service.get_self_info(
        server_name=request.app.title or "unknown",
        server_version=request.app.version or "unknown",
        client_host=request.client.host if request.client else "unknown",
    )
    return present_self_info(self_info)


@router.get("/", operation_id="health_readiness")
@router.get("/ready", operation_id="health_ready")
async def get_health(
    response: Response,
    service: Annotated[HealthService, Depends(get_health_service)],
) -> APIResponse[HealthDataDTO]:
    result: HealthResultDTO = await service.get_health()
    return present_health(result, response)
