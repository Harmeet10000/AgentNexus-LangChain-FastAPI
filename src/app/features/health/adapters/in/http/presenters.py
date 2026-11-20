"""HTTP presenters for application health results."""

from fastapi import Response

from app.kernel.result import APIResponse, http_response

from ....application.dto.aggregate import (  # ty: ignore[unresolved-import]
    HealthDataDTO,
    HealthResultDTO,
)
from .dto import SelfInfoDTO  # ty: ignore[unresolved-import]


def present_self_info(
    data: dict[str, object],
    *,
    status_code: int = 200,
) -> APIResponse[SelfInfoDTO]:
    """Render application metadata into the transport DTO and envelope."""
    return http_response(
        message="Server information retrieved",
        data=SelfInfoDTO.model_validate(data),
        status_code=status_code,
    )


def present_health(
    result: HealthResultDTO,
    response: Response,
) -> APIResponse[HealthDataDTO]:
    """Render the aggregate while preserving the existing status code."""
    response.status_code = result.status_code
    return http_response(
        message=result.message,
        data=result.data,
        status_code=result.status_code,
    )
