from typing import Annotated

from fastapi import APIRouter, File, Request, UploadFile
from fastapi.responses import ORJSONResponse

from app.features.auth.dependencies import (
    CurrentVerifiedUser,
    get_refresh_token_repository,
    get_user_repository,
)
from app.features.auth.dto import UserResponse
from app.features.auth.security import TokenClaims
from app.features.profile.dto import AvatarResponse, ChangePasswordRequest, UpdateProfileRequest
from app.features.profile.service import ProfileService
from app.shared.response_type import APIResponse
from app.shared.services.storage import StorageService
from app.utils import http_response
from app.utils.exceptions import ValidationException

router = APIRouter(prefix="/profile", tags=["profile"])


async def _get_profile_service(request: Request) -> ProfileService:
    """Resolve ProfileService with storage from app.state."""
    storage: StorageService = request.app.state.storage
    user_repo = await get_user_repository(request.app.state.mongodb)
    token_repo = await get_refresh_token_repository(request.app.state.redis)
    return ProfileService(user_repo, token_repo, storage)


@router.get("/", response_model=APIResponse[UserResponse])
async def get_profile(user: CurrentVerifiedUser) -> ORJSONResponse:
    result = UserResponse(
        id=str(user.id),
        email=user.email,
        full_name=user.full_name,
        role=user.role.value,
        is_verified=user.is_verified,
        is_active=user.is_active,
        created_at=user.created_at,
    )
    return http_response("Profile retrieved", data=result)


@router.patch("/", response_model=APIResponse[UserResponse])
async def update_profile(
    body: UpdateProfileRequest,
    user: CurrentVerifiedUser,
    request: Request,
) -> ORJSONResponse:
    service = await _get_profile_service(request)
    updated = await service.update_profile(user, body)
    result = UserResponse(
        id=str(updated.id),
        email=updated.email,
        full_name=updated.full_name,
        role=updated.role.value,
        is_verified=updated.is_verified,
        is_active=updated.is_active,
        created_at=updated.created_at,
    )
    return http_response("Profile updated", data=result)


@router.post("/change-password", response_model=APIResponse[None])
async def change_password(
    body: ChangePasswordRequest,
    user: CurrentVerifiedUser,
    request: Request,
    # Resolve claims to get session_id for session preservation
    claims: Annotated[
        TokenClaims,
        __import__("fastapi").Depends(
            __import__(
                "app.features.auth.dependencies", fromlist=["get_token_claims"]
            ).get_token_claims
        ),
    ],
) -> ORJSONResponse:
    service = await _get_profile_service(request)
    await service.change_password(
        user=user,
        current_password=body.current_password,
        new_password=body.new_password,
        current_session_id=claims.sid,
        revoke_other_sessions=body.revoke_other_sessions,
    )
    msg = (
        "Password changed. Other sessions have been revoked."
        if body.revoke_other_sessions
        else "Password changed."
    )
    return http_response(msg)


@router.post("/avatar", response_model=APIResponse[AvatarResponse])
async def upload_avatar(
    user: CurrentVerifiedUser,
    request: Request,
    file: Annotated[UploadFile, File()],
) -> ORJSONResponse:
    if not file.content_type:
        raise ValidationException("Content-Type header is required for file upload")

    contents = await file.read()
    service = await _get_profile_service(request)
    result = await service.upload_avatar(
        user=user,
        file_data=contents,
        content_type=file.content_type,
    )
    return http_response("Avatar uploaded", data=result)
