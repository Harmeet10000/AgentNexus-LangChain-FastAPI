from typing import Annotated

from fastapi import APIRouter, File, Request, Response, UploadFile
from returns.result import Failure

from app.features.auth.dependencies import CurrentVerifiedUser
from app.features.auth.dto import UserResponse
from app.kernel.result import render_result
from app.security.tokens import TokenClaims
from app.utils import APIResponse, http_response

from .dependencies import get_profile_service
from .dto import (
    AvatarResponse,
    ChangePasswordRequest,
    UpdateProfileRequest,
)
from .exceptions import ProfileValidationError

router = APIRouter(prefix="/profile", tags=["profile"])


@router.get("/")
async def get_profile(
    user: CurrentVerifiedUser,
    request: Request,
    response: Response,
) -> APIResponse[UserResponse]:
    service = await get_profile_service(request)
    avatar_result = await service.signed_avatar_url(user.avatar_url)
    if isinstance(avatar_result, Failure):
        return render_result(Failure(avatar_result.failure()), response)
    result = UserResponse(
        id=str(user.id),
        email=user.email,
        full_name=user.full_name,
        avatar_url=avatar_result.unwrap(),
        role=user.role.value,
        is_verified=user.is_verified,
        is_active=user.is_active,
        created_at=user.created_at,
    )
    return http_response("Profile retrieved", data=result)


@router.patch("/")
async def update_profile(
    body: UpdateProfileRequest,
    user: CurrentVerifiedUser,
    request: Request,
    response: Response,
) -> APIResponse[UserResponse]:
    service = await get_profile_service(request)
    result = await service.update_profile(user, body)
    if isinstance(result, Failure):
        return render_result(result, response, message="Profile updated")
    updated = result.unwrap()
    avatar_result = await service.signed_avatar_url(updated.avatar_url)
    if isinstance(avatar_result, Failure):
        return render_result(Failure(avatar_result.failure()), response)
    return http_response(
        "Profile updated",
        data=UserResponse(
            id=str(updated.id),
            email=updated.email,
            full_name=updated.full_name,
            avatar_url=avatar_result.unwrap(),
            role=updated.role.value,
            is_verified=updated.is_verified,
            is_active=updated.is_active,
            created_at=updated.created_at,
        ),
    )


@router.post("/change-password", response_model=APIResponse[None])
async def change_password(
    body: ChangePasswordRequest,
    user: CurrentVerifiedUser,
    request: Request,
    response: Response,
    # Resolve claims to get session_id for session preservation
    claims: Annotated[
        TokenClaims,
        __import__("fastapi").Depends(
            __import__(
                "app.features.auth.dependencies", fromlist=["get_token_claims"]
            ).get_token_claims
        ),
    ],
) -> APIResponse[None]:
    service = await get_profile_service(request)
    result = await service.change_password(
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
    return render_result(result, response, message=msg)


@router.post("/avatar")
async def upload_avatar(
    user: CurrentVerifiedUser,
    request: Request,
    response: Response,
    file: Annotated[UploadFile, File()],
) -> APIResponse[AvatarResponse]:
    if not file.content_type:
        return render_result(
            Failure(
                ProfileValidationError(message="Content-Type header is required for file upload")
            ),
            response,
        )

    contents = await file.read()
    service = await get_profile_service(request)
    result = await service.upload_avatar(
        user=user,
        file_data=contents,
        content_type=file.content_type,
    )
    return render_result(result, response, message="Avatar uploaded")
