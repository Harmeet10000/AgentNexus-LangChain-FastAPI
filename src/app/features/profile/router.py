from typing import TYPE_CHECKING, Annotated

from fastapi import APIRouter, File, Request, UploadFile

from app.features.auth import (
    CurrentVerifiedUser,
    TokenClaims,
    UserResponse,
    get_refresh_token_repository,
    get_user_repository,
)
from app.utils import (
    APIResponse,
    ServiceUnavailableException,
    ValidationException,
    http_response,
)

from .dto import (
    AvatarResponse,
    ChangePasswordRequest,
    UpdateProfileRequest,
)
from .service import ProfileService

if TYPE_CHECKING:
    from app.shared.services.storage import StorageService

router = APIRouter(prefix="/profile", tags=["profile"])


async def _get_profile_service(request: Request) -> ProfileService:
    """Resolve ProfileService against the client names startup actually publishes.

    Startup publishes ``object_store``, ``db`` and ``redis`` on ``app.state`` — never
    ``storage`` or ``mongodb`` — and it sets each of them to ``None`` when that
    connection fails instead of aborting boot. Every read is therefore resolved
    defensively: a missing or absent client answers ``503`` from here, rather than
    becoming a ``None`` that fails deeper in the request, far from its cause.

    The object store is *not* required here. Only avatar upload uses it, so it is
    passed through as optional and its absence is answered at the point of use —
    a password change must not fail because object storage is down.
    """
    state = request.app.state

    db = getattr(state, "db", None)
    if db is None:
        msg = "User database is unavailable"
        raise ServiceUnavailableException(msg)

    redis = getattr(state, "redis", None)
    if redis is None:
        msg = "Session store is unavailable"
        raise ServiceUnavailableException(msg)

    storage: StorageService | None = getattr(state, "object_store", None)
    user_repo = await get_user_repository(db)
    token_repo = await get_refresh_token_repository(redis)
    return ProfileService(user_repo, token_repo, storage)


@router.get("/")
async def get_profile(user: CurrentVerifiedUser) -> APIResponse[UserResponse]:
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


@router.patch("/")
async def update_profile(
    body: UpdateProfileRequest,
    user: CurrentVerifiedUser,
    request: Request,
) -> APIResponse[UserResponse]:
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
) -> APIResponse[None]:
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


@router.post("/avatar")
async def upload_avatar(
    user: CurrentVerifiedUser,
    request: Request,
    file: Annotated[UploadFile, File()],
) -> APIResponse[AvatarResponse]:
    if not file.content_type:
        msg = "Content-Type header is required for file upload"
        raise ValidationException(msg)

    contents = await file.read()
    service = await _get_profile_service(request)
    result = await service.upload_avatar(
        user=user,
        file_data=contents,
        content_type=file.content_type,
    )
    return http_response("Avatar uploaded", data=result)
