"""HTTP routes for the profile resource."""

from typing import Annotated, Any

from fastapi import APIRouter, Depends, File, Request, Response, UploadFile
from returns.result import Failure

from app.features.auth.api.http.dependencies import CurrentVerifiedUser
from app.features.profile.adapters.inbound.http.dto import (
    AvatarResponse,
    ChangePasswordRequest,
    ProfileResponse,
    UpdateProfileRequest,
)
from app.features.profile.adapters.inbound.http.presenters import present_profile
from app.features.profile.application.dto.profile import (
    ChangePasswordCommand,
    GetProfileQuery,
    UpdateProfileCommand,
    UploadAvatarCommand,
)
from app.features.profile.domain.errors import ProfileValidationError
from app.features.profile.module import ProfileApplication, build_profile_application
from app.security.tokens import TokenClaims
from app.shared.rate_limit import get_rate_limiter
from app.shared.result.render import APIResponse, http_response, render_result

# Per-IP GCRA guards, one class per sensitivity: password changes are
# auth-critical (tight, mirroring forgot-password), profile edits are routine,
# avatar uploads are write-shaped like document uploads.
limit_profile_mutation = get_rate_limiter(burst=10, rate=10, period=60, scope="profile-mutation")
limit_change_password = get_rate_limiter(burst=5, rate=5, period=300, scope="change-password")
limit_avatar_upload = get_rate_limiter(burst=5, rate=5, period=300, scope="avatar-upload")

router = APIRouter(prefix="/profile", tags=["profile"])


async def _profile_for_user(application: ProfileApplication, user_id: str) -> Any:
    return await application.queries.get_profile(GetProfileQuery(user_id=user_id))


@router.get("/")
async def get_profile(
    user: CurrentVerifiedUser,
    request: Request,
    response: Response,
) -> APIResponse[ProfileResponse]:
    application = await build_profile_application(request)
    result = await _profile_for_user(application, str(user.id))
    if isinstance(result, Failure):
        return render_result(result, response)
    profile = result.unwrap()
    avatar_result = await application.queries.signed_avatar_url(profile.avatar_url)
    if isinstance(avatar_result, Failure):
        return render_result(Failure(avatar_result.failure()), response)
    return http_response(
        "Profile retrieved",
        data=present_profile(profile, avatar_url=avatar_result.unwrap()),
    )


@router.patch("/", dependencies=[Depends(limit_profile_mutation)])
async def update_profile(
    body: UpdateProfileRequest,
    user: CurrentVerifiedUser,
    request: Request,
    response: Response,
) -> APIResponse[ProfileResponse]:
    application = await build_profile_application(request)
    result = await _profile_for_user(application, str(user.id))
    if isinstance(result, Failure):
        return render_result(result, response, message="Profile updated")
    current = result.unwrap()
    updated_result = await application.commands.update_profile(
        UpdateProfileCommand(profile=current, full_name=body.full_name)
    )
    if isinstance(updated_result, Failure):
        return render_result(updated_result, response, message="Profile updated")
    updated = updated_result.unwrap()
    avatar_result = await application.queries.signed_avatar_url(updated.avatar_url)
    if isinstance(avatar_result, Failure):
        return render_result(Failure(avatar_result.failure()), response)
    return http_response(
        "Profile updated",
        data=present_profile(updated, avatar_url=avatar_result.unwrap()),
    )


@router.post(
    "/change-password",
    response_model=APIResponse[None],
    dependencies=[Depends(limit_change_password)],
)
async def change_password(
    body: ChangePasswordRequest,
    user: CurrentVerifiedUser,
    request: Request,
    response: Response,
    # Resolve claims to get session_id for session preservation.
    claims: Annotated[
        TokenClaims,
        __import__("fastapi").Depends(
            __import__(
                "app.features.auth.api.http.dependencies", fromlist=["get_token_claims"]
            ).get_token_claims
        ),
    ],
) -> APIResponse[None]:
    application = await build_profile_application(request)
    result = await _profile_for_user(application, str(user.id))
    if isinstance(result, Failure):
        return render_result(result, response)
    change_result = await application.commands.change_password(
        ChangePasswordCommand(
            profile=result.unwrap(),
            current_password=body.current_password,
            new_password=body.new_password,
            current_session_id=claims.sid,
            revoke_other_sessions=body.revoke_other_sessions,
        )
    )
    message = (
        "Password changed. Other sessions have been revoked."
        if body.revoke_other_sessions
        else "Password changed."
    )
    return render_result(change_result, response, message=message)


@router.post("/avatar", dependencies=[Depends(limit_avatar_upload)])
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

    application = await build_profile_application(request)
    profile_result = await _profile_for_user(application, str(user.id))
    if isinstance(profile_result, Failure):
        return render_result(profile_result, response)
    contents = await file.read()
    uploaded = await application.commands.upload_avatar(
        UploadAvatarCommand(
            profile=profile_result.unwrap(),
            file_data=contents,
            content_type=file.content_type,
        ),
        signed_avatar_url=application.queries.signed_avatar_url,
    )
    if isinstance(uploaded, Failure):
        return render_result(uploaded, response, message="Avatar uploaded")
    return http_response(
        "Avatar uploaded",
        data=AvatarResponse(avatar_url=uploaded.unwrap().avatar_url),
    )
