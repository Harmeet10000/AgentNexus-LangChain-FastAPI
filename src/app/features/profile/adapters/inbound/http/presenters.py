"""Present profile application values as auth HTTP response DTOs."""

from app.features.auth.api.http.dto import UserResponse
from app.features.profile.domain.entities.profile import Profile


def present_profile(profile: Profile, *, avatar_url: str | None) -> UserResponse:
    return UserResponse(
        id=profile.id,
        email=profile.email,
        full_name=profile.full_name,
        avatar_url=avatar_url,
        role=profile.role,
        is_verified=profile.is_verified,
        is_active=profile.is_active,
        created_at=profile.created_at,
    )
