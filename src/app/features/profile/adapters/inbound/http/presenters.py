"""Present profile application values as profile HTTP response DTOs."""

from app.features.profile.adapters.inbound.http.dto import ProfileResponse
from app.features.profile.domain.entities.profile import Profile


def present_profile(profile: Profile, *, avatar_url: str | None) -> ProfileResponse:
    return ProfileResponse(
        id=profile.id,
        email=profile.email,
        full_name=profile.full_name,
        avatar_url=avatar_url,
        role=profile.role,
        is_verified=profile.is_verified,
        is_active=profile.is_active,
        created_at=profile.created_at,
    )
