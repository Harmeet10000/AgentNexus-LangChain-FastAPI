"""Mongo-backed profile repository using the existing auth account store."""

from typing import Any, override

from returns.result import Failure, Success

from app.features.auth.public import (
    ProfileMutation,
    get_account_by_id,
    mutate_profile,
)
from app.features.profile.domain.entities.profile import Profile
from app.features.profile.domain.errors import (
    ProfileInfrastructureError,
    ProfileResult,
)
from app.features.profile.domain.repositories import ProfileRepository

from .mappers import to_profile


def _persistence_failure(error: object, *, operation: str) -> ProfileInfrastructureError:
    return ProfileInfrastructureError(
        message=getattr(error, "message", "Profile persistence failed"),
        details=getattr(error, "details", None),
        source="profile_service",
        operation=operation,
    )


class MongoProfileRepository(ProfileRepository):
    """Adapt auth's Mongo account persistence to the profile port."""

    def __init__(self, account_store: Any) -> None:
        self._account_store = account_store

    @override
    async def get_profile(self, user_id: str) -> ProfileResult[Profile | None]:
        result = await get_account_by_id(self._account_store, user_id)
        if isinstance(result, Failure):
            return Failure(_persistence_failure(result.failure(), operation="get_profile"))
        return Success(to_profile(result.unwrap()))

    @override
    async def update_profile(self, user_id: str, *, full_name: str) -> ProfileResult[Profile]:
        result = await mutate_profile(
            self._account_store,
            user_id,
            ProfileMutation(display_name=full_name),
        )
        if isinstance(result, Failure):
            return Failure(_persistence_failure(result.failure(), operation="update_profile"))
        return Success(to_profile(result.unwrap()))

    @override
    async def set_avatar_url(self, user_id: str, avatar_url: str) -> ProfileResult[None]:
        result = await mutate_profile(
            self._account_store,
            user_id,
            ProfileMutation(avatar_url=avatar_url),
        )
        if isinstance(result, Failure):
            return Failure(_persistence_failure(result.failure(), operation="upload_avatar"))
        return Success(None)
