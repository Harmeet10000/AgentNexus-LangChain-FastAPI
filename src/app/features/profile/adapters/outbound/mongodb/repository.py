"""Mongo-backed profile repository using the existing auth account store."""

from typing import override

from returns.result import Failure, Success

from app.features.auth.adapters.persistence.mongodb.user_store import (
    AuthAccountStore,
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

    def __init__(self, account_store: AuthAccountStore) -> None:
        self._account_store = account_store

    @override
    async def get_profile(self, user_id: str) -> ProfileResult[Profile | None]:
        result = await self._account_store.find_by_id(user_id)
        if isinstance(result, Failure):
            return Failure(_persistence_failure(result.failure(), operation="get_profile"))
        account = result.unwrap()
        return Success(None if account is None else to_profile(account))

    @override
    async def update_profile(self, user_id: str, *, full_name: str) -> ProfileResult[Profile]:
        result = await self._account_store.update_profile(user_id, full_name=full_name)
        if isinstance(result, Failure):
            return Failure(_persistence_failure(result.failure(), operation="update_profile"))
        return Success(to_profile(result.unwrap()))

    @override
    async def replace_password_hash(
        self,
        user_id: str,
        password_hash: str,
    ) -> ProfileResult[None]:
        result = await self._account_store.replace_password_hash(user_id, password_hash)
        if isinstance(result, Failure):
            return Failure(
                _persistence_failure(result.failure(), operation="change_password")
            )
        return Success(None)

    @override
    async def set_avatar_url(self, user_id: str, avatar_url: str) -> ProfileResult[None]:
        result = await self._account_store.set_avatar_url(user_id, avatar_url)
        if isinstance(result, Failure):
            return Failure(_persistence_failure(result.failure(), operation="upload_avatar"))
        return Success(None)
