"""Profile read use case and private-avatar URL resolution."""

from returns.result import Failure, Success

from app.features.profile.application.dto.profile import GetProfileQuery
from app.features.profile.application.ports.profile import AvatarStorage
from app.features.profile.domain.entities.profile import Profile
from app.features.profile.domain.errors import ProfileResult, ProfileStorageError
from app.features.profile.domain.repositories import ProfileRepository


class GetProfileQueryHandler:
    """Load a profile and resolve its private avatar reference for HTTP."""

    def __init__(
        self,
        repository: ProfileRepository,
        *,
        storage: AvatarStorage | None,
    ) -> None:
        self._repository = repository
        self._storage = storage

    async def get_profile(self, query: GetProfileQuery) -> ProfileResult[Profile]:
        result = await self._repository.get_profile(query.user_id)
        if isinstance(result, Failure):
            return result
        profile = result.unwrap()
        if profile is None:
            return Failure(
                ProfileStorageError(
                    message="User profile is unavailable",
                )
            )
        return Success(profile)

    async def signed_avatar_url(self, avatar_reference: str | None) -> ProfileResult[str | None]:
        """Resolve a private object reference to a time-limited read URL."""
        if avatar_reference is None:
            return Success(None)
        storage = self._storage
        if storage is None:
            return Success(avatar_reference)
        if not avatar_reference.startswith("s3://"):
            return Success(avatar_reference)

        location = storage.uri_to_location(avatar_reference)
        if isinstance(location, Failure):
            error = location.failure()
            return Failure(ProfileStorageError(message=error.message, details=error.details))
        object_location = location.unwrap()
        if object_location.bucket != storage.bucket:
            return Failure(ProfileStorageError(message="Avatar belongs to an unexpected bucket"))
        signed = await storage.get_signed_get_url(key=object_location.key)
        if isinstance(signed, Failure):
            error = signed.failure()
            return Failure(ProfileStorageError(message=error.message, details=error.details))
        return Success(signed.unwrap())
