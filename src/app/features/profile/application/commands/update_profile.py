"""Profile mutation use cases."""

from typing import Any

from returns.result import Failure, Success

from app.features.profile.application.dto.profile import (
    ChangePasswordCommand,
    UpdateProfileCommand,
    UploadAvatarCommand,
    UploadedAvatar,
)
from app.features.profile.application.ports.profile import (
    AvatarStorage,
    CredentialChanger,
)
from app.features.profile.domain.entities.profile import Profile
from app.features.profile.domain.errors import (
    ProfileResult,
    ProfileStorageError,
)
from app.features.profile.domain.repositories import ProfileRepository


class UpdateProfileCommandHandler:
    """Execute profile mutations without knowing their transport or store."""

    def __init__(
        self,
        repository: ProfileRepository,
        *,
        credentials: CredentialChanger,
        storage: AvatarStorage | None,
        logger: Any,
    ) -> None:
        self._repository = repository
        self._credentials = credentials
        self._storage = storage
        self._logger = logger

    async def update_profile(self, command: UpdateProfileCommand) -> ProfileResult[Profile]:
        if command.full_name is None:
            return Success(command.profile)
        result = await self._repository.update_profile(
            command.profile.id,
            full_name=command.full_name,
        )
        if isinstance(result, Failure):
            return result
        self._logger.bind(user_id=command.profile.id).info("Profile updated")
        return result

    async def change_password(self, command: ChangePasswordCommand) -> ProfileResult[None]:
        result = await self._credentials.change_password(command)
        if isinstance(result, Failure):
            return result
        self._logger.bind(user_id=command.profile.id).info("Password changed")
        return Success(None)

    async def upload_avatar(
        self,
        command: UploadAvatarCommand,
        *,
        signed_avatar_url: Any,
    ) -> ProfileResult[UploadedAvatar]:
        storage = self._storage
        if storage is None:
            return Failure(ProfileStorageError(message="Object storage is not configured"))

        key = f"avatars/{command.profile.id}"
        stored = await storage.put_object(
            key=key,
            data=command.file_data,
            content_type=command.content_type,
            metadata={"user_id": command.profile.id},
        )
        if isinstance(stored, Failure):
            error = stored.failure()
            return Failure(ProfileStorageError(message=error.message, details=error.details))
        object_uri = stored.unwrap()

        saved = await self._repository.set_avatar_url(command.profile.id, object_uri)
        if isinstance(saved, Failure):
            return saved

        old_avatar = command.profile.avatar_url
        if old_avatar and old_avatar.startswith("s3://"):
            old_location = storage.uri_to_location(old_avatar)
            if isinstance(old_location, Success):
                old_key = old_location.unwrap().key
                if old_key != key:
                    await storage.delete_object(key=old_key)

        signed_url = await signed_avatar_url(object_uri)
        if isinstance(signed_url, Failure):
            return signed_url
        self._logger.bind(user_id=command.profile.id).info("Avatar uploaded")
        return Success(UploadedAvatar(avatar_url=signed_url.unwrap() or object_uri))
