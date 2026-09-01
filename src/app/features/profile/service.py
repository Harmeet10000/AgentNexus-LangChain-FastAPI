from returns.result import Failure, Success

from app.features.auth import (
    RefreshTokenRepository,
    User,
    UserRepository,
    hash_password,
    verify_password,
)
from app.shared.services.storage import StorageService
from app.utils import logger

from .dto import AvatarResponse, UpdateProfileRequest
from .errors import (
    ProfileAuthenticationError,
    ProfileConflictError,
    ProfileInfrastructureError,
    ProfileResult,
    ProfileStorageError,
)


class ProfileService:
    def __init__(
        self,
        user_repo: UserRepository,
        token_repo: RefreshTokenRepository,
        storage: StorageService | None,
    ) -> None:
        self._user_repo = user_repo
        self._token_repo = token_repo
        self._storage = storage

    async def update_profile(
        self,
        user: User,
        dto: UpdateProfileRequest,
    ) -> ProfileResult[User]:
        if dto.full_name is not None:
            user.full_name = dto.full_name
        result = await self._user_repo.save(user)
        if isinstance(result, Failure):
            error = result.failure()
            return Failure(
                ProfileInfrastructureError(
                    message=error.message,
                    details=error.details,
                    source="profile_service",
                    operation="update_profile",
                )
            )
        updated = result.unwrap()
        logger.bind(user_id=str(user.id)).info("Profile updated")
        return Success(updated)

    async def change_password(
        self,
        user: User,
        current_password: str,
        new_password: str,
        current_session_id: str | None,
        *,
        revoke_other_sessions: bool,
    ) -> ProfileResult[None]:
        if user.hashed_password is None:
            return Failure(
                ProfileConflictError(
                    message=(
                        "Password cannot be changed on an OAuth-only account. "
                        "Link a password via account settings."
                    ),
                    source="profile_service",
                    operation="change_password",
                )
            )
        if not verify_password(user.hashed_password, current_password):
            return Failure(
                ProfileAuthenticationError(
                    message="Current password is incorrect",
                    source="profile_service",
                )
            )
        if current_password == new_password:
            return Failure(
                ProfileConflictError(
                    message="New password must differ from current password",
                    source="profile_service",
                    operation="change_password",
                )
            )

        user.hashed_password = hash_password(new_password)
        saved = await self._user_repo.save(user)
        if isinstance(saved, Failure):
            error = saved.failure()
            return Failure(
                ProfileInfrastructureError(
                    message=error.message,
                    details=error.details,
                    source="profile_service",
                    operation="change_password",
                )
            )

        if revoke_other_sessions:
            revoked = await self._token_repo.revoke_all_user_sessions(
                user_id=str(user.id),
                except_session_id=current_session_id,
                reason="password_change",
            )
            if isinstance(revoked, Failure):
                error = revoked.failure()
                return Failure(
                    ProfileInfrastructureError(
                        message=error.message,
                        details=error.details,
                        source="profile_service",
                        operation="revoke_sessions",
                    )
                )
        logger.bind(user_id=str(user.id)).info("Password changed")
        return Success(None)

    async def upload_avatar(
        self,
        user: User,
        file_data: bytes,
        content_type: str,
    ) -> ProfileResult[AvatarResponse]:
        # Startup publishes object_store as None when S3 access cannot be verified,
        # so answer 503 here instead of raising AttributeError on a None client.
        storage = self._storage
        if storage is None:
            return Failure(ProfileStorageError(message="Object storage is not configured"))

        key = f"avatars/{user.id}"
        stored = await storage.put_object(
            key=key,
            data=file_data,
            content_type=content_type,
            metadata={"user_id": str(user.id)},
        )
        if isinstance(stored, Failure):
            error = stored.failure()
            return Failure(ProfileStorageError(message=error.message, details=error.details))
        public_url = f"{storage.public_url}/{key}" if storage.public_url else stored.unwrap()

        # Optionally delete old avatar — best-effort, non-blocking
        old_avatar: str | None = getattr(user, "avatar_url", None)

        user.avatar_url = public_url
        await self._user_repo.save(user)

        if old_avatar and storage.public_url:
            # Extract key from old URL and delete asynchronously
            old_key = old_avatar.removeprefix(storage.public_url + "/")
            await storage.delete_object(key=old_key)

        logger.bind(user_id=str(user.id)).info("Avatar uploaded")
        return Success(AvatarResponse(avatar_url=public_url))
