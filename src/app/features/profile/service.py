from returns.result import Failure, Success

from app.connections.object_storage import StorageService
from app.features.auth.account_store import AuthAccount, AuthAccountStore
from app.features.auth.repository import RefreshTokenRepository
from app.observability.logging import logger, trace_layer
from app.security.passwords import hash_password, verify_password

from .dto import AvatarResponse, UpdateProfileRequest
from .exceptions import (
    ProfileAuthenticationError,
    ProfileConflictError,
    ProfileInfrastructureError,
    ProfileResult,
    ProfileStorageError,
)


class ProfileService:
    def __init__(
        self,
        account_store: AuthAccountStore,
        token_repo: RefreshTokenRepository,
        storage: StorageService | None,
    ) -> None:
        self._account_store: AuthAccountStore = account_store
        self._token_repo: RefreshTokenRepository = token_repo
        self._storage: StorageService | None = storage

    @trace_layer(layer_name="service")
    async def update_profile(
        self,
        user: AuthAccount,
        dto: UpdateProfileRequest,
    ) -> ProfileResult[AuthAccount]:
        if dto.full_name is None:
            return Success(user)
        result = await self._account_store.update_profile(user.id, full_name=dto.full_name)
        if isinstance(result, Failure):
            error = result.failure()
            return Failure(
                inner_value=ProfileInfrastructureError(
                    message=error.message,
                    details=error.details,
                    source="profile_service",
                    operation="update_profile",
                )
            )
        updated = result.unwrap()
        logger.bind(user_id=user.id).info("Profile updated")
        return Success(updated)

    @trace_layer(layer_name="service")
    async def change_password(
        self,
        user: AuthAccount,
        current_password: str,
        new_password: str,
        current_session_id: str | None,
        *,
        revoke_other_sessions: bool,
    ) -> ProfileResult[None]:
        if user.password_hash is None:
            return Failure(
                inner_value=ProfileConflictError(
                    message=(
                        "Password cannot be changed on an OAuth-only account. "
                        "Link a password via account settings."
                    ),
                    source="profile_service",
                    operation="change_password",
                )
            )
        if not verify_password(user.password_hash, current_password):
            return Failure(
                inner_value=ProfileAuthenticationError(
                    message="Current password is incorrect",
                    source="profile_service",
                )
            )
        if current_password == new_password:
            return Failure(
                inner_value=ProfileConflictError(
                    message="New password must differ from current password",
                    source="profile_service",
                    operation="change_password",
                )
            )

        saved = await self._account_store.replace_password_hash(
            user.id, hash_password(new_password)
        )
        if isinstance(saved, Failure):
            error = saved.failure()
            return Failure(
                inner_value=ProfileInfrastructureError(
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
                    inner_value=ProfileInfrastructureError(
                        message=error.message,
                        details=error.details,
                        source="profile_service",
                        operation="revoke_sessions",
                    )
                )
        logger.bind(user_id=user.id).info("Password changed")
        return Success(None)

    @trace_layer(layer_name="service")
    async def upload_avatar(
        self,
        user: AuthAccount,
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
        old_avatar = user.avatar_url
        saved = await self._account_store.set_avatar_url(user.id, public_url)
        if isinstance(saved, Failure):
            error = saved.failure()
            return Failure(
                ProfileInfrastructureError(
                    message=error.message,
                    details=error.details,
                    source="profile_service",
                    operation="upload_avatar",
                )
            )

        if old_avatar and storage.public_url:
            # Extract key from old URL and delete asynchronously
            old_key = old_avatar.removeprefix(storage.public_url + "/")
            await storage.delete_object(key=old_key)

        logger.bind(user_id=user.id).info("Avatar uploaded")
        return Success(AvatarResponse(avatar_url=public_url))
