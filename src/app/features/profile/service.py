from returns.result import Failure, Success

from app.features.auth.adapters.persistence.mongodb.account_repository import RefreshTokenRepository
from app.features.auth.adapters.persistence.mongodb.user_store import AuthAccount, AuthAccountStore
from app.observability.logging import logger, trace_layer
from app.security.passwords import hash_password, verify_password
from app.utils.storage import StorageService, uri_to_location

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
        object_uri = stored.unwrap()

        # Persist the private object reference; the response URL is signed below.
        old_avatar = user.avatar_url
        saved = await self._account_store.set_avatar_url(user.id, object_uri)
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

        if old_avatar and old_avatar.startswith("s3://"):
            old_location = uri_to_location(old_avatar)
            if isinstance(old_location, Success):
                old_key = old_location.unwrap().key
                # This key is intentionally constant per user. The new PUT
                # overwrites the same object, so deleting old_key here would
                # delete the newly uploaded avatar.
                if old_key != key:
                    await storage.delete_object(key=old_key)

        signed_url = await self.signed_avatar_url(object_uri)
        if isinstance(signed_url, Failure):
            return signed_url

        logger.bind(user_id=user.id).info("Avatar uploaded")
        return Success(AvatarResponse(avatar_url=signed_url.unwrap() or object_uri))

    async def signed_avatar_url(self, avatar_reference: str | None) -> ProfileResult[str | None]:
        """Resolve a stored private object reference to a time-limited read URL."""
        if avatar_reference is None:
            return Success(None)
        storage = self._storage
        if storage is None:
            return Success(avatar_reference)
        if not avatar_reference.startswith("s3://"):
            # Preserve pre-migration external URLs while existing records age out.
            return Success(avatar_reference)
        location = uri_to_location(avatar_reference)
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
