from app.features.auth.model import User
from app.features.auth.repository import RefreshTokenRepository, UserRepository
from app.features.auth.security import (
    hash_password,
    verify_password,
)
from app.features.profile.dto import AvatarResponse, UpdateProfileRequest
from app.shared.services.storage import StorageService
from app.utils import ConflictException, UnauthorizedException, logger


class ProfileService:
    def __init__(
        self,
        user_repo: UserRepository,
        token_repo: RefreshTokenRepository,
        storage: StorageService,
    ) -> None:
        self._user_repo = user_repo
        self._token_repo = token_repo
        self._storage = storage

    async def update_profile(
        self,
        user: User,
        dto: UpdateProfileRequest,
    ) -> User:
        if dto.full_name is not None:
            user.full_name = dto.full_name
        updated = await self._user_repo.save(user)
        logger.bind(user_id=str(user.id)).info("Profile updated")
        return updated

    async def change_password(
        self,
        user: User,
        current_password: str,
        new_password: str,
        current_session_id: str | None,
        *,
        revoke_other_sessions: bool,
    ) -> None:
        if user.hashed_password is None:
            raise ConflictException(
                "Password cannot be changed on an OAuth-only account. "
                "Link a password via account settings."
            )
        if not verify_password(user.hashed_password, current_password):
            raise UnauthorizedException("Current password is incorrect")
        if current_password == new_password:
            raise ConflictException("New password must differ from current password")

        user.hashed_password = hash_password(new_password)
        await self._user_repo.save(user)

        if revoke_other_sessions:
            await self._token_repo.revoke_all_user_sessions(
                user_id=str(user.id),
                except_session_id=current_session_id,
                reason="password_change",
            )
        logger.bind(user_id=str(user.id)).info("Password changed")

    async def upload_avatar(
        self,
        user: User,
        file_data: bytes,
        content_type: str,
    ) -> AvatarResponse:
        # StorageService validates type and size — raises ValidationException if invalid
        public_url = await self._storage.upload_avatar(
            user_id=str(user.id),
            data=file_data,
            content_type=content_type,
        )

        # Optionally delete old avatar — best-effort, non-blocking
        old_avatar: str | None = getattr(user, "avatar_url", None)

        user.avatar_url = public_url
        await self._user_repo.save(user)

        if old_avatar:
            # Extract key from old URL and delete asynchronously
            old_key = old_avatar.removeprefix(self._storage.public_url + "/")
            await self._storage.delete_object(old_key)

        logger.bind(user_id=str(user.id)).info("Avatar uploaded")
        return AvatarResponse(avatar_url=public_url)
