"""Application ports for capabilities outside the profile aggregate."""

from typing import Any, Protocol

from returns.result import Result


class PasswordHasher(Protocol):
    """Password verification and hashing capability."""

    def verify(self, password_hash: str, password: str) -> bool:
        """Return whether a clear-text password matches a stored hash."""
        ...

    def hash(self, password: str) -> str:
        """Hash a clear-text password for persistence."""
        ...


class SessionRevoker(Protocol):
    """Auth-owned session revocation capability."""

    async def revoke_all_user_sessions(
        self,
        *,
        user_id: str,
        except_session_id: str | None,
        reason: str,
    ) -> Result[Any, Any]:
        """Revoke all sessions except the current one when requested."""
        ...


class AvatarStorage(Protocol):
    """Object-storage capability used by avatar mutations."""

    bucket: str

    def uri_to_location(self, uri: str) -> Result[Any, Any]:
        """Parse a stored object URI at the storage adapter boundary."""
        ...

    async def put_object(
        self,
        *,
        key: str,
        data: bytes,
        content_type: str,
        metadata: dict[str, str],
    ) -> Result[str, Any]:
        ...

    async def delete_object(self, *, key: str) -> Result[None, Any]:
        ...

    async def get_signed_get_url(self, *, key: str) -> Result[str, Any]:
        ...
