"""Application ports for capabilities outside the profile aggregate."""

from typing import Any, Protocol

from returns.result import Result


class CredentialChanger(Protocol):
    """Auth-owned credential use case as consumed by profile."""

    async def change_password(self, command: Any) -> Result[None, Any]: ...


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
    ) -> Result[str, Any]: ...

    async def delete_object(self, *, key: str) -> Result[None, Any]: ...

    async def get_signed_get_url(self, *, key: str) -> Result[str, Any]: ...
