"""Persistence ports owned by the profile domain."""

from typing import Protocol

from returns.result import Result

from .entities.profile import Profile
from .errors import ProfileError


class ProfileRepository(Protocol):
    """The profile slice's persistence boundary."""

    async def get_profile(self, user_id: str) -> Result[Profile | None, ProfileError]:
        """Load the profile projection for an authenticated account."""
        ...

    async def update_profile(
        self,
        user_id: str,
        *,
        full_name: str,
    ) -> Result[Profile, ProfileError]:
        """Persist profile-owned fields and return the refreshed projection."""
        ...

    async def replace_password_hash(
        self,
        user_id: str,
        password_hash: str,
    ) -> Result[None, ProfileError]:
        """Persist a password hash through the existing account store."""
        ...

    async def set_avatar_url(
        self,
        user_id: str,
        avatar_url: str,
    ) -> Result[None, ProfileError]:
        """Persist the private object-storage reference for an avatar."""
        ...
