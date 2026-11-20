"""Map the auth public projection into the profile entity."""

from app.features.auth.public import AccountRead
from app.features.profile.domain.entities.profile import Profile


def to_profile(account: AccountRead) -> Profile:
    """Translate an auth account without leaking it past the adapter boundary."""
    return Profile(
        id=account.id,
        email=account.email,
        full_name=account.display_name,
        is_active=account.is_active,
        is_verified=account.is_verified,
        role=account.role,
        avatar_url=account.avatar_url,
        created_at=account.created_at,
    )
