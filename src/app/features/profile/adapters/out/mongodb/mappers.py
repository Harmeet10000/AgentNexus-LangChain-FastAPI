"""Map the auth-owned Mongo projection into the profile entity."""

from app.features.auth.adapters.persistence.mongodb.user_store import AuthAccount
from app.features.profile.domain.entities.profile import Profile


def to_profile(account: AuthAccount) -> Profile:
    """Translate an auth account without leaking it past the adapter boundary."""
    role = account.role.value if hasattr(account.role, "value") else str(account.role)
    return Profile(
        id=account.id,
        email=account.email,
        full_name=account.full_name,
        password_hash=account.password_hash,
        is_active=account.is_active,
        is_verified=account.is_verified,
        role=role,
        avatar_url=account.avatar_url,
        created_at=account.created_at,
    )
