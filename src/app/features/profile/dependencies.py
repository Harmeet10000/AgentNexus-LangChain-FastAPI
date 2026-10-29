"""Profile resource composition dependencies."""

from typing import TYPE_CHECKING

from fastapi import Request

from app.features.auth.dependencies import get_account_store, get_refresh_token_repository
from app.utils import ServiceUnavailableException

from .service import ProfileService

if TYPE_CHECKING:
    from app.features.auth.account_store import AuthAccountStore
    from app.utils.storage import StorageService


async def get_profile_service(request: Request) -> ProfileService:
    """Compose profile resources and short-circuit when required stores are absent."""
    state = request.app.state
    db = getattr(state, "db", None)
    if db is None:
        message = "User database is unavailable"
        raise ServiceUnavailableException(message)

    redis = getattr(state, "redis", None)
    if redis is None:
        message = "Session store is unavailable"
        raise ServiceUnavailableException(message)

    storage: StorageService | None = getattr(state, "object_store", None)
    account_store: AuthAccountStore = await get_account_store()
    token_repo = await get_refresh_token_repository(redis)
    return ProfileService(account_store, token_repo, storage)
