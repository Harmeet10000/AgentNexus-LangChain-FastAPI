"""Profile resource composition dependencies."""

from typing import TYPE_CHECKING

from fastapi import Request

from app.features.auth import get_refresh_token_repository, get_user_repository
from app.utils import ServiceUnavailableException

from .service import ProfileService

if TYPE_CHECKING:
    from app.shared.services.storage import StorageService


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
    user_repo = await get_user_repository(db)
    token_repo = await get_refresh_token_repository(redis)
    return ProfileService(user_repo, token_repo, storage)
