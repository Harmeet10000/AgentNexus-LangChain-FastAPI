"""Profile slice composition descriptor and request-scoped wiring."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from app.shared.module import Module

if TYPE_CHECKING:
    from typing import Any


@dataclass(frozen=True, slots=True)
class ProfileApplication:
    """The two use-case entry points composed for one request."""

    queries: Any
    commands: Any


async def build_profile_application(request: Any) -> ProfileApplication:
    """Compose profile adapters from the process state when HTTP needs them."""
    state = request.app.state  # type: ignore[attr-defined]
    db = getattr(state, "db", None)
    if db is None:
        from app.utils import ServiceUnavailableException  # noqa: PLC0415

        message = "User database is unavailable"
        raise ServiceUnavailableException(message)

    redis = getattr(state, "redis", None)
    if redis is None:
        from app.utils import ServiceUnavailableException  # noqa: PLC0415

        message = "Session store is unavailable"
        raise ServiceUnavailableException(message)

    from app.features.auth.api.http.dependencies import (  # noqa: PLC0415
        get_account_store,
        get_refresh_token_repository,
    )
    from app.features.profile.adapters.outbound.auth.credentials import (  # noqa: PLC0415
        AuthCredentialGateway,
    )
    from app.features.profile.adapters.outbound.mongodb.repository import (  # noqa: PLC0415
        MongoProfileRepository,
    )
    from app.features.profile.application.commands.update_profile import (  # noqa: PLC0415
        UpdateProfileCommandHandler,
    )
    from app.features.profile.application.queries.get_profile import (  # noqa: PLC0415
        GetProfileQueryHandler,
    )
    from app.observability.logging import logger  # noqa: PLC0415
    from app.security.passwords import hash_password, verify_password  # noqa: PLC0415
    from app.utils.storage import uri_to_location  # noqa: PLC0415

    account_store = await get_account_store()
    token_repo = await get_refresh_token_repository(redis)
    storage = getattr(state, "object_store", None)

    class _PasswordHasher:
        @staticmethod
        def verify(password_hash: str, password: str) -> bool:
            return verify_password(password_hash, password)

        @staticmethod
        def hash(password: str) -> str:
            return hash_password(password)

    class _AvatarStorage:
        def __init__(self, service: Any) -> None:
            self._service = service

        @property
        def bucket(self) -> str:
            return self._service.bucket

        @staticmethod
        def uri_to_location(uri: str) -> Any:
            return uri_to_location(uri)

        async def put_object(self, **kwargs: Any) -> Any:
            return await self._service.put_object(**kwargs)

        async def delete_object(self, **kwargs: Any) -> Any:
            return await self._service.delete_object(**kwargs)

        async def get_signed_get_url(self, **kwargs: Any) -> Any:
            return await self._service.get_signed_get_url(**kwargs)

    repository = MongoProfileRepository(account_store)
    storage_port = _AvatarStorage(storage) if storage is not None else None
    queries = GetProfileQueryHandler(repository, storage=storage_port)
    commands = UpdateProfileCommandHandler(
        repository,
        credentials=AuthCredentialGateway(account_store, token_repo, _PasswordHasher()),
        storage=storage_port,
        logger=logger,
    )
    return ProfileApplication(queries=queries, commands=commands)


def build_router() -> tuple[object, ...]:
    """Build only the HTTP surface when an API composition root asks for it."""
    from app.features.profile.adapters.inbound.http.router import router  # noqa: PLC0415

    return (router,)


MODULE = Module(name="profile", routes=(build_router,))
