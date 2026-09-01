from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, cast

from beanie import PydanticObjectId
from beanie.operators import In, Set
from motor.motor_asyncio import AsyncIOMotorDatabase
from pydantic import BaseModel, ConfigDict
from pymongo.errors import DuplicateKeyError, PyMongoError
from redis.asyncio import Redis
from redis.exceptions import RedisError
from returns.result import Failure, Success

from .errors import (
    AuthConflictError,
    AuthInfrastructureError,
    AuthNotFoundError,
    AuthResult,
    AuthValidationError,
)
from .model import OAuthAccount, User
from .token_audit_log import TokenAuditLog

if TYPE_CHECKING:
    from collections.abc import Awaitable

_SESSION_KEY = "auth:session:{}"
_USER_SESSIONS_KEY = "auth:user_sessions:{}"


class SessionData(BaseModel):
    """Redis-serializable session record. Frozen for safe pipeline use."""

    model_config = ConfigDict(frozen=True)

    session_id: str
    user_id: str
    device_id: str
    created_at: datetime
    expires_at: datetime
    ttl: int  # seconds — used for Redis SETEX
    device_name: str | None = None
    ip_address: str | None = None
    user_agent: str | None = None


async def _do_find_or_create_oauth_user(
    email: str,
    provider: str,
    provider_user_id: str,
    provider_email: str | None,
    full_name: str | None,
) -> AuthResult[tuple[User, bool]]:
    user = await User.find_one(User.email == email.lower())
    if user is not None:
        already_linked = any(
            a.provider == provider and a.provider_user_id == provider_user_id
            for a in user.oauth_accounts
        )
        if not already_linked:
            user.oauth_accounts.append(
                OAuthAccount(
                    provider=provider,
                    provider_user_id=provider_user_id,
                    provider_email=provider_email,
                )
            )
            user.is_verified = True
            await user.save()
        return Success((user, False))

    new_user = User(
        email=email.lower(),
        full_name=full_name,
        is_verified=True,
        oauth_accounts=[
            OAuthAccount(
                provider=provider,
                provider_user_id=provider_user_id,
                provider_email=provider_email,
            )
        ],
    )
    created = await new_user.insert()
    return Success((created, True))


class UserRepository:
    def __init__(self, db: AsyncIOMotorDatabase[Any]) -> None:
        self._db: AsyncIOMotorDatabase[Any] = db  # retained for raw Motor queries when needed

    @staticmethod
    async def find_by_id(user_id: str) -> AuthResult[User | None]:
        if not PydanticObjectId.is_valid(user_id):
            return Failure(
                AuthValidationError(
                    message="Invalid user identifier",
                    details={"user_id": user_id},
                    source="auth_repository",
                )
            )
        return Success(await User.get(PydanticObjectId(user_id)))

    @staticmethod
    async def find_by_email(email: str) -> AuthResult[User | None]:
        try:
            user = await User.find_one(User.email == email.lower())
            if user is None:
                return Failure(
                    AuthNotFoundError(
                        message="User not found with the given email",
                        details={"email": email},
                        source="auth_repository",
                    )
                )
            return Success(user)
        except PyMongoError as exc:
            return Failure(
                AuthInfrastructureError(
                    message="Database error while finding user by email",
                    details={"email": email, "error": str(exc)},
                    source="auth_repository",
                )
            )

    @staticmethod
    async def find_by_verification_token_hash(
        token_hash: str,
    ) -> AuthResult[User | None]:
        try:
            user = await User.find_one(User.verification_token_hash == token_hash)
            if user is None:
                return Failure(
                    AuthNotFoundError(
                        message="User not found with the given verification token hash",
                        details={"token_hash": token_hash},
                        source="auth_repository",
                    )
                )
            return Success(user)
        except PyMongoError as exc:
            return Failure(
                AuthInfrastructureError(
                    message="Database error while finding user by verification token hash",
                    details={"token_hash": token_hash, "error": str(exc)},
                    source="auth_repository",
                )
            )

    @staticmethod
    async def find_by_reset_token_hash(
        token_hash: str,
    ) -> AuthResult[User | None]:
        try:
            user = await User.find_one(User.reset_token_hash == token_hash)
            if user is None:
                return Failure(
                    AuthNotFoundError(
                        message="User not found with the given reset token hash",
                        details={"token_hash": token_hash},
                        source="auth_repository",
                    )
                )
            return Success(user)
        except PyMongoError as exc:
            return Failure(
                AuthInfrastructureError(
                    message="Database error while finding user by reset token hash",
                    details={"token_hash": token_hash, "error": str(exc)},
                    source="auth_repository",
                )
            )

    @staticmethod
    async def create(user: User) -> AuthResult[User]:
        try:
            created = await user.insert()
            return Success(created)
        except DuplicateKeyError as exc:
            return Failure(
                AuthConflictError(
                    message="User with this email already exists",
                    details={"email": user.email, "error": str(exc)},
                    source="auth_repository",
                )
            )
        except PyMongoError as exc:
            return Failure(
                AuthInfrastructureError(
                    message="Database error while creating user",
                    details={"email": user.email, "error": str(exc)},
                    source="auth_repository",
                )
            )

    @staticmethod
    async def save(user: User) -> AuthResult[User]:
        try:
            user.updated_at = datetime.now(tz=UTC)
            await user.save()
            return Success(user)
        except PyMongoError as exc:
            return Failure(
                AuthInfrastructureError(
                    message="Database error while saving user",
                    details={"user_id": str(user.id), "error": str(exc)},
                    source="auth_repository",
                )
            )

    @staticmethod
    async def email_exists(email: str) -> AuthResult[bool]:
        try:
            count = await User.find(User.email == email.lower()).count()
            return Success(count > 0)
        except PyMongoError as exc:
            return Failure(
                AuthInfrastructureError(
                    message="Database error while checking email existence",
                    details={"email": email, "error": str(exc)},
                    source="auth_repository",
                )
            )

    @staticmethod
    async def find_or_create_oauth_user(
        email: str,
        provider: str,
        provider_user_id: str,
        provider_email: str | None,
        full_name: str | None,
    ) -> AuthResult[tuple[User, bool]]:
        try:
            return await _do_find_or_create_oauth_user(
                email,
                provider,
                provider_user_id,
                provider_email,
                full_name,
            )
        except DuplicateKeyError as exc:
            return Failure(
                AuthConflictError(
                    message="OAuth user creation failed due to a duplicate",
                    details={"email": email, "error": str(exc)},
                    source="auth_repository",
                )
            )
        except PyMongoError as exc:
            return Failure(
                AuthInfrastructureError(
                    message="Database error during OAuth user find-or-create",
                    details={"email": email, "error": str(exc)},
                    source="auth_repository",
                )
            )


class RefreshTokenRepository:
    """Redis-primary, MongoDB-audit session store."""

    def __init__(self, redis: Redis) -> None:
        self._redis = redis

    async def _do_store_session(
        self,
        session: SessionData,
        session_key: str,
        user_key: str,
    ) -> AuthResult[None]:
        async with self._redis.pipeline(transaction=True) as pipe:
            pipe.setex(session_key, session.ttl, session.model_dump_json())
            pipe.sadd(user_key, session.session_id)
            pipe.expire(user_key, session.ttl)
            await pipe.execute()
        await TokenAuditLog(
            session_id=session.session_id,
            user_id=PydanticObjectId(session.user_id),
            device_id=session.device_id,
            device_name=session.device_name,
            ip_address=session.ip_address,
            user_agent=session.user_agent,
            created_at=session.created_at,
            expires_at=session.expires_at,
        ).insert()
        return Success(None)

    async def _do_get_user_sessions(self, user_id: str) -> AuthResult[list[SessionData]]:
        raw_ids: set[str] = await cast(
            "Awaitable[set[str]]",
            self._redis.smembers(_USER_SESSIONS_KEY.format(user_id)),
        )
        if not raw_ids:
            return Success([])
        sid_list = list(raw_ids)
        async with self._redis.pipeline() as pipe:
            for sid in sid_list:
                pipe.get(_SESSION_KEY.format(sid))
            results: list[bytes | None] = await pipe.execute()
        sessions: list[SessionData] = []
        dead: list[str] = []
        for sid, raw in zip(sid_list, results, strict=True):
            if raw is None:
                dead.append(sid)
            else:
                sessions.append(SessionData.model_validate_json(raw))
        if dead:
            await cast(
                "Awaitable[int]",
                self._redis.srem(_USER_SESSIONS_KEY.format(user_id), *dead),
            )
        return Success(sessions)

    async def _do_revoke_all_user_sessions(
        self,
        user_id: str,
        except_session_id: str | None,
        reason: str,
    ) -> AuthResult[None]:
        raw_ids: set[str] = await cast(
            "Awaitable[set[str]]",
            self._redis.smembers(_USER_SESSIONS_KEY.format(user_id)),
        )
        to_revoke = [sid for sid in raw_ids if sid != except_session_id]
        if not to_revoke:
            return Success(None)
        async with self._redis.pipeline(transaction=True) as pipe:
            for sid in to_revoke:
                pipe.delete(_SESSION_KEY.format(sid))
            if except_session_id:
                for sid in to_revoke:
                    pipe.srem(_USER_SESSIONS_KEY.format(user_id), sid)
            else:
                pipe.delete(_USER_SESSIONS_KEY.format(user_id))
            await pipe.execute()
        await TokenAuditLog.find(In(TokenAuditLog.session_id, to_revoke)).update(
            Set(
                {
                    "is_revoked": True,
                    "revoked_at": datetime.now(UTC),
                    "revoke_reason": reason,
                }
            )
        )
        return Success(None)

    async def store_session(self, session: SessionData) -> AuthResult[None]:
        session_key = _SESSION_KEY.format(session.session_id)
        user_key = _USER_SESSIONS_KEY.format(session.user_id)
        try:
            return await self._do_store_session(session, session_key, user_key)
        except (RedisError, PyMongoError) as exc:
            return Failure(
                AuthInfrastructureError(
                    message="Failed to store session",
                    details={"session_id": session.session_id, "error": str(exc)},
                    source="auth_repository",
                )
            )

    async def get_session(self, session_id: str) -> AuthResult[SessionData | None]:
        try:
            raw = await self._redis.get(_SESSION_KEY.format(session_id))
            if raw is None:
                return Success(None)
            return Success(SessionData.model_validate_json(raw))
        except RedisError as exc:
            return Failure(
                AuthInfrastructureError(
                    message="Failed to retrieve session",
                    details={"session_id": session_id, "error": str(exc)},
                    source="auth_repository",
                )
            )

    async def revoke_session(
        self,
        session_id: str,
        user_id: str,
        reason: str = "logout",
    ) -> AuthResult[None]:
        try:
            async with self._redis.pipeline(transaction=True) as pipe:
                pipe.delete(_SESSION_KEY.format(session_id))
                pipe.srem(_USER_SESSIONS_KEY.format(user_id), session_id)
                await pipe.execute()

            await TokenAuditLog.find_one(TokenAuditLog.session_id == session_id).update(
                Set(
                    {
                        "is_revoked": True,
                        "revoked_at": datetime.now(UTC),
                        "revoke_reason": reason,
                    }
                )
            )
            return Success(None)
        except (RedisError, PyMongoError) as exc:
            return Failure(
                AuthInfrastructureError(
                    message="Failed to revoke session",
                    details={"session_id": session_id, "error": str(exc)},
                    source="auth_repository",
                )
            )

    async def get_user_sessions(self, user_id: str) -> AuthResult[list[SessionData]]:
        try:
            return await self._do_get_user_sessions(user_id)
        except RedisError as exc:
            return Failure(
                AuthInfrastructureError(
                    message="Failed to list user sessions",
                    details={"user_id": user_id, "error": str(exc)},
                    source="auth_repository",
                )
            )

    async def revoke_all_user_sessions(
        self,
        user_id: str,
        except_session_id: str | None = None,
        reason: str = "revoke_all",
    ) -> AuthResult[None]:
        try:
            return await self._do_revoke_all_user_sessions(user_id, except_session_id, reason)
        except (RedisError, PyMongoError) as exc:
            return Failure(
                AuthInfrastructureError(
                    message="Failed to revoke all user sessions",
                    details={"user_id": user_id, "error": str(exc)},
                    source="auth_repository",
                )
            )
