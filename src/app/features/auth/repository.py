from datetime import datetime, timezone
from typing import cast

from beanie import PydanticObjectId
from beanie.operators import In, Set
from motor.motor_asyncio import AsyncIOMotorDatabase
from pydantic import BaseModel, ConfigDict
from pymongo.errors import DuplicateKeyError, PyMongoError
from redis.asyncio import Redis
from redis.exceptions import RedisError
from returns.result import Failure, Success

from app.features.auth.model import OAuthAccount, User
from app.features.auth.token_audit_log import TokenAuditLog
from app.shared.result import (
    AppResult,
    ConflictAppError,
    InfrastructureAppError,
    NotFoundAppError,
    ValidationAppError,
    app_error_to_exception,
)

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


class UserRepository:
    def __init__(self, db: AsyncIOMotorDatabase) -> None:
        self._db = db  # retained for raw Motor queries when needed

    async def find_by_id(self, user_id: str) -> User | None:
        result = await self.find_by_id_result(user_id)
        if isinstance(result, Failure):
            return None
        return result.unwrap()

    async def find_by_id_result(self, user_id: str) -> AppResult[User | None]:
        if not PydanticObjectId.is_valid(user_id):
            return Failure(
                ValidationAppError(
                    code="INVALID_USER_ID",
                    message="Invalid user identifier",
                    details={"user_id": user_id},
                    source="auth_repository",
                )
            )
        return Success(await User.get(PydanticObjectId(user_id)))

    async def find_by_email(self, email: str) -> User | None:
        result: AppResult[User | None] = await self.find_by_email_result(email=email)
        if isinstance(result, Failure):
            return None
        return result.unwrap()

    async def find_by_email_result(self, email: str) -> AppResult[User | None]:
        try:
            user = await User.find_one(User.email == email.lower())
            if user is None:
                return Failure(
                    NotFoundAppError(
                        code="USER_NOT_FOUND",
                        message="User not found with the given email",
                        details={"email": email},
                        source="auth_repository",
                    )
                )
            return Success(user)
        except PyMongoError as exc:
            return Failure(
                InfrastructureAppError(
                    code="DB_ERROR",
                    message="Database error while finding user by email",
                    details={"email": email, "error": str(exc)},
                    source="auth_repository",
                )
            )

    async def find_by_verification_token_hash(self, token_hash: str) -> User | None:
        result = await self.find_by_verification_token_hash_result(token_hash=token_hash)
        if isinstance(result, Failure):
            return None
        return result.unwrap()

    async def find_by_verification_token_hash_result(
        self,
        token_hash: str,
    ) -> AppResult[User | None]:
        try:
            user = await User.find_one(User.verification_token_hash == token_hash)
            if user is None:
                return Failure(
                    NotFoundAppError(
                        code="USER_NOT_FOUND",
                        message="User not found with the given verification token hash",
                        details={"token_hash": token_hash},
                        source="auth_repository",
                    )
                )
            return Success(user)
        except PyMongoError as exc:
            return Failure(
                InfrastructureAppError(
                    code="DB_ERROR",
                    message="Database error while finding user by verification token hash",
                    details={"token_hash": token_hash, "error": str(exc)},
                    source="auth_repository",
                )
            )

    async def find_by_reset_token_hash(self, token_hash: str) -> User | None:
        result = await self.find_by_reset_token_hash_result(token_hash=token_hash)
        if isinstance(result, Failure):
            return None
        return result.unwrap()

    async def find_by_reset_token_hash_result(
        self,
        token_hash: str,
    ) -> AppResult[User | None]:
        try:
            user = await User.find_one(User.reset_token_hash == token_hash)
            if user is None:
                return Failure(
                    NotFoundAppError(
                        code="USER_NOT_FOUND",
                        message="User not found with the given reset token hash",
                        details={"token_hash": token_hash},
                        source="auth_repository",
                    )
                )
            return Success(user)
        except PyMongoError as exc:
            return Failure(
                InfrastructureAppError(
                    code="DB_ERROR",
                    message="Database error while finding user by reset token hash",
                    details={"token_hash": token_hash, "error": str(exc)},
                    source="auth_repository",
                )
            )

    async def create(self, user: User) -> User:
        result = await self.create_result(user=user)
        if isinstance(result, Failure):
            raise app_error_to_exception(result.failure())
        return result.unwrap()

    async def create_result(self, user: User) -> AppResult[User]:
        try:
            created = await user.insert()
            return Success(created)
        except DuplicateKeyError as exc:
            return Failure(
                ConflictAppError(
                    code="USER_CONFLICT",
                    message="User with this email already exists",
                    details={"email": user.email, "error": str(exc)},
                    source="auth_repository",
                )
            )
        except PyMongoError as exc:
            return Failure(
                InfrastructureAppError(
                    code="DB_ERROR",
                    message="Database error while creating user",
                    details={"email": user.email, "error": str(exc)},
                    source="auth_repository",
                )
            )

    async def save(self, user: User) -> User:
        result = await self.save_result(user=user)
        if isinstance(result, Failure):
            raise app_error_to_exception(result.failure())
        return result.unwrap()

    async def save_result(self, user: User) -> AppResult[User]:
        try:
            user.updated_at = datetime.now(tz=timezone.utc)
            await user.save()
            return Success(user)
        except PyMongoError as exc:
            return Failure(
                InfrastructureAppError(
                    code="DB_ERROR",
                    message="Database error while saving user",
                    details={"user_id": str(user.id), "error": str(exc)},
                    source="auth_repository",
                )
            )

    async def email_exists(self, email: str) -> bool:
        result = await self.email_exists_result(email=email)
        if isinstance(result, Failure):
            return False
        return result.unwrap()

    async def email_exists_result(self, email: str) -> AppResult[bool]:
        try:
            count = await User.find(User.email == email.lower()).count()
            return Success(count > 0)
        except PyMongoError as exc:
            return Failure(
                InfrastructureAppError(
                    code="DB_ERROR",
                    message="Database error while checking email existence",
                    details={"email": email, "error": str(exc)},
                    source="auth_repository",
                )
            )

    async def find_or_create_oauth_user(
        self,
        email: str,
        provider: str,
        provider_user_id: str,
        provider_email: str | None,
        full_name: str | None,
    ) -> tuple[User, bool]:
        result = await self.find_or_create_oauth_user_result(
            email=email,
            provider=provider,
            provider_user_id=provider_user_id,
            provider_email=provider_email,
            full_name=full_name,
        )
        if isinstance(result, Failure):
            raise app_error_to_exception(result.failure())
        return result.unwrap()

    async def find_or_create_oauth_user_result(
        self,
        email: str,
        provider: str,
        provider_user_id: str,
        provider_email: str | None,
        full_name: str | None,
    ) -> AppResult[tuple[User, bool]]:
        try:
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
        except DuplicateKeyError as exc:
            return Failure(
                ConflictAppError(
                    code="OAUTH_USER_CONFLICT",
                    message="OAuth user creation failed due to a duplicate",
                    details={"email": email, "error": str(exc)},
                    source="auth_repository",
                )
            )
        except PyMongoError as exc:
            return Failure(
                InfrastructureAppError(
                    code="DB_ERROR",
                    message="Database error during OAuth user find-or-create",
                    details={"email": email, "error": str(exc)},
                    source="auth_repository",
                )
            )


class RefreshTokenRepository:
    """Redis-primary, MongoDB-audit session store."""

    def __init__(self, redis: Redis) -> None:
        self._redis = redis

    async def store_session(self, session: SessionData) -> None:
        result = await self.store_session_result(session=session)
        if isinstance(result, Failure):
            raise app_error_to_exception(result.failure())
        return result.unwrap()

    async def store_session_result(self, session: SessionData) -> AppResult[None]:
        session_key = _SESSION_KEY.format(session.session_id)
        user_key = _USER_SESSIONS_KEY.format(session.user_id)
        try:
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
        except (RedisError, PyMongoError) as exc:
            return Failure(
                InfrastructureAppError(
                    code="SESSION_STORE_FAILED",
                    message="Failed to store session",
                    details={"session_id": session.session_id, "error": str(exc)},
                    source="auth_repository",
                )
            )

    async def get_session(self, session_id: str) -> SessionData | None:
        result = await self.get_session_result(session_id=session_id)
        if isinstance(result, Failure):
            return None
        return result.unwrap()

    async def get_session_result(self, session_id: str) -> AppResult[SessionData | None]:
        try:
            raw = await self._redis.get(_SESSION_KEY.format(session_id))
            if raw is None:
                return Success(None)
            return Success(SessionData.model_validate_json(raw))
        except RedisError as exc:
            return Failure(
                InfrastructureAppError(
                    code="SESSION_GET_FAILED",
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
    ) -> None:
        result = await self.revoke_session_result(
            session_id=session_id,
            user_id=user_id,
            reason=reason,
        )
        if isinstance(result, Failure):
            raise app_error_to_exception(result.failure())
        return result.unwrap()

    async def revoke_session_result(
        self,
        session_id: str,
        user_id: str,
        reason: str = "logout",
    ) -> AppResult[None]:
        try:
            async with self._redis.pipeline(transaction=True) as pipe:
                pipe.delete(_SESSION_KEY.format(session_id))
                pipe.srem(_USER_SESSIONS_KEY.format(user_id), session_id)
                await pipe.execute()

            await TokenAuditLog.find_one(TokenAuditLog.session_id == session_id).update(
                Set(
                    {
                        "is_revoked": True,
                        "revoked_at": datetime.utcnow(),
                        "revoke_reason": reason,
                    }
                )
            )
            return Success(None)
        except (RedisError, PyMongoError) as exc:
            return Failure(
                InfrastructureAppError(
                    code="SESSION_REVOKE_FAILED",
                    message="Failed to revoke session",
                    details={"session_id": session_id, "error": str(exc)},
                    source="auth_repository",
                )
            )

    async def get_user_sessions(self, user_id: str) -> list[SessionData]:
        result = await self.get_user_sessions_result(user_id=user_id)
        if isinstance(result, Failure):
            return []
        return result.unwrap()

    async def get_user_sessions_result(self, user_id: str) -> AppResult[list[SessionData]]:
        try:
            raw_ids = cast(
                "set[str]",
                await self._redis.smembers(_USER_SESSIONS_KEY.format(user_id)),  # ty: ignore[invalid-await]
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
                await self._redis.srem(_USER_SESSIONS_KEY.format(user_id), *dead)  # ty: ignore[invalid-await]

            return Success(sessions)
        except RedisError as exc:
            return Failure(
                InfrastructureAppError(
                    code="SESSION_LIST_FAILED",
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
    ) -> None:
        result = await self.revoke_all_user_sessions_result(
            user_id=user_id,
            except_session_id=except_session_id,
            reason=reason,
        )
        if isinstance(result, Failure):
            raise app_error_to_exception(result.failure())
        return result.unwrap()

    async def revoke_all_user_sessions_result(
        self,
        user_id: str,
        except_session_id: str | None = None,
        reason: str = "revoke_all",
    ) -> AppResult[None]:
        try:
            raw_ids = cast(
                "set[str]",
                await self._redis.smembers(_USER_SESSIONS_KEY.format(user_id)),  # ty: ignore[invalid-await]
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
                        "revoked_at": datetime.utcnow(),
                        "revoke_reason": reason,
                    }
                )
            )
            return Success(None)
        except (RedisError, PyMongoError) as exc:
            return Failure(
                InfrastructureAppError(
                    code="SESSION_REVOKE_ALL_FAILED",
                    message="Failed to revoke all user sessions",
                    details={"user_id": user_id, "error": str(exc)},
                    source="auth_repository",
                )
            )
