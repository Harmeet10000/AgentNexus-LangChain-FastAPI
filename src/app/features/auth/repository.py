from datetime import datetime

from beanie import PydanticObjectId
from beanie.operators import In, Set
from motor.motor_asyncio import AsyncIOMotorDatabase
from pydantic import BaseModel, ConfigDict
from redis.asyncio import Redis

from app.features.auth.model import OAuthAccount, User
from app.features.auth.token_audit_log import TokenAuditLog

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
        try:
            return await User.get(PydanticObjectId(user_id))
        except Exception:
            return None

    async def find_by_email(self, email: str) -> User | None:
        return await User.find_one(User.email == email.lower())

    async def find_by_verification_token_hash(self, token_hash: str) -> User | None:
        return await User.find_one(User.verification_token_hash == token_hash)

    async def find_by_reset_token_hash(self, token_hash: str) -> User | None:
        return await User.find_one(User.reset_token_hash == token_hash)

    async def create(self, user: User) -> User:
        return await user.insert()

    async def save(self, user: User) -> User:
        user.updated_at = datetime.now(tz=datetime.timezone.utc)
        await user.save()
        return user

    async def email_exists(self, email: str) -> bool:
        return await User.find(User.email == email.lower()).count() > 0

    async def find_or_create_oauth_user(
        self,
        email: str,
        provider: str,
        provider_user_id: str,
        provider_email: str | None,
        full_name: str | None,
    ) -> tuple[User, bool]:
        """Return (user, was_created). Links OAuth account if user already exists."""
        user = await self.find_by_email(email)

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
                user.is_verified = True  # OAuth email is pre-verified
                await self.save(user)
            return user, False

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
        return created, True


class RefreshTokenRepository:
    """Redis-primary, MongoDB-audit session store."""

    def __init__(self, redis: Redis) -> None:
        self._redis = redis

    async def store_session(self, session: SessionData) -> None:
        session_key = _SESSION_KEY.format(session.session_id)
        user_key = _USER_SESSIONS_KEY.format(session.user_id)

        # Atomic pipeline: session data + user index in one MULTI/EXEC
        async with self._redis.pipeline(transaction=True) as pipe:
            pipe.setex(session_key, session.ttl, session.model_dump_json())
            pipe.sadd(user_key, session.session_id)
            pipe.expire(user_key, session.ttl)
            await pipe.execute()

        # Audit log is best-effort — don't block the login response on it
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

    async def get_session(self, session_id: str) -> SessionData | None:
        raw = await self._redis.get(_SESSION_KEY.format(session_id))
        if raw is None:
            return None
        return SessionData.model_validate_json(raw)

    async def revoke_session(
        self,
        session_id: str,
        user_id: str,
        reason: str = "logout",
    ) -> None:
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

    async def get_user_sessions(self, user_id: str) -> list[SessionData]:
        """Returns only sessions still active in Redis; lazily cleans up expired IDs."""
        raw_ids: set[str] = await self._redis.smembers(_USER_SESSIONS_KEY.format(user_id))
        if not raw_ids:
            return []

        # Convert set → list for consistent ordering across the two iterations below
        sid_list = list(raw_ids)

        async with self._redis.pipeline() as pipe:
            for sid in sid_list:
                pipe.get(_SESSION_KEY.format(sid))
            results: list[bytes | None] = await pipe.execute()

        sessions: list[SessionData] = []
        dead: list[str] = []

        for sid, raw in zip(sid_list, results):
            if raw is None:
                dead.append(sid)
            else:
                sessions.append(SessionData.model_validate_json(raw))

        # Lazy cleanup: remove expired session IDs from the user index Set
        if dead:
            await self._redis.srem(_USER_SESSIONS_KEY.format(user_id), *dead)

        return sessions

    async def revoke_all_user_sessions(
        self,
        user_id: str,
        except_session_id: str | None = None,
        reason: str = "revoke_all",
    ) -> None:
        raw_ids: set[str] = await self._redis.smembers(_USER_SESSIONS_KEY.format(user_id))
        to_revoke = [sid for sid in raw_ids if sid != except_session_id]

        if not to_revoke:
            return

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
