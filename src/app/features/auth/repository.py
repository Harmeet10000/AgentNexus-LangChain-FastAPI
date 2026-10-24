import secrets
from datetime import UTC, datetime, timedelta
from hashlib import sha256
from typing import TYPE_CHECKING, Any, cast

from beanie import PydanticObjectId
from beanie.operators import In, Set
from motor.motor_asyncio import AsyncIOMotorDatabase
from pydantic import BaseModel, ConfigDict, ValidationError
from pymongo.errors import DuplicateKeyError, PyMongoError
from redis.asyncio import Redis
from redis.exceptions import RedisError
from returns.result import Failure, Success

from app.observability.logging import logger, trace_layer

from .exceptions import (
    AuthAuthenticationError,
    AuthConflictError,
    AuthInfrastructureError,
    AuthNotFoundError,
    AuthResult,
    AuthValidationError,
)
from .models import OAuthAccount, User
from .token_audit_log import TokenAuditLog

if TYPE_CHECKING:
    from collections.abc import Awaitable

_SESSION_KEY = "auth:session:{}"
_USER_SESSIONS_KEY = "auth:user_sessions:{}"
_USER_AUTHZ_REVISION_KEY = "auth:user_authz_revision:{}"
_OAUTH_LINK_STATE_KEY = "auth:oauth_link_state:{}"
_OAUTH_LINK_ACTION_KEY = "auth:oauth_link_action:{}"
_OAUTH_LINK_COMPLETE_KEY = "auth:oauth_link_completed:{}"

_STORE_SESSION_SCRIPT = """
local current_revision = redis.call('GET', KEYS[3])
local expected_revision = tonumber(ARGV[1])
if current_revision and tonumber(current_revision) > expected_revision then
    return 0
end
if not current_revision or expected_revision > tonumber(current_revision) then
    redis.call('SET', KEYS[3], ARGV[1])
end
redis.call('SETEX', KEYS[1], ARGV[2], ARGV[3])
redis.call('SADD', KEYS[2], ARGV[4])
local current_ttl = redis.call('TTL', KEYS[2])
if current_ttl < tonumber(ARGV[2]) then
    redis.call('EXPIRE', KEYS[2], ARGV[2])
end
return 1
"""

_REVOKE_ALL_SESSIONS_SCRIPT = """
if ARGV[3] ~= '' then
    local current_revision = redis.call('GET', KEYS[2])
    if current_revision and tonumber(current_revision) > tonumber(ARGV[3]) then
        return {}
    end
    redis.call('SET', KEYS[2], ARGV[3])
end
local session_ids = redis.call('SMEMBERS', KEYS[1])
local revoked = {}
for _, session_id in ipairs(session_ids) do
    if ARGV[1] == '' or session_id ~= ARGV[1] then
        redis.call('DEL', ARGV[2] .. session_id)
        redis.call('SREM', KEYS[1], session_id)
        table.insert(revoked, session_id)
    end
end
if ARGV[1] == '' then
    redis.call('DEL', KEYS[1])
end
return revoked
"""


def _log_audit_failure(
    exc: PyMongoError,
    *,
    operation: str,
    session_ids: list[str],
    user_id: str,
    reason: str | None = None,
) -> None:
    """Record a repairable audit gap after Redis has already committed."""
    exc.add_note(f"auth_repository operation={operation}")
    logger.bind(
        operation=operation,
        user_id=user_id,
        session_ids=session_ids,
        reason=reason,
        error=str(exc),
        repair_required=True,
    ).exception("auth_session_audit_failed")


class SessionData(BaseModel):
    """Redis-serializable session record. Frozen for safe pipeline use."""

    model_config = ConfigDict(frozen=True)

    session_id: str
    user_id: str
    device_id: str
    created_at: datetime
    expires_at: datetime
    ttl: int  # seconds — used for Redis SETEX
    authz_revision: int = 0
    device_name: str | None = None
    ip_address: str | None = None
    user_agent: str | None = None


class OAuthLinkIntent(BaseModel):
    """Short-lived, Redis-backed account-link transaction.

    The raw OAuth state and action token are never persisted.  The state key
    and action key are SHA-256 digests, while this value binds the provider
    callback to the authenticated session and authorization generation.
    """

    model_config = ConfigDict(frozen=True)

    state_digest: str
    user_id: str
    session_id: str
    provider: str
    # The local email at link start is part of the transaction binding.  It
    # prevents an account-email change between callback and confirmation from
    # silently changing which local identity the provider assertion applies to.
    account_email: str | None = None
    authz_revision: int
    created_at: datetime
    expires_at: datetime
    action_digest: str | None = None
    provider_user_id: str | None = None
    provider_email: str | None = None
    full_name: str | None = None

    @property
    def ready_for_confirmation(self) -> bool:
        return self.action_digest is not None and self.provider_user_id is not None


async def _do_find_or_create_oauth_user(
    email: str,
    provider: str,
    provider_user_id: str,
    provider_email: str | None,
    full_name: str | None,
) -> AuthResult[tuple[User, bool]]:
    user = await User.find_one(User.email == email.lower())
    if user is not None:
        if not user.is_verified:
            return Failure(
                AuthAuthenticationError(
                    message="Local account email is not verified",
                    source="auth_repository",
                )
            )
        already_linked = any(
            a.provider == provider and a.provider_user_id == provider_user_id
            for a in user.oauth_accounts
        )
        if not already_linked:
            return Failure(
                AuthConflictError(
                    message="OAuth identity must be explicitly linked",
                    source="auth_repository",
                )
            )
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
    @trace_layer("repository")
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
    @trace_layer("repository")
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
            exc.add_note("auth_repository operation=find_by_email")
            return Failure(
                AuthInfrastructureError(
                    message="Database error while finding user by email",
                    details={"email": email, "error": str(exc)},
                    source="auth_repository",
                )
            )

    @staticmethod
    @trace_layer("repository")
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
            exc.add_note("auth_repository operation=find_by_verification_token_hash")
            return Failure(
                AuthInfrastructureError(
                    message="Database error while finding user by verification token hash",
                    details={"token_hash": token_hash, "error": str(exc)},
                    source="auth_repository",
                )
            )

    @staticmethod
    @trace_layer("repository")
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
            exc.add_note("auth_repository operation=find_by_reset_token_hash")
            return Failure(
                AuthInfrastructureError(
                    message="Database error while finding user by reset token hash",
                    details={"token_hash": token_hash, "error": str(exc)},
                    source="auth_repository",
                )
            )

    @staticmethod
    @trace_layer("repository")
    async def create(user: User) -> AuthResult[User]:
        try:
            created = await user.insert()
            return Success(created)
        except DuplicateKeyError as exc:
            exc.add_note("auth_repository operation=create duplicate_email")
            return Failure(
                AuthConflictError(
                    message="User with this email already exists",
                    details={"email": user.email, "error": str(exc)},
                    source="auth_repository",
                )
            )
        except PyMongoError as exc:
            exc.add_note("auth_repository operation=create")
            return Failure(
                AuthInfrastructureError(
                    message="Database error while creating user",
                    details={"email": user.email, "error": str(exc)},
                    source="auth_repository",
                )
            )

    @staticmethod
    @trace_layer("repository")
    async def save(user: User) -> AuthResult[User]:
        try:
            user.updated_at = datetime.now(tz=UTC)
            await user.save()
            return Success(user)
        except PyMongoError as exc:
            exc.add_note("auth_repository operation=save")
            return Failure(
                AuthInfrastructureError(
                    message="Database error while saving user",
                    details={"user_id": str(user.id), "error": str(exc)},
                    source="auth_repository",
                )
            )

    @staticmethod
    @trace_layer("repository")
    async def email_exists(email: str) -> AuthResult[bool]:
        try:
            count = await User.find(User.email == email.lower()).count()
            return Success(count > 0)
        except PyMongoError as exc:
            exc.add_note("auth_repository operation=email_exists")
            return Failure(
                AuthInfrastructureError(
                    message="Database error while checking email existence",
                    details={"email": email, "error": str(exc)},
                    source="auth_repository",
                )
            )

    @staticmethod
    @trace_layer("repository")
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
            exc.add_note("auth_repository operation=find_or_create_oauth_user duplicate")
            return Failure(
                AuthConflictError(
                    message="OAuth user creation failed due to a duplicate",
                    details={"email": email, "error": str(exc)},
                    source="auth_repository",
                )
            )
        except PyMongoError as exc:
            exc.add_note("auth_repository operation=find_or_create_oauth_user")
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

    @staticmethod
    def _digest(value: str) -> str:
        return sha256(value.encode()).hexdigest()

    async def create_oauth_link_intent(
        self,
        *,
        user_id: str,
        session_id: str,
        provider: str,
        authz_revision: int,
        state: str,
        account_email: str | None = None,
        ttl: int = 300,
    ) -> AuthResult[None]:
        now = datetime.now(UTC)
        intent = OAuthLinkIntent(
            state_digest=self._digest(state),
            user_id=user_id,
            session_id=session_id,
            provider=provider,
            account_email=account_email.lower() if account_email is not None else None,
            authz_revision=authz_revision,
            created_at=now,
            expires_at=now + timedelta(seconds=ttl),
        )
        try:
            created = await self._redis.set(
                _OAUTH_LINK_STATE_KEY.format(intent.state_digest),
                intent.model_dump_json(),
                ex=ttl,
                nx=True,
            )
            if not created:
                return Failure(
                    AuthConflictError(
                        message="OAuth link transaction already exists",
                        source="auth_repository",
                    )
                )
            return Success(None)
        except RedisError as exc:
            exc.add_note("auth_repository operation=create_oauth_link_intent")
            return Failure(
                AuthInfrastructureError(
                    message="Failed to start OAuth account linking",
                    details={"operation": "create_oauth_link_intent", "error": str(exc)},
                    source="auth_repository",
                )
            )

    async def get_oauth_link_intent(self, state: str) -> AuthResult[OAuthLinkIntent | None]:
        try:
            raw = await self._redis.get(_OAUTH_LINK_STATE_KEY.format(self._digest(state)))
            if raw is None:
                return Success(None)
            return Success(OAuthLinkIntent.model_validate_json(raw))
        except (RedisError, ValidationError) as exc:
            exc.add_note("auth_repository operation=get_oauth_link_intent")
            return Failure(
                AuthInfrastructureError(
                    message="Failed to retrieve OAuth link transaction",
                    details={"operation": "get_oauth_link_intent", "error": str(exc)},
                    source="auth_repository",
                )
            )

    async def complete_oauth_link_intent(
        self,
        *,
        state: str,
        provider_user_id: str,
        provider_email: str,
        full_name: str | None,
        ttl: int = 300,
    ) -> AuthResult[tuple[OAuthLinkIntent, str]]:
        intent_result = await self.get_oauth_link_intent(state)
        if isinstance(intent_result, Failure):
            return Failure(intent_result.failure())
        intent = intent_result.unwrap()
        if intent is None or intent.ready_for_confirmation:
            return Failure(
                AuthAuthenticationError(
                    message="OAuth link transaction is invalid or already completed",
                    source="auth_repository",
                )
            )
        completion_key = _OAUTH_LINK_COMPLETE_KEY.format(intent.state_digest)
        try:
            claimed = await self._redis.set(completion_key, "1", ex=ttl, nx=True)
        except RedisError as exc:
            exc.add_note("auth_repository operation=claim_oauth_link_intent")
            return Failure(
                AuthInfrastructureError(
                    message="Failed to claim OAuth link transaction",
                    details={"operation": "claim_oauth_link_intent", "error": str(exc)},
                    source="auth_repository",
                )
            )
        if not claimed:
            return Failure(
                AuthAuthenticationError(
                    message="OAuth link transaction is invalid or already completed",
                    source="auth_repository",
                )
            )
        action_token = secrets.token_urlsafe(32)
        action_digest = self._digest(action_token)
        completed = intent.model_copy(
            update={
                "action_digest": action_digest,
                "provider_user_id": provider_user_id,
                "provider_email": provider_email,
                "full_name": full_name,
            }
        )
        try:
            await self._redis.set(
                _OAUTH_LINK_STATE_KEY.format(intent.state_digest),
                completed.model_dump_json(),
                ex=ttl,
            )
            await self._redis.set(
                _OAUTH_LINK_ACTION_KEY.format(action_digest),
                completed.model_dump_json(),
                ex=ttl,
            )
            return Success((completed, action_token))
        except RedisError as exc:
            try:
                await self._redis.delete(completion_key)
            except RedisError:
                logger.bind(operation="complete_oauth_link_intent").warning(
                    "oauth_link_completion_claim_cleanup_failed"
                )
            exc.add_note("auth_repository operation=complete_oauth_link_intent")
            return Failure(
                AuthInfrastructureError(
                    message="Failed to prepare OAuth account link confirmation",
                    details={"operation": "complete_oauth_link_intent", "error": str(exc)},
                    source="auth_repository",
                )
            )

    async def get_oauth_link_candidate(
        self, action_token: str
    ) -> AuthResult[OAuthLinkIntent | None]:
        try:
            raw = await self._redis.get(_OAUTH_LINK_ACTION_KEY.format(self._digest(action_token)))
            if raw is None:
                return Success(None)
            return Success(OAuthLinkIntent.model_validate_json(raw))
        except (RedisError, ValidationError) as exc:
            exc.add_note("auth_repository operation=get_oauth_link_candidate")
            return Failure(
                AuthInfrastructureError(
                    message="Failed to retrieve OAuth link confirmation",
                    details={"operation": "get_oauth_link_candidate", "error": str(exc)},
                    source="auth_repository",
                )
            )

    async def consume_oauth_link_candidate(
        self, action_token: str
    ) -> AuthResult[OAuthLinkIntent | None]:
        """Atomically consume a confirmation token; the first confirmer wins."""
        key = _OAUTH_LINK_ACTION_KEY.format(self._digest(action_token))
        try:
            raw = await self._redis.getdel(key)
            if raw is None:
                return Success(None)
            return Success(OAuthLinkIntent.model_validate_json(raw))
        except (RedisError, ValidationError) as exc:
            exc.add_note("auth_repository operation=consume_oauth_link_candidate")
            return Failure(
                AuthInfrastructureError(
                    message="Failed to consume OAuth link confirmation",
                    details={"operation": "consume_oauth_link_candidate", "error": str(exc)},
                    source="auth_repository",
                )
            )

    async def _do_store_session(
        self,
        session: SessionData,
        session_key: str,
        user_key: str,
    ) -> AuthResult[None]:
        stored = await self._redis.eval(
            _STORE_SESSION_SCRIPT,
            3,
            session_key,
            user_key,
            _USER_AUTHZ_REVISION_KEY.format(session.user_id),
            session.authz_revision,
            session.ttl,
            session.model_dump_json(),
            session.session_id,
        )
        if int(stored) != 1:
            return Failure(
                AuthAuthenticationError(
                    message="Authorization changed during session creation",
                    source="auth_repository",
                )
            )
        try:
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
        except PyMongoError as exc:
            _log_audit_failure(
                exc,
                operation="store_session_audit",
                session_ids=[session.session_id],
                user_id=session.user_id,
            )
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
            await self._redis.srem(_USER_SESSIONS_KEY.format(user_id), *dead)
        return Success(sessions)

    async def _do_revoke_all_user_sessions(
        self,
        user_id: str,
        except_session_id: str | None,
        reason: str,
        authz_revision: int | None,
    ) -> AuthResult[list[str]]:
        raw_ids = await self._redis.eval(
            _REVOKE_ALL_SESSIONS_SCRIPT,
            2,
            _USER_SESSIONS_KEY.format(user_id),
            _USER_AUTHZ_REVISION_KEY.format(user_id),
            except_session_id or "",
            "auth:session:",
            "" if authz_revision is None else authz_revision,
        )
        revoked_ids = [item.decode() if isinstance(item, bytes) else str(item) for item in raw_ids]
        if revoked_ids:
            try:
                await TokenAuditLog.find(In(TokenAuditLog.session_id, revoked_ids)).update(
                    Set(
                        {
                            "is_revoked": True,
                            "revoked_at": datetime.now(UTC),
                            "revoke_reason": reason,
                        }
                    )
                )
            except PyMongoError as exc:
                _log_audit_failure(
                    exc,
                    operation="revoke_all_sessions_audit",
                    session_ids=revoked_ids,
                    user_id=user_id,
                    reason=reason,
                )
        return Success(revoked_ids)

    @trace_layer("repository")
    async def store_session(self, session: SessionData) -> AuthResult[None]:
        session_key = _SESSION_KEY.format(session.session_id)
        user_key = _USER_SESSIONS_KEY.format(session.user_id)
        try:
            return await self._do_store_session(session, session_key, user_key)
        except RedisError as exc:
            exc.add_note("auth_repository operation=store_session")
            return Failure(
                AuthInfrastructureError(
                    message="Failed to store session",
                    details={"session_id": session.session_id, "error": str(exc)},
                    source="auth_repository",
                )
            )

    @trace_layer("repository")
    async def get_session(self, session_id: str) -> AuthResult[SessionData | None]:
        try:
            raw = await self._redis.get(_SESSION_KEY.format(session_id))
            if raw is None:
                return Success(None)
            return Success(SessionData.model_validate_json(raw))
        except (RedisError, ValidationError) as exc:
            exc.add_note("auth_repository operation=get_session")
            return Failure(
                AuthInfrastructureError(
                    message="Failed to retrieve session",
                    details={"session_id": session_id, "error": str(exc)},
                    source="auth_repository",
                )
            )

    @trace_layer("repository")
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
        except RedisError as exc:
            exc.add_note("auth_repository operation=revoke_session")
            return Failure(
                AuthInfrastructureError(
                    message="Failed to revoke session",
                    details={"session_id": session_id, "error": str(exc)},
                    source="auth_repository",
                )
            )

        try:
            await TokenAuditLog.find_one(TokenAuditLog.session_id == session_id).update(
                Set(
                    {
                        "is_revoked": True,
                        "revoked_at": datetime.now(UTC),
                        "revoke_reason": reason,
                    }
                )
            )
        except PyMongoError as exc:
            _log_audit_failure(
                exc,
                operation="revoke_session_audit",
                session_ids=[session_id],
                user_id=user_id,
                reason=reason,
            )
        return Success(None)

    @trace_layer("repository")
    async def get_user_sessions(self, user_id: str) -> AuthResult[list[SessionData]]:
        try:
            return await self._do_get_user_sessions(user_id)
        except (RedisError, ValidationError) as exc:
            exc.add_note("auth_repository operation=get_user_sessions")
            return Failure(
                AuthInfrastructureError(
                    message="Failed to list user sessions",
                    details={"user_id": user_id, "error": str(exc)},
                    source="auth_repository",
                )
            )

    @trace_layer("repository")
    async def revoke_all_user_sessions(
        self,
        user_id: str,
        except_session_id: str | None = None,
        reason: str = "revoke_all",
        authz_revision: int | None = None,
    ) -> AuthResult[list[str]]:
        try:
            return await self._do_revoke_all_user_sessions(
                user_id, except_session_id, reason, authz_revision
            )
        except RedisError as exc:
            exc.add_note("auth_repository operation=revoke_all_user_sessions")
            return Failure(
                AuthInfrastructureError(
                    message="Failed to revoke all user sessions",
                    details={"user_id": user_id, "error": str(exc)},
                    source="auth_repository",
                )
            )
