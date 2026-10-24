import secrets
import traceback
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

from app.features.auth.websocket_revocation import new_event, publish_event
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
_REFRESH_FAMILY_KEY = "auth:refresh_family:{}"
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
        local session_key = ARGV[2] .. session_id
        local session_raw = redis.call('GET', session_key)
        if session_raw then
            local ok, session = pcall(cjson.decode, session_raw)
            if ok and type(session.family_id) == 'string' then
                local family_key = (ARGV[6] or 'auth:refresh_family:') .. session.family_id
                if redis.call('EXISTS', family_key) == 1 then
                    redis.call('HSET', family_key,
                        'status', 'revoked', 'revoke_reason', ARGV[4], 'revoked_at', ARGV[5])
                end
            end
        end
        redis.call('DEL', session_key)
        redis.call('SREM', KEYS[1], session_id)
        table.insert(revoked, session_id)
    end
end
if ARGV[1] == '' then
    redis.call('DEL', KEYS[1])
end
return revoked
"""

_CREATE_REFRESH_FAMILY_SCRIPT = """
if redis.call('EXISTS', KEYS[1]) == 1 then
    return 0
end
redis.call('HSET', KEYS[1],
    'family_id', ARGV[1],
    'user_id', ARGV[2],
    'session_id', ARGV[3],
    'device_id', ARGV[4],
    'authz_revision', ARGV[5],
    'generation', ARGV[6],
    'current_id_hash', ARGV[7],
    'current_secret_hash', ARGV[8],
    'current_parent_hash', '',
    'current_issued_at', ARGV[9],
    'expires_at', ARGV[10],
    'previous_id_hash', '',
    'previous_secret_hash', '',
    'previous_generation', '',
    'previous_grace_until', '',
    'legacy', ARGV[11],
    'session_key', ARGV[12],
    'user_sessions_key', ARGV[13],
    'status', 'active')
redis.call('EXPIREAT', KEYS[1], ARGV[10])
return 1
"""

_ATTACH_REFRESH_FAMILY_SCRIPT = """
local raw = redis.call('GET', KEYS[1])
if not raw then return 0 end
local ok, session = pcall(cjson.decode, raw)
if not ok then return 0 end
session.family_id = ARGV[1]
local encoded = cjson.encode(session)
local ttl = redis.call('TTL', KEYS[1])
if ttl > 0 then
    redis.call('SETEX', KEYS[1], ttl, encoded)
else
    redis.call('SET', KEYS[1], encoded, 'KEEPTTL')
end
return 1
"""

_ROTATE_REFRESH_TOKEN_SCRIPT = """
local status = redis.call('HGET', KEYS[1], 'status')
if not status then return {'missing'} end
if status ~= 'active' then return {'revoked'} end
if tonumber(redis.call('HGET', KEYS[1], 'authz_revision')) ~= tonumber(ARGV[8]) then
    return {'stale'}
end
local current_id = redis.call('HGET', KEYS[1], 'current_id_hash')
local current_secret = redis.call('HGET', KEYS[1], 'current_secret_hash')
local current_generation = tonumber(redis.call('HGET', KEYS[1], 'generation'))
if ARGV[9] ~= '' and tonumber(ARGV[9]) < current_generation - 1 then
    redis.call('HSET', KEYS[1], 'status', 'revoked', 'revoke_reason', 'refresh_reuse', 'revoked_at', ARGV[6])
    local session_key = redis.call('HGET', KEYS[1], 'session_key')
    local user_sessions_key = redis.call('HGET', KEYS[1], 'user_sessions_key')
    local session_id = redis.call('HGET', KEYS[1], 'session_id')
    redis.call('DEL', session_key)
    redis.call('SREM', user_sessions_key, session_id)
    return {'reused'}
end
if ARGV[1] == current_id then
    if ARGV[2] ~= current_secret then
        redis.call('HSET', KEYS[1], 'status', 'revoked', 'revoke_reason', 'refresh_secret_mismatch', 'revoked_at', ARGV[6])
        local session_key = redis.call('HGET', KEYS[1], 'session_key')
        local user_sessions_key = redis.call('HGET', KEYS[1], 'user_sessions_key')
        local session_id = redis.call('HGET', KEYS[1], 'session_id')
        redis.call('DEL', session_key)
        redis.call('SREM', user_sessions_key, session_id)
        return {'reused'}
    end
    redis.call('HSET', KEYS[1],
        'previous_id_hash', current_id,
        'previous_secret_hash', current_secret,
        'previous_generation', current_generation,
        'previous_grace_until', tostring(tonumber(ARGV[6]) + tonumber(ARGV[7])),
        'current_id_hash', ARGV[3],
        'current_secret_hash', ARGV[4],
        'current_parent_hash', ARGV[1],
        'generation', ARGV[5],
        'current_issued_at', ARGV[6])
    return {'rotated', ARGV[5]}
end
local previous_id = redis.call('HGET', KEYS[1], 'previous_id_hash')
local previous_secret = redis.call('HGET', KEYS[1], 'previous_secret_hash')
local grace_until = tonumber(redis.call('HGET', KEYS[1], 'previous_grace_until') or '0')
if tonumber(ARGV[7]) > 0 and ARGV[1] == previous_id and ARGV[2] == previous_secret and tonumber(ARGV[6]) <= grace_until then
    return {'grace', tostring(current_generation)}
end
if ARGV[1] == previous_id or ARGV[1] == current_id then
    redis.call('HSET', KEYS[1], 'status', 'revoked', 'revoke_reason', 'refresh_reuse', 'revoked_at', ARGV[6])
    local session_key = redis.call('HGET', KEYS[1], 'session_key')
    local user_sessions_key = redis.call('HGET', KEYS[1], 'user_sessions_key')
    local session_id = redis.call('HGET', KEYS[1], 'session_id')
    redis.call('DEL', session_key)
    redis.call('SREM', user_sessions_key, session_id)
    return {'reused'}
end
return {'invalid'}
"""  # noqa: S105

_REVOKE_REFRESH_FAMILY_SCRIPT = """
if redis.call('EXISTS', KEYS[1]) == 0 then return 0 end
redis.call('HSET', KEYS[1], 'status', 'revoked', 'revoke_reason', ARGV[1], 'revoked_at', ARGV[2])
local session_key = redis.call('HGET', KEYS[1], 'session_key')
local user_sessions_key = redis.call('HGET', KEYS[1], 'user_sessions_key')
redis.call('DEL', session_key)
redis.call('SREM', user_sessions_key, ARGV[4])
return 1
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
    family_id: str | None = None
    device_name: str | None = None
    ip_address: str | None = None
    user_agent: str | None = None


class RefreshRotationOutcome(BaseModel):
    """Result of the Redis Lua rotation protocol."""

    status: str
    family_id: str
    generation: int | None = None
    user_id: str | None = None
    session_id: str | None = None
    device_id: str | None = None
    authz_revision: int | None = None
    expires_at: int | None = None


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

    def __init__(self, redis: Redis, *, key_prefix: str = "") -> None:
        """Create a session store, optionally isolating all keys under ``key_prefix``."""
        self._redis = redis
        self._key_prefix = key_prefix

    def _session_key(self, session_id: str) -> str:
        return f"{self._key_prefix}{_SESSION_KEY.format(session_id)}"

    def _user_sessions_key(self, user_id: str) -> str:
        return f"{self._key_prefix}{_USER_SESSIONS_KEY.format(user_id)}"

    def _authz_revision_key(self, user_id: str) -> str:
        return f"{self._key_prefix}{_USER_AUTHZ_REVISION_KEY.format(user_id)}"

    def _session_key_prefix(self) -> str:
        return f"{self._key_prefix}auth:session:"

    def _refresh_family_key(self, family_id: str) -> str:
        return f"{self._key_prefix}{_REFRESH_FAMILY_KEY.format(family_id)}"

    def _refresh_family_key_prefix(self) -> str:
        return f"{self._key_prefix}auth:refresh_family:"

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
    ) -> AuthResult[tuple[OAuthLinkIntent, str]]:
        intent_result = await self.get_oauth_link_intent(state)
        if isinstance(intent_result, Failure):
            return Failure(intent_result.failure())
        intent = intent_result.unwrap()
        now = datetime.now(UTC)
        if intent is None or intent.ready_for_confirmation or intent.expires_at <= now:
            return Failure(
                AuthAuthenticationError(
                    message="OAuth link transaction is invalid or already completed",
                    source="auth_repository",
                )
            )
        remaining_ttl = max(1, int((intent.expires_at - now).total_seconds()))
        completion_key = _OAUTH_LINK_COMPLETE_KEY.format(intent.state_digest)
        try:
            claimed = await self._redis.set(completion_key, "1", ex=remaining_ttl, nx=True)
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
                ex=remaining_ttl,
            )
            await self._redis.set(
                _OAUTH_LINK_ACTION_KEY.format(action_digest),
                completed.model_dump_json(),
                ex=remaining_ttl,
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

    async def create_refresh_family(
        self,
        *,
        family_id: str,
        user_id: str,
        session_id: str,
        device_id: str,
        authz_revision: int,
        token_id: str,
        token_secret: str,
        issued_at: datetime,
        expires_at: datetime,
        legacy: bool = False,
    ) -> AuthResult[None]:
        """Create a family record containing only token digests."""
        try:
            created = await self._redis.eval(
                _CREATE_REFRESH_FAMILY_SCRIPT,
                1,
                self._refresh_family_key(family_id),
                family_id,
                user_id,
                session_id,
                device_id,
                authz_revision,
                0,
                self._digest(token_id),
                self._digest(token_secret),
                int(issued_at.timestamp()),
                int(expires_at.timestamp()),
                "1" if legacy else "0",
                self._session_key(session_id),
                self._user_sessions_key(user_id),
            )
            if int(created) == 0:
                return Success(None)
            return Success(None)
        except RedisError as exc:
            exc.add_note("auth_repository operation=create_refresh_family")
            exc.add_note(traceback.format_exc())
            logger.bind(
                operation="create_refresh_family",
                diagnostic_notes=list(getattr(exc, "__notes__", [])),
            ).warning("auth_repository infrastructure failure")
            return Failure(
                AuthInfrastructureError(
                    message="Failed to create refresh-token family",
                    details={"family_id": family_id, "error": str(exc)},
                    source="auth_repository",
                )
            )

    async def attach_refresh_family(self, *, session_id: str, family_id: str) -> AuthResult[None]:
        """Bind a migrated legacy session to its newly-created family."""
        try:
            attached = await self._redis.eval(
                _ATTACH_REFRESH_FAMILY_SCRIPT,
                1,
                self._session_key(session_id),
                family_id,
            )
            if int(attached) != 1:
                return Failure(
                    AuthAuthenticationError(
                        message="Session disappeared during refresh migration",
                        source="auth_repository",
                    )
                )
            return Success(None)
        except RedisError as exc:
            exc.add_note("auth_repository operation=attach_refresh_family")
            exc.add_note(traceback.format_exc())
            logger.bind(
                operation="attach_refresh_family",
                diagnostic_notes=list(getattr(exc, "__notes__", [])),
            ).warning("auth_repository infrastructure failure")
            return Failure(
                AuthInfrastructureError(
                    message="Failed to bind legacy session to refresh family",
                    details={"session_id": session_id, "family_id": family_id, "error": str(exc)},
                    source="auth_repository",
                )
            )

    async def rotate_refresh_token(
        self,
        *,
        family_id: str,
        token_id: str,
        token_secret: str,
        child_token_id: str,
        child_token_secret: str,
        child_generation: int,
        expected_authz_revision: int,
        now: datetime,
        grace_seconds: int,
        presented_generation: int | None = None,
    ) -> AuthResult[RefreshRotationOutcome]:
        """Atomically rotate a family and classify benign retry versus reuse."""
        family_key = self._refresh_family_key(family_id)
        try:
            result = await self._redis.eval(
                _ROTATE_REFRESH_TOKEN_SCRIPT,
                1,
                family_key,
                self._digest(token_id),
                self._digest(token_secret),
                self._digest(child_token_id),
                self._digest(child_token_secret),
                child_generation,
                int(now.timestamp()),
                grace_seconds,
                expected_authz_revision,
                "" if presented_generation is None else presented_generation,
            )
            values = [item.decode() if isinstance(item, bytes) else str(item) for item in result]
            status = values[0]
            if status in {"rotated", "grace"}:
                metadata = await self._redis.hgetall(family_key)
                return Success(
                    RefreshRotationOutcome(
                        status=status,
                        family_id=family_id,
                        generation=int(values[1]),
                        user_id=str(metadata.get("user_id")),
                        session_id=str(metadata.get("session_id")),
                        device_id=str(metadata.get("device_id")),
                        authz_revision=int(metadata.get("authz_revision", expected_authz_revision)),
                        expires_at=int(metadata.get("expires_at", 0)),
                    )
                )
            return Success(RefreshRotationOutcome(status=status, family_id=family_id))
        except (RedisError, ValidationError, ValueError) as exc:
            exc.add_note("auth_repository operation=rotate_refresh_token")
            exc.add_note(traceback.format_exc())
            logger.bind(
                operation="rotate_refresh_token",
                family_id=family_id,
                diagnostic_notes=list(getattr(exc, "__notes__", [])),
            ).warning("auth_repository infrastructure failure")
            return Failure(
                AuthInfrastructureError(
                    message="Failed to rotate refresh token",
                    details={"family_id": family_id, "error": str(exc)},
                    source="auth_repository",
                )
            )

    async def revoke_refresh_family(
        self, family_id: str, *, reason: str, user_id: str, session_id: str
    ) -> AuthResult[None]:
        """Revoke a family and its session in one Redis transaction."""
        try:
            await self._redis.eval(
                _REVOKE_REFRESH_FAMILY_SCRIPT,
                1,
                self._refresh_family_key(family_id),
                reason,
                int(datetime.now(UTC).timestamp()),
                user_id,
                session_id,
            )
            await publish_event(
                self._redis,
                new_event(
                    scope="session",
                    user_id=user_id,
                    session_id=session_id,
                    reason=reason,
                    origin="revoke_refresh_family",
                ),
            )
            return Success(None)
        except RedisError as exc:
            exc.add_note("auth_repository operation=revoke_refresh_family")
            exc.add_note(traceback.format_exc())
            logger.bind(
                operation="revoke_refresh_family",
                family_id=family_id,
                diagnostic_notes=list(getattr(exc, "__notes__", [])),
            ).warning("auth_repository infrastructure failure")
            return Failure(
                AuthInfrastructureError(
                    message="Failed to revoke refresh-token family",
                    details={"family_id": family_id, "error": str(exc)},
                    source="auth_repository",
                )
            )

    async def revoke_all_refresh_families(
        self, user_id: str, *, reason: str = "revoke_all_families"
    ) -> AuthResult[list[str]]:
        """Revoke every currently indexed family for incident response."""
        sessions_result = await self.get_user_sessions(user_id)
        if isinstance(sessions_result, Failure):
            return Failure(sessions_result.failure())
        revoked: list[str] = []
        for session in sessions_result.unwrap():
            if not session.family_id:
                continue
            result = await self.revoke_refresh_family(
                session.family_id,
                reason=reason,
                user_id=user_id,
                session_id=session.session_id,
            )
            if isinstance(result, Failure):
                return Failure(result.failure())
            revoked.append(session.family_id)
        return Success(revoked)

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
            self._authz_revision_key(session.user_id),
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
            self._redis.smembers(self._user_sessions_key(user_id)),
        )
        if not raw_ids:
            return Success([])
        sid_list = list(raw_ids)
        async with self._redis.pipeline() as pipe:
            for sid in sid_list:
                pipe.get(self._session_key(sid))
            results: list[bytes | None] = await pipe.execute()
        sessions: list[SessionData] = []
        dead: list[str] = []
        for sid, raw in zip(sid_list, results, strict=True):
            if raw is None:
                dead.append(sid)
            else:
                sessions.append(SessionData.model_validate_json(raw))
        if dead:
            await self._redis.srem(self._user_sessions_key(user_id), *dead)
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
            self._user_sessions_key(user_id),
            self._authz_revision_key(user_id),
            except_session_id or "",
            self._session_key_prefix(),
            "" if authz_revision is None else authz_revision,
            reason,
            int(datetime.now(UTC).timestamp()),
            self._refresh_family_key_prefix(),
        )
        revoked_ids = [item.decode() if isinstance(item, bytes) else str(item) for item in raw_ids]
        await publish_event(
            self._redis,
            new_event(
                scope="user",
                user_id=user_id,
                except_session_id=except_session_id,
                reason=reason,
                authz_revision=authz_revision,
                origin="revoke_all_user_sessions",
            ),
        )
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
        session_key = self._session_key(session.session_id)
        user_key = self._user_sessions_key(session.user_id)
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
            raw = await self._redis.get(self._session_key(session_id))
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
                pipe.delete(self._session_key(session_id))
                pipe.srem(self._user_sessions_key(user_id), session_id)
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

        await publish_event(
            self._redis,
            new_event(
                scope="session",
                user_id=user_id,
                session_id=session_id,
                reason=reason,
                origin="revoke_session",
            ),
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
