"""Focused contract tests for the auth account-store adapters."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING

import pytest
from fastapi import Response
from returns.result import Failure, Success

from app.features.auth.adapters.persistence.mongodb.user_store import (
    BeanieAuthAccountStore,
    InMemoryAuthAccountStore,
    NewAuthAccount,
    OAuthAccountInput,
)
from app.features.auth.domain.errors import AuthConflictError, AuthInfrastructureError
from app.features.auth.domain.user import LegalRole, UserRole
from app.kernel.result import render_result

pytestmark = pytest.mark.unit

if TYPE_CHECKING:
    from typing import Any, ClassVar


class _FakeCollection:
    def __init__(self, result: object) -> None:
        self.result = result
        self.filter: dict[str, object] | None = None
        self.update: dict[str, object] | None = None
        self.return_document: object | None = None

    async def find_one_and_update(
        self,
        filter_: dict[str, object],
        update: dict[str, object],
        *,
        return_document: object,
    ) -> object:
        self.filter = filter_
        self.update = update
        self.return_document = return_document
        return self.result


class _ResetUser:
    collection = _FakeCollection({"_id": "updated"})

    @classmethod
    def get_pymongo_collection(cls) -> _FakeCollection:
        return cls.collection


async def test_beanie_reset_consumption_uses_atomic_expiry_guard() -> None:
    store = BeanieAuthAccountStore()
    store._User = _ResetUser

    result = await store.consume_reset_token_and_replace_password(
        "507f1f77bcf86cd799439011",
        "secret-token-hash",
        "new-password-hash",
    )

    assert isinstance(result, Success)
    assert _ResetUser.collection.filter is not None
    assert _ResetUser.collection.filter["reset_token_hash"] == "secret-token-hash"
    expiry = _ResetUser.collection.filter["reset_token_expires_at"]
    assert isinstance(expiry, dict)
    assert "$gt" in expiry
    assert _ResetUser.collection.update == {
        "$set": {
            "hashed_password": "new-password-hash",
            "reset_token_hash": None,
            "reset_token_expires_at": None,
            "updated_at": expiry["$gt"],
        }
    }


class _OAuthCollection:
    calls = 0
    filter: dict[str, object] | None = None
    update: dict[str, object] | None = None
    return_document: object | None = None

    async def find_one_and_update(
        self,
        filter_: dict[str, object],
        update: dict[str, object],
        *,
        return_document: object,
    ) -> object | None:
        type(self).calls += 1
        type(self).filter = filter_
        type(self).update = update
        type(self).return_document = return_document
        if _OAuthUser.oauth_accounts:
            return None
        payload = update["$push"]["oauth_accounts"]
        assert isinstance(payload, dict)
        payload = {key: value for key, value in payload.items() if key != "linked_at"}
        _OAuthUser.oauth_accounts.append(OAuthAccountInput.model_validate(payload))
        return {"_id": _OAuthUser.id}


class _OAuthUser:
    id = "507f1f77bcf86cd799439011"
    oauth_accounts: ClassVar[list[Any]] = []
    collection = _OAuthCollection()
    is_verified = False
    updated_at = datetime.now(UTC)

    @classmethod
    async def get(cls, _account_id: object) -> _OAuthUser:
        return cls()

    @classmethod
    def get_pymongo_collection(cls) -> _OAuthCollection:
        return cls.collection


async def test_beanie_oauth_link_is_idempotent_and_rejects_provider_replacement() -> None:
    _OAuthUser.oauth_accounts = []
    _OAuthCollection.calls = 0
    _OAuthCollection.filter = None
    _OAuthCollection.update = None
    store = BeanieAuthAccountStore()
    store._User = _OAuthUser

    first = await store.link_oauth_account(_OAuthUser.id, "google", "google-1", "user@example.com")
    second = await store.link_oauth_account(_OAuthUser.id, "google", "google-1", "user@example.com")
    replacement = await store.link_oauth_account(
        _OAuthUser.id, "google", "google-2", "other@example.com"
    )

    assert isinstance(first, Success)
    assert isinstance(second, Success)
    assert isinstance(replacement, Failure)
    assert isinstance(replacement.failure(), AuthConflictError)
    assert len(_OAuthUser.oauth_accounts) == 1
    assert _OAuthCollection.calls == 1
    assert _OAuthCollection.filter is not None
    assert _OAuthCollection.filter["oauth_accounts"] == {
        "$not": {"$elemMatch": {"provider": "google"}}
    }
    assert _OAuthCollection.update is not None
    assert "$push" in _OAuthCollection.update
    assert "$set" in _OAuthCollection.update


def test_auth_infrastructure_details_are_safe_for_rendering() -> None:
    error = AuthInfrastructureError(
        message="database unavailable",
        details={
            "email": "alice@example.com",
            "token_hash": "secret-token-hash",
            "error": "driver leaked details",
            "operation": "find_by_email",
        },
    )

    assert error.details == {"operation": "find_by_email"}
    assert "alice@example.com" not in str(error.model_dump())
    assert "secret-token-hash" not in str(error.model_dump())
    assert "driver leaked details" not in str(error.model_dump())

    response = Response()
    envelope = render_result(Failure(error), response)
    assert "alice@example.com" not in str(envelope.model_dump())
    assert "secret-token-hash" not in str(envelope.model_dump())
    assert "driver leaked details" not in str(envelope.model_dump())


@pytest.mark.asyncio
async def test_tenant_binding_does_not_advance_platform_session_revision() -> None:
    store = InMemoryAuthAccountStore()
    created = await store.create(
        NewAuthAccount(
            email="tenant@example.com",
            password_hash="hash",
            role=UserRole.USER,
            tenant_id="tenant-a",
            legal_roles=(LegalRole.ANALYST,),
        )
    )
    assert isinstance(created, Success)
    account = created.unwrap()
    account = account.model_copy(update={"role_revision": 7, "tenant_role_revision": 2})
    store._unindex_account(created.unwrap())
    store._index_account(account)

    result = await store.set_tenant_binding(
        account.id,
        tenant_id="tenant-a",
        legal_roles=(LegalRole.REVIEWER,),
        role_revision=3,
    )

    assert isinstance(result, Success)
    refreshed = (await store.find_by_id(account.id)).unwrap()
    assert refreshed is not None
    assert refreshed.role_revision == 7
    assert refreshed.tenant_role_revision == 3


async def test_inmemory_tenant_binding_rejects_stale_revision_cas() -> None:
    store = InMemoryAuthAccountStore()
    created = await store.create(
        NewAuthAccount(email="cas@example.com", password_hash="hash", is_verified=True)
    )
    assert isinstance(created, Success)
    account_id = created.unwrap().id

    newer = await store.set_tenant_binding(
        account_id,
        tenant_id="personal:cas",
        legal_roles=(LegalRole.ANALYST,),
        role_revision=4,
    )
    assert isinstance(newer, Success)

    stale = await store.set_tenant_binding(
        account_id,
        tenant_id="personal:cas",
        legal_roles=(LegalRole.REVIEWER,),
        role_revision=2,
    )
    assert isinstance(stale, Failure)
    assert isinstance(stale.failure(), AuthConflictError)

    # Equal revision is an idempotent replay, not a stale write.
    replay = await store.set_tenant_binding(
        account_id,
        tenant_id="personal:cas",
        legal_roles=(LegalRole.ANALYST,),
        role_revision=4,
    )
    assert isinstance(replay, Success)
    refreshed = (await store.find_by_id(account_id)).unwrap()
    assert refreshed is not None
    assert refreshed.legal_roles == (LegalRole.ANALYST,)
    assert refreshed.provisioning_state.value == "completed"


async def test_beanie_tenant_binding_uses_revision_monotonic_atomic_guard() -> None:
    collection = _FakeCollection({"_id": "updated"})

    class _BindingUser:
        collection_ref = collection

        @classmethod
        def get_pymongo_collection(cls):
            return cls.collection_ref

    store = BeanieAuthAccountStore()
    store._User = _BindingUser

    result = await store.set_tenant_binding(
        "507f1f77bcf86cd799439011",
        tenant_id="personal:507f1f77bcf86cd799439011",
        legal_roles=(LegalRole.ANALYST,),
        role_revision=3,
    )

    assert isinstance(result, Success)
    filter_ = collection.filter
    assert filter_ is not None
    # A stale writer must lose atomically: the revision guard lives in the
    # update filter, not in a read-then-write precondition.
    revision_guard = filter_["$or"]
    assert {"tenant_role_revision": {"$lte": 3}} in revision_guard
    update = collection.update
    assert update is not None
    assert update["$set"]["provisioning_state"] == "completed"


async def test_beanie_tenant_binding_reports_stale_rejection_when_guard_matches_nothing() -> None:
    class _NoMatchCollection:
        async def find_one_and_update(self, *args: object, **kwargs: object) -> object:
            # The atomic guard matched nothing: a newer mirror revision won.
            return None

    class _BindingUser:
        @staticmethod
        def get_pymongo_collection() -> _NoMatchCollection:
            return _NoMatchCollection()

    class _Current:
        id = "507f1f77bcf86cd799439011"
        tenant_role_revision = 9

    async def _current_user(_account_id: str):
        return Success(_Current())

    store = BeanieAuthAccountStore()
    store._User = _BindingUser
    store._get_user = _current_user  # type: ignore[method-assign]

    result = await store.set_tenant_binding(
        "507f1f77bcf86cd799439011",
        tenant_id="personal:507f1f77bcf86cd799439011",
        legal_roles=(LegalRole.ANALYST,),
        role_revision=3,
    )

    assert isinstance(result, Failure)
    assert isinstance(result.failure(), AuthConflictError)
