"""Focused tests for the Mongo/SQL tenant migration and membership authority."""

from __future__ import annotations

import pytest
from pymongo.errors import PyMongoError
from returns.result import Failure, Success

from app.features.auth.exceptions import AuthInfrastructureError
from app.features.auth.models import LegalRole
from app.features.auth.tenancy import personal_tenant_id
from app.features.auth.tenant_migration import PersonalTenantMigration
from app.features.auth.tenant_models import Tenant, TenantMembership
from app.features.auth.tenant_repository import TenantMembershipService


class FakeUser:
    def __init__(
        self,
        user_id: str,
        *,
        tenant_id: str | None = None,
        legal_roles: tuple[LegalRole, ...] = (LegalRole.ANALYST,),
        role_revision: int = 0,
        is_active: bool = True,
    ) -> None:
        self.id = user_id
        self.tenant_id = tenant_id
        self.legal_roles = legal_roles
        self.role_revision = role_revision
        self.is_active = is_active
        self.save_calls = 0
        self.fail_save = False

    async def save(self) -> None:
        self.save_calls += 1
        if self.fail_save:
            message = "temporary Mongo outage"
            raise PyMongoError(message)


class FakeSession:
    def __init__(self, store: FakeStore) -> None:
        self.store = store

    async def __aenter__(self) -> FakeSession:
        return self

    async def __aexit__(self, *_args: object) -> None:
        return None

    def begin(self) -> FakeSession:
        return self

    async def get(self, model: type, key: object) -> object | None:
        if model is Tenant:
            return self.store.tenants.get(key)
        return self.store.memberships.get(key)

    def add(self, value: object) -> None:
        if isinstance(value, Tenant):
            self.store.tenants[value.id] = value
        else:
            membership = value
            self.store.memberships[membership.tenant_id, membership.user_id] = membership

    async def flush(self) -> None:
        return None


class FakeStore:
    def __init__(self) -> None:
        self.tenants: dict[str, Tenant] = {}
        self.memberships: dict[tuple[str, str], TenantMembership] = {}

    def factory(self) -> FakeSession:
        return FakeSession(self)


def users(*items: FakeUser):
    async def iterator():
        for item in items:
            yield item

    return iterator()


@pytest.mark.asyncio
async def test_personal_tenant_migration_is_idempotent() -> None:
    store = FakeStore()
    user = FakeUser("user-1", legal_roles=(LegalRole.ANALYST, LegalRole.REVIEWER))
    runner = PersonalTenantMigration(store.factory, dry_run=False)

    first = await runner.run(users(user))
    second = await runner.run(users(user))

    assert first.migrated == 1
    assert second.already_current == 1
    assert len(store.tenants) == 1
    assert len(store.memberships) == 1
    membership = store.memberships[personal_tenant_id("user-1"), "user-1"]
    assert membership.legal_roles == ["analyst", "reviewer"]


@pytest.mark.asyncio
async def test_partial_migration_recovers_after_mongo_failure() -> None:
    store = FakeStore()
    user = FakeUser("user-2")
    user.fail_save = True
    runner = PersonalTenantMigration(store.factory, dry_run=False)

    failed = await runner.run(users(user))
    assert failed.failures
    assert len(store.tenants) == 1

    user.fail_save = False
    recovered = await runner.run(users(user))
    assert recovered.already_current == 1
    assert recovered.failures == []


@pytest.mark.asyncio
async def test_existing_non_personal_tenant_is_reported_without_overwrite() -> None:
    store = FakeStore()
    user = FakeUser("user-3", tenant_id="firm-existing")

    report = await PersonalTenantMigration(store.factory).run(users(user))

    assert report.conflicts == [
        {
            "user_id": "user-3",
            "kind": "existing_tenant",
            "current_tenant_id": "firm-existing",
            "expected_tenant_id": "personal:user-3",
        }
    ]
    assert not store.tenants


@pytest.mark.asyncio
async def test_disabled_users_are_migrated_and_reported() -> None:
    store = FakeStore()
    user = FakeUser("user-4", is_active=False)

    report = await PersonalTenantMigration(store.factory).run(users(user))

    assert report.disabled == 1
    assert report.migrated == 1
    assert user.tenant_id == "personal:user-4"


@pytest.mark.asyncio
async def test_sql_membership_role_update_increments_revision_and_syncs_mongo() -> None:
    store = FakeStore()
    tenant_id = personal_tenant_id("user-5")
    store.tenants[tenant_id] = Tenant(id=tenant_id, display_name="Personal")
    store.memberships[tenant_id, "user-5"] = TenantMembership(
        tenant_id=tenant_id,
        user_id="user-5",
        legal_role="analyst",
        legal_roles=["analyst"],
        role_revision=2,
    )

    class AccountStore:
        def __init__(self) -> None:
            self.calls: list[dict[str, object]] = []

        async def set_tenant_binding(self, account_id: str, **kwargs: object):
            self.calls.append({"account_id": account_id, **kwargs})
            return Success(None)

    account_store = AccountStore()
    service = TenantMembershipService(store.factory, account_store=account_store)

    result = await service.update_roles(
        tenant_id=tenant_id,
        user_id="user-5",
        legal_roles=(LegalRole.REVIEWER,),
        expected_revision=2,
    )

    assert result == Success((LegalRole.REVIEWER,))
    assert store.memberships[tenant_id, "user-5"].role_revision == 3
    assert account_store.calls[0]["role_revision"] == 3


@pytest.mark.asyncio
async def test_self_review_setting_defaults_allowed_and_can_be_disabled() -> None:
    store = FakeStore()
    tenant_id = personal_tenant_id("user-6")
    store.tenants[tenant_id] = Tenant(id=tenant_id, display_name="Personal")
    service = TenantMembershipService(store.factory)

    result = await service.set_self_review_allowed(tenant_id=tenant_id, allowed=False)

    assert result == Success(False)
    assert store.tenants[tenant_id].self_review_allowed is False


@pytest.mark.asyncio
async def test_sql_commit_marks_failed_mongo_mirror_for_reconciliation() -> None:
    store = FakeStore()
    tenant_id = personal_tenant_id("user-7")
    store.tenants[tenant_id] = Tenant(id=tenant_id, display_name="Personal")
    store.memberships[tenant_id, "user-7"] = TenantMembership(
        tenant_id=tenant_id,
        user_id="user-7",
        legal_role="analyst",
        legal_roles=["analyst"],
        role_revision=1,
    )

    class FailingAccountStore:
        async def set_tenant_binding(self, account_id: str, **kwargs: object):
            return Failure(
                AuthInfrastructureError(
                    message="Mongo unavailable", source="test", details={"operation": "mirror"}
                )
            )

    service = TenantMembershipService(store.factory, account_store=FailingAccountStore())
    result = await service.update_roles(
        tenant_id=tenant_id,
        user_id="user-7",
        legal_roles=(LegalRole.REVIEWER,),
        expected_revision=1,
    )

    assert isinstance(result, Failure)
    membership = store.memberships[tenant_id, "user-7"]
    assert membership.legal_roles == [LegalRole.REVIEWER.value]
    assert membership.mirror_sync_pending is True
    assert membership.mirror_sync_error == "Mongo unavailable"


def test_legacy_admin_legal_role_normalizes_to_corpus_admin() -> None:
    from app.features.auth.tenant_repository import normalize_legal_roles

    assert normalize_legal_roles(("admin",)) == (LegalRole.CORPUS_ADMIN,)
