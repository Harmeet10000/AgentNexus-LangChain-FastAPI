"""Unit tests for the auth-owned PersonalTenantProvisioner deep module."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from typing import TYPE_CHECKING

import pytest
from returns.result import Failure, Success

from app.features.auth.adapters.persistence.mongodb.user_store import (
    AuthAccount,
    InMemoryAuthAccountStore,
)
from app.features.auth.adapters.persistence.postgres.tenancy_repository import (
    TenantMembershipService,
)
from app.features.auth.application.personal_tenant_provisioning import (
    PersonalTenantProvisioner,
)
from app.features.auth.application.tenancy_migration import check_tenant_readiness
from app.features.auth.domain.errors import AuthInfrastructureError
from app.features.auth.domain.tenancy import LegalRole, personal_tenant_id
from app.features.auth.domain.user import UserRole

from .provisioning_fakes import FakeTenantSQLStore

if TYPE_CHECKING:
    from typing import Any


class FlakyMirrorStore(InMemoryAuthAccountStore):
    """In-memory account store whose Mongo mirror write can be made to fail."""

    def __init__(self) -> None:
        super().__init__()
        self.fail_binding = False

    async def set_tenant_binding(self, *args: Any, **kwargs: Any) -> Any:
        if self.fail_binding:
            return Failure(
                AuthInfrastructureError(
                    message="Mongo unavailable",
                    details={"operation": "set_tenant_binding"},
                    source="test",
                )
            )
        return await super().set_tenant_binding(*args, **kwargs)


def make_account(user_id: str = "user-1") -> AuthAccount:
    now = datetime.now(UTC)
    return AuthAccount(
        id=user_id,
        email="user-1@example.com",
        full_name="User One",
        password_hash=None,
        is_active=True,
        is_verified=True,
        role=UserRole.USER,
        permissions=frozenset(),
        created_at=now,
        updated_at=now,
    )


@pytest.mark.asyncio
async def test_provision_creates_tenant_membership_and_completes_mirror() -> None:
    sql = FakeTenantSQLStore()
    accounts = FlakyMirrorStore()
    accounts._index_account(make_account())
    provisioner = PersonalTenantProvisioner(sql.factory, accounts)

    result = await provisioner.provision(
        "user-1", display_name="User One", legal_roles=(LegalRole.REVIEWER,)
    )

    assert isinstance(result, Success)
    payload = result.unwrap()
    assert payload.tenant_id == personal_tenant_id("user-1")
    assert payload.legal_roles == (LegalRole.REVIEWER,)
    assert payload.role_revision == 0
    assert payload.mirror_synced is True
    assert len(sql.tenants) == 1
    membership = sql.memberships[personal_tenant_id("user-1"), "user-1"]
    assert membership.legal_roles == ["reviewer"]
    assert membership.mirror_sync_pending is False
    mirrored = (await accounts.find_by_id("user-1")).unwrap()
    assert mirrored is not None
    assert mirrored.tenant_id == personal_tenant_id("user-1")
    assert mirrored.provisioning_state.value == "completed"


@pytest.mark.asyncio
async def test_provision_is_idempotent_and_preserves_sql_authority() -> None:
    sql = FakeTenantSQLStore()
    accounts = FlakyMirrorStore()
    accounts._index_account(make_account())
    provisioner = PersonalTenantProvisioner(sql.factory, accounts)

    first = await provisioner.provision(
        "user-1", display_name="User One", legal_roles=(LegalRole.ANALYST,)
    )
    second = await provisioner.provision(
        "user-1", display_name="Renamed", legal_roles=(LegalRole.CORPUS_ADMIN,)
    )

    assert isinstance(first, Success)
    assert isinstance(second, Success)
    assert second.unwrap().tenant_id == first.unwrap().tenant_id
    assert second.unwrap().role_revision == first.unwrap().role_revision
    # Exactly one tenant/membership; a retry never overwrites SQL authority.
    assert len(sql.tenants) == 1
    membership = sql.memberships[personal_tenant_id("user-1"), "user-1"]
    assert membership.legal_roles == ["analyst"]


@pytest.mark.asyncio
async def test_sql_failure_keeps_account_pending_and_returns_retryable_error() -> None:
    sql = FakeTenantSQLStore()
    accounts = FlakyMirrorStore()
    accounts._index_account(make_account())
    provisioner = PersonalTenantProvisioner(sql.factory, accounts)
    sql.fail_next_commit = True

    result = await provisioner.provision(
        "user-1", display_name="User One", legal_roles=(LegalRole.ANALYST,)
    )

    assert isinstance(result, Failure)
    error = result.failure()
    assert isinstance(error, AuthInfrastructureError)
    assert error.retryable is True
    # Nothing durable was written and the mirror never saw a binding.
    assert not sql.tenants
    assert not sql.memberships
    mirrored = (await accounts.find_by_id("user-1")).unwrap()
    assert mirrored is not None
    assert mirrored.tenant_id == ""
    assert mirrored.provisioning_state.value == "completed"


@pytest.mark.asyncio
async def test_mirror_failure_keeps_sql_authority_and_reconciles_later() -> None:
    sql = FakeTenantSQLStore()
    accounts = FlakyMirrorStore()
    accounts._index_account(make_account())
    provisioner = PersonalTenantProvisioner(sql.factory, accounts)
    accounts.fail_binding = True

    committed = await provisioner.provision(
        "user-1", display_name="User One", legal_roles=(LegalRole.ANALYST,)
    )

    assert isinstance(committed, Success)
    assert committed.unwrap().mirror_synced is False
    membership = sql.memberships[personal_tenant_id("user-1"), "user-1"]
    assert membership.mirror_sync_pending is True
    # Account stays usable through SQL authority even with a stale mirror.
    service = TenantMembershipService(sql.factory, account_store=accounts)
    selected = await service.resolve_tenant(user_id="user-1")
    assert isinstance(selected, Success)
    assert selected.unwrap() == personal_tenant_id("user-1")

    # Durable repair: the retry path (or the reconcile worker) converges.
    accounts.fail_binding = False
    repair = await provisioner.provision(
        "user-1", display_name="User One", legal_roles=(LegalRole.ANALYST,)
    )
    assert isinstance(repair, Success)
    assert repair.unwrap().mirror_synced is True
    assert sql.memberships[personal_tenant_id("user-1"), "user-1"].mirror_sync_pending is False
    mirrored = (await accounts.find_by_id("user-1")).unwrap()
    assert mirrored is not None
    assert mirrored.provisioning_state.value == "completed"


@pytest.mark.asyncio
async def test_concurrent_provisioning_creates_exactly_one_tenant_and_membership() -> None:
    sql = FakeTenantSQLStore()
    sql.yield_between_statements = True
    accounts = FlakyMirrorStore()
    accounts._index_account(make_account())
    provisioner = PersonalTenantProvisioner(sql.factory, accounts)

    results = await asyncio.gather(
        provisioner.provision(
            "user-1", display_name="User One", legal_roles=(LegalRole.REVIEWER,)
        ),
        provisioner.provision(
            "user-1", display_name="User One", legal_roles=(LegalRole.ANALYST,)
        ),
    )

    assert all(isinstance(result, Success) for result in results)
    assert len(sql.tenants) == 1
    assert len(sql.memberships) == 1
    membership = sql.memberships[personal_tenant_id("user-1"), "user-1"]
    assert membership.role_revision == 0
    tenant_ids = {
        result.unwrap().tenant_id for result in results if isinstance(result, Success)
    }
    assert tenant_ids == {personal_tenant_id("user-1")}


@pytest.mark.asyncio
async def test_has_membership_reflects_sql_and_maps_outage_to_infra_error() -> None:
    sql = FakeTenantSQLStore()
    accounts = FlakyMirrorStore()
    provisioner = PersonalTenantProvisioner(sql.factory, accounts)

    before = await provisioner.has_membership("user-1")
    assert isinstance(before, Success)
    assert before.unwrap() is False

    await provisioner.provision(
        "user-1", display_name="User One", legal_roles=(LegalRole.ANALYST,)
    )
    after = await provisioner.has_membership("user-1")
    assert isinstance(after, Success)
    assert after.unwrap() is True

    sql.fail_select = True
    outage = await provisioner.has_membership("user-1")
    assert isinstance(outage, Failure)
    assert isinstance(outage.failure(), AuthInfrastructureError)
    assert outage.failure().retryable is True


@pytest.mark.asyncio
async def test_reconcile_never_provisions_missing_sql_state() -> None:
    sql = FakeTenantSQLStore()
    accounts = FlakyMirrorStore()
    service = TenantMembershipService(sql.factory, account_store=accounts)

    report = await service.reconcile_pending_mirrors(limit=10)

    assert report == {"scanned": 0, "succeeded": 0, "failed": 0}
    assert not sql.tenants
    assert not sql.memberships
    # The readiness gate, not --reconcile, is what reports missing SQL state.
    readiness = await check_tenant_readiness(sql.factory, _legacy_accounts())
    assert not readiness.ready
    assert readiness.missing_membership == ["legacy-user"]


async def _legacy_accounts():
    class _LegacyAccount:
        id = "legacy-user"
        is_active = True
        tenant_id = None
        legal_roles = (LegalRole.ANALYST,)
        role_revision = 0

    yield _LegacyAccount()


@pytest.mark.asyncio
async def test_unexpected_provisioning_errors_are_not_swallowed() -> None:
    failure_message = "unexpected programmer failure"

    class ExplodingFactory:
        def __call__(self) -> object:
            raise RuntimeError(failure_message)

    provisioner = PersonalTenantProvisioner(ExplodingFactory(), FlakyMirrorStore())  # type: ignore[arg-type]

    with pytest.raises(RuntimeError, match="unexpected programmer failure"):
        await provisioner.provision(
            "user-1", display_name="User One", legal_roles=(LegalRole.ANALYST,)
        )


@pytest.mark.asyncio
async def test_role_update_on_second_tenant_survives_higher_personal_mirror_revision() -> None:
    """Regression: the mirror CAS must not compare revisions across tenants.

    A personal mirror at revision 5 used to reject a valid team-tenant mirror
    write at revision 1, so the caller saw a conflict after the authoritative
    SQL update had already committed and reconciliation kept failing.
    """
    sql = FakeTenantSQLStore()
    accounts = FlakyMirrorStore()
    accounts._index_account(make_account())
    # Personal tenant mirror already advanced to revision 5.
    bound = await accounts.set_tenant_binding(
        "user-1",
        tenant_id=personal_tenant_id("user-1"),
        legal_roles=(LegalRole.ANALYST,),
        role_revision=5,
    )
    assert isinstance(bound, Success)

    # A team membership exists in SQL (as a corpus-admin would have created).
    from app.features.auth.application.personal_tenant_provisioning import (
        PersonalTenantProvisioningRepository,
    )

    async with sql.factory() as session, session.begin():
        await PersonalTenantProvisioningRepository(session).upsert_membership(
            "team-a",
            "user-1",
            legal_roles=(LegalRole.ANALYST,),
            role_revision=0,
        )

    service = TenantMembershipService(sql.factory, account_store=accounts)
    result = await service.update_roles(
        tenant_id="team-a",
        user_id="user-1",
        legal_roles=(LegalRole.REVIEWER,),
        expected_revision=0,
    )

    assert isinstance(result, Success)
    membership = sql.memberships["team-a", "user-1"]
    assert membership.legal_roles == [LegalRole.REVIEWER.value]
    # The mirror accepted the team write: no reconcile marker left behind.
    assert membership.mirror_sync_pending is False
    mirrored = (await accounts.find_by_id("user-1")).unwrap()
    assert mirrored is not None
    assert mirrored.tenant_id == "team-a"
    assert mirrored.tenant_role_revision == 1


@pytest.mark.asyncio
async def test_plain_session_exit_discards_staged_state_without_consuming_commit_failure() -> None:
    """Only ``begin()`` owns the commit boundary — a bare session exit commits nothing."""
    sql = FakeTenantSQLStore()
    sql.fail_next_commit = True
    from app.features.auth.application.personal_tenant_provisioning import (
        PersonalTenantProvisioningRepository,
    )

    session = sql.factory()
    async with session:
        await PersonalTenantProvisioningRepository(session).upsert_membership(
            personal_tenant_id("user-1"),
            "user-1",
            legal_roles=(LegalRole.ANALYST,),
            role_revision=0,
        )

    assert not sql.tenants
    assert not sql.memberships
    # The failure flag was not consumed by the session-level exit.
    assert sql.fail_next_commit is True
