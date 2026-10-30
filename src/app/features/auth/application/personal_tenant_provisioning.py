"""Automatic personal-tenant provisioning for every account-creation path.

This module is the single deep auth-owned interface described by the approved
automatic-provisioning design:

- :meth:`PersonalTenantProvisioner.provision` upserts the deterministic
  ``personal:<user_id>`` tenant, its membership, and a durable provisioning
  record in **one PostgreSQL transaction**, using database-native
  ``INSERT ... ON CONFLICT`` so concurrent provisioning converges on exactly
  one tenant/membership (no read-then-insert races).
- The Mongo tenant mirror is updated best-effort **after** the SQL commit with
  a revision-monotonic compare-and-set; SQL stays authoritative and a failed
  mirror write leaves the staged ``mirror_sync_pending`` marker in place for
  the existing reconciler.
- :meth:`PersonalTenantProvisioner.has_membership` is the read-only invariant
  check used by login/refresh before a session is minted.  It is deliberately
  not a provisioning mechanism.

The durable provisioning record is the ``app_tenant_memberships``
``mirror_sync_pending`` marker staged inside the same transaction (together
with the membership row itself).  A dedicated outbox event type would need a
new immutable job-catalog entry outside auth's ownership; the marker is
consumed by ``TenantMembershipService.reconcile_pending_mirrors`` and therefore
already has a resumable repair driver.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from opentelemetry import metrics
from returns.result import Failure, Success
from sqlalchemy import func, select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.exc import SQLAlchemyError

from app.features.auth.adapters.persistence.postgres.tenancy_repository import (
    TenantMembershipRepository,
    normalize_legal_roles,
)
from app.features.auth.domain.errors import AuthInfrastructureError
from app.features.auth.domain.tenancy import LegalRole, personal_tenant_id
from app.features.auth.domain.tenancy_models import Tenant, TenantMembership
from app.observability.logging import logger, trace_layer

if TYPE_CHECKING:
    from sqlalchemy import Select
    from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

    from app.features.auth.adapters.persistence.mongodb.user_store import AuthAccountStore
    from app.features.auth.domain.errors import AuthResult

_PROVISIONING_METER = metrics.get_meter("app.auth.provisioning")
_PROVISION_COMMITTED = _PROVISIONING_METER.create_counter("auth.provisioning.sql_committed")
_PROVISION_FAILED = _PROVISIONING_METER.create_counter("auth.provisioning.sql_failed")
_PROVISION_MIRROR_FAILED = _PROVISIONING_METER.create_counter(
    "auth.provisioning.mirror_failed"
)


@dataclass(frozen=True, slots=True)
class ProvisioningResult:
    """Outcome of one idempotent personal-tenant provisioning call."""

    user_id: str
    tenant_id: str
    legal_roles: tuple[LegalRole, ...]
    role_revision: int
    # False means SQL is committed and authoritative while the Mongo mirror
    # could not be updated; the staged pending marker drives reconciliation.
    mirror_synced: bool


class PersonalTenantProvisioningRepository:
    """SQL-only provisioning statements; transaction ownership stays with the caller."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def upsert_tenant(self, tenant_id: str, *, display_name: str) -> None:
        """Insert the tenant if absent; a concurrent winner's row is preserved."""
        await self._session.execute(
            pg_insert(Tenant)
            .values(id=tenant_id, display_name=display_name, self_review_allowed=True)
            .on_conflict_do_nothing(index_elements=[Tenant.id])
        )

    async def upsert_membership(
        self,
        tenant_id: str,
        user_id: str,
        *,
        legal_roles: tuple[LegalRole, ...],
        role_revision: int,
    ) -> None:
        """Insert the membership if absent, staging the durable reconcile marker.

        Existing rows are never overwritten: SQL membership authority wins over
        a retried or concurrent provisioning attempt.  ``mirror_sync_pending``
        is set on insert only, so the marker survives a crash between the SQL
        commit and the best-effort Mongo mirror write.
        """
        normalized = [role.value for role in normalize_legal_roles(legal_roles)]
        await self._session.execute(
            pg_insert(TenantMembership)
            .values(
                tenant_id=tenant_id,
                user_id=user_id,
                legal_role=normalized[0],
                legal_roles=normalized,
                role_revision=max(0, role_revision),
                mirror_sync_pending=True,
            )
            .on_conflict_do_nothing(
                index_elements=[TenantMembership.tenant_id, TenantMembership.user_id]
            )
        )

    async def get_membership(self, tenant_id: str, user_id: str) -> TenantMembership | None:
        return await self._session.get(TenantMembership, (tenant_id, user_id))

    async def count_memberships_for_user(self, user_id: str) -> int:
        statement: Select[tuple[int]] = (
            select(func.count()).select_from(TenantMembership).where(
                TenantMembership.user_id == user_id
            )
        )
        return int((await self._session.execute(statement)).scalar_one())


class PersonalTenantProvisioner:
    """Owns transactions, idempotency, mirror sync, durable repair, observability.

    Constructed with the application's async session factory and the account
    store; every account-creation path and the session-issuance invariant go
    through this one object.
    """

    def __init__(
        self,
        session_factory: async_sessionmaker[AsyncSession],
        account_store: AuthAccountStore | None = None,
    ) -> None:
        self._session_factory = session_factory
        self._account_store = account_store

    @staticmethod
    async def _upsert_personal_tenant(
        session: AsyncSession,
        *,
        tenant_id: str,
        user_id: str,
        display_name: str,
        legal_roles: tuple[LegalRole, ...],
    ) -> tuple[tuple[LegalRole, ...], int]:
        """Run the idempotent upserts in the caller's transaction and read back authority.

        Concurrent/duplicate callers converge: ``ON CONFLICT DO NOTHING``
        preserves the winning row, and the read-back returns the authoritative
        roles/revision either way.  A missing row after the upsert is a store
        contract violation and fails the transaction.
        """
        repository = PersonalTenantProvisioningRepository(session)
        await repository.upsert_tenant(tenant_id, display_name=display_name)
        await repository.upsert_membership(
            tenant_id,
            user_id,
            legal_roles=legal_roles,
            role_revision=0,
        )
        membership = await repository.get_membership(tenant_id, user_id)
        if membership is None:  # pragma: no cover — defensive against store bugs
            message = "provisioned membership row missing after upsert"
            raise SQLAlchemyError(message)
        return normalize_legal_roles(membership.legal_roles), membership.role_revision

    @trace_layer("provisioning")
    async def provision(
        self,
        user_id: str,
        *,
        display_name: str,
        legal_roles: tuple[LegalRole, ...] = (LegalRole.ANALYST,),
    ) -> AuthResult[ProvisioningResult]:
        """Idempotently create the personal tenant and membership in one SQL transaction.

        Duplicate/concurrent calls deterministically return the same tenant and
        membership (database-native ``ON CONFLICT`` semantics).  After the SQL
        commit, the Mongo mirror update is best-effort with a revision-monotonic
        compare-and-set; its failure never fails provisioning because SQL stays
        authoritative.
        """
        tenant_id = personal_tenant_id(user_id)
        safe_display_name = display_name or f"Personal tenant {user_id}"
        try:
            async with self._session_factory() as session, session.begin():
                roles, role_revision = await self._upsert_personal_tenant(
                    session,
                    tenant_id=tenant_id,
                    user_id=user_id,
                    display_name=safe_display_name,
                    legal_roles=legal_roles,
                )
        except SQLAlchemyError as exc:
            exc.add_note(f"user_id={user_id}, operation=provision_personal_tenant")
            logger.bind(user_id=user_id, tenant_id=tenant_id).exception(
                "personal_tenant_provisioning_sql_failed"
            )
            _PROVISION_FAILED.add(1)
            return Failure(
                AuthInfrastructureError(
                    message="Personal tenant provisioning failed",
                    details={"operation": "provision_personal_tenant"},
                    source="personal_tenant_provisioner",
                )
            )
        _PROVISION_COMMITTED.add(1)

        # The staged ``mirror_sync_pending`` marker already survived the commit;
        # a crash here leaves a resumable record.  Mirror sync is best-effort.
        mirror_synced = await self._sync_mirror(
            user_id=user_id,
            tenant_id=tenant_id,
            legal_roles=roles,
            role_revision=role_revision,
        )
        logger.bind(
            user_id=user_id,
            tenant_id=tenant_id,
            role_revision=role_revision,
            mirror_synced=mirror_synced,
        ).info("personal_tenant_provisioned")
        return Success(
            ProvisioningResult(
                user_id=user_id,
                tenant_id=tenant_id,
                legal_roles=roles,
                role_revision=role_revision,
                mirror_synced=mirror_synced,
            )
        )

    @trace_layer("provisioning")
    async def has_membership(self, user_id: str) -> AuthResult[bool]:
        """Read-only SQL membership existence check for the session invariant.

        The only failure kind returned is :class:`AuthInfrastructureError` —
        SQL being unreachable must render as a retryable 503, never as a
        401/403, and never as lazy provisioning.
        """
        try:
            async with self._session_factory() as session:
                count = await PersonalTenantProvisioningRepository(
                    session
                ).count_memberships_for_user(user_id)
        except SQLAlchemyError as exc:
            exc.add_note(f"user_id={user_id}, operation=membership_invariant_check")
            logger.bind(user_id=user_id).exception("membership_invariant_check_failed")
            return Failure(
                AuthInfrastructureError(
                    message="SQL membership check failed",
                    details={"operation": "membership_invariant_check"},
                    source="personal_tenant_provisioner",
                )
            )
        return Success(count > 0)

    async def _sync_mirror(
        self,
        *,
        user_id: str,
        tenant_id: str,
        legal_roles: tuple[LegalRole, ...],
        role_revision: int,
    ) -> bool:
        """Best-effort Mongo mirror write guarded by a revision-monotonic CAS.

        Returns ``True`` only when the mirror accepted the write and the staged
        pending marker was cleared.  Any failure leaves the SQL-staged marker
        for :meth:`TenantMembershipService.reconcile_pending_mirrors`.
        """
        if self._account_store is None:
            return False
        result = await self._account_store.set_tenant_binding(
            user_id,
            tenant_id=tenant_id,
            legal_roles=legal_roles,
            tenant_role_revision=role_revision,
            role_revision=role_revision,
        )
        if isinstance(result, Failure):
            _PROVISION_MIRROR_FAILED.add(1)
            logger.bind(user_id=user_id, tenant_id=tenant_id).error(
                "personal_tenant_mirror_sync_failed"
            )
            await self._record_mirror_error(
                tenant_id, user_id, error=result.failure().message
            )
            return False
        try:
            async with self._session_factory() as session, session.begin():
                await TenantMembershipRepository(session).clear_mirror_sync_pending(
                    tenant_id,
                    user_id,
                    role_revision=role_revision,
                )
        except SQLAlchemyError as exc:
            exc.add_note(
                f"tenant_id={tenant_id}, user_id={user_id}, operation=clear_mirror_sync_pending"
            )
            logger.bind(user_id=user_id, tenant_id=tenant_id).exception(
                "personal_tenant_mirror_marker_clear_failed"
            )
            return False
        return True

    async def _record_mirror_error(self, tenant_id: str, user_id: str, *, error: str) -> None:
        """Refresh the pending marker's error text after a failed mirror write.

        The marker itself was staged inside the provisioning transaction, so a
        failure here only loses the diagnostic string — never the retry intent.
        """
        try:
            async with self._session_factory() as session, session.begin():
                await TenantMembershipRepository(session).mark_mirror_sync_pending(
                    tenant_id, user_id, error=error
                )
        except SQLAlchemyError as exc:
            exc.add_note(
                f"tenant_id={tenant_id}, user_id={user_id}, operation=mark_mirror_sync_pending"
            )
            logger.bind(user_id=user_id, tenant_id=tenant_id).exception(
                "personal_tenant_mirror_marker_write_failed"
            )
