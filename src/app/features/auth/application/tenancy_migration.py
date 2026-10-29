"""Idempotent Mongo user → SQL personal-tenant migration.

The migration is intentionally application-level because the source users live
in Mongo while tenant ownership and memberships live in PostgreSQL.  It is
safe to rerun: personal tenant IDs are deterministic, existing SQL membership
roles win over a retry, and a user carrying a different explicit tenant is
reported as a conflict rather than overwritten.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from pymongo.errors import PyMongoError
from sqlalchemy import select
from sqlalchemy.exc import SQLAlchemyError

from app.features.auth.adapters.persistence.postgres.tenancy_repository import (
    TenantMembershipRepository,
    normalize_legal_roles,
)
from app.features.auth.domain.tenancy import LegalRole, personal_tenant_id
from app.features.auth.domain.tenancy_models import Tenant, TenantMembership
from app.features.auth.domain.user import User
from app.observability.logging import logger

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker


@dataclass(slots=True)
class TenantMigrationReport:
    """Counters and conflict details emitted by one migration invocation."""

    scanned: int = 0
    migrated: int = 0
    already_current: int = 0
    disabled: int = 0
    tenants_created: int = 0
    memberships_created: int = 0
    conflicts: list[dict[str, str]] = field(default_factory=list)
    failures: list[dict[str, str]] = field(default_factory=list)
    pending_mirror_sync: int = 0

    def as_dict(self) -> dict[str, object]:
        return {
            "scanned": self.scanned,
            "migrated": self.migrated,
            "already_current": self.already_current,
            "disabled": self.disabled,
            "tenants_created": self.tenants_created,
            "memberships_created": self.memberships_created,
            "conflicts": list(self.conflicts),
            "failures": list(self.failures),
            "pending_mirror_sync": self.pending_mirror_sync,
        }


@dataclass(slots=True)
class TenantReadinessReport:
    """Rollout gate proving active accounts have valid SQL memberships.

    Team memberships are intentionally not a duplicate condition: an account
    may belong to many tenants.  Every active account must, however, retain a
    valid personal tenant for the compatibility/default path.
    """

    active_accounts: int = 0
    valid_accounts: int = 0
    missing_membership: list[str] = field(default_factory=list)
    missing_personal_tenant: list[str] = field(default_factory=list)
    duplicate_membership: list[str] = field(default_factory=list)
    invalid_membership: list[str] = field(default_factory=list)
    missing_tenant: list[str] = field(default_factory=list)
    mirror_mismatch: list[str] = field(default_factory=list)
    pending_mirror_sync: list[str] = field(default_factory=list)

    @property
    def ready(self) -> bool:
        return not any(
            (
                self.missing_membership,
                self.missing_personal_tenant,
                self.duplicate_membership,
                self.invalid_membership,
                self.missing_tenant,
                self.mirror_mismatch,
                self.pending_mirror_sync,
            )
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "ready": self.ready,
            "active_accounts": self.active_accounts,
            "valid_accounts": self.valid_accounts,
            "missing_membership": list(self.missing_membership),
            "missing_personal_tenant": list(self.missing_personal_tenant),
            "duplicate_membership": list(self.duplicate_membership),
            "invalid_membership": list(self.invalid_membership),
            "missing_tenant": list(self.missing_tenant),
            "mirror_mismatch": list(self.mirror_mismatch),
            "pending_mirror_sync": list(self.pending_mirror_sync),
        }


def migration_roles(user: User) -> tuple[LegalRole, ...]:
    """Return valid source roles, defaulting malformed/legacy users to analyst."""
    try:
        return normalize_legal_roles(user.legal_roles)
    except (TypeError, ValueError):
        return (LegalRole.ANALYST,)


async def iter_users() -> AsyncIterator[User]:
    """Stream users so migration memory is bounded by one document."""
    async for user in User.find_all():
        yield user


class PersonalTenantMigration:
    """One-user-at-a-time migration runner with dry-run support."""

    def __init__(
        self,
        session_factory: async_sessionmaker[AsyncSession],
        *,
        dry_run: bool = False,
    ) -> None:
        self._session_factory = session_factory
        self._dry_run = dry_run

    async def run(self, users: AsyncIterator[User] | None = None) -> TenantMigrationReport:
        report = TenantMigrationReport()
        source = users if users is not None else iter_users()
        async for user in source:
            report.scanned += 1
            user_id = str(user.id)
            if not user.is_active:
                report.disabled += 1
            await self._migrate_user(user, user_id=user_id, report=report)
        return report

    async def _migrate_user(  # noqa: PLR0912
        self, user: User, *, user_id: str, report: TenantMigrationReport
    ) -> None:
        expected_tenant = personal_tenant_id(user_id)
        current_tenant = user.tenant_id
        if current_tenant is not None and current_tenant != expected_tenant:
            report.conflicts.append(
                {
                    "user_id": user_id,
                    "kind": "existing_tenant",
                    "current_tenant_id": current_tenant,
                    "expected_tenant_id": expected_tenant,
                }
            )
            return

        source_roles = migration_roles(user)
        try:
            async with self._session_factory() as session, session.begin():
                repo = TenantMembershipRepository(session)
                tenant = await repo.get_tenant(expected_tenant)
                if tenant is None:
                    if not self._dry_run:
                        await repo.ensure_tenant(
                            expected_tenant, display_name=f"Personal tenant {user_id}"
                        )
                    report.tenants_created += 1
                membership = await repo.get_membership(expected_tenant, user_id)
                if membership is None:
                    if not self._dry_run:
                        # ``tenant_role_revision`` is the SQL tenant authority.
                        # The fallback is only for rolling Mongo documents
                        # written before that mirror field existed.
                        tenant_revision = getattr(user, "tenant_role_revision", None)
                        if tenant_revision is None:
                            tenant_revision = user.role_revision
                        membership, _ = await repo.ensure_membership(
                            expected_tenant,
                            user_id,
                            legal_roles=source_roles,
                            role_revision=max(0, int(tenant_revision)),
                        )
                    report.memberships_created += 1
                elif not self._dry_run:
                    # SQL is authoritative on retry.  Do not overwrite a
                    # tenant-admin edit with stale Mongo values.
                    source_roles = normalize_legal_roles(membership.legal_roles)
                    user.tenant_role_revision = membership.role_revision

            if self._dry_run:
                if current_tenant == expected_tenant:
                    report.already_current += 1
                else:
                    report.migrated += 1
                return

            user.tenant_id = expected_tenant
            user.legal_roles = source_roles
            await user.save()
            async with self._session_factory() as session, session.begin():
                await TenantMembershipRepository(session).clear_mirror_sync_pending(
                    expected_tenant,
                    user_id,
                    role_revision=max(0, int(getattr(user, "tenant_role_revision", 0))),
                )
            if current_tenant == expected_tenant:
                report.already_current += 1
            else:
                report.migrated += 1
        except (SQLAlchemyError, PyMongoError, TypeError, ValueError) as exc:
            exc.add_note(f"user_id={user_id}, operation=personal_tenant_migration")
            logger.bind(user_id=user_id).exception("personal_tenant_migration_failed")
            try:
                async with self._session_factory() as session, session.begin():
                    marked = await TenantMembershipRepository(session).mark_mirror_sync_pending(
                        expected_tenant, user_id, error=str(exc)
                    )
                if marked:
                    report.pending_mirror_sync += 1
            except SQLAlchemyError as marker_exc:
                marker_exc.add_note(f"user_id={user_id}, operation=mark_migration_mirror_pending")
                logger.bind(user_id=user_id).exception("personal_tenant_migration_marker_failed")
            report.failures.append(
                {"user_id": user_id, "error": type(exc).__name__, "message": str(exc)}
            )


async def migrate_personal_tenants(
    session_factory: async_sessionmaker[AsyncSession],
    *,
    dry_run: bool = False,
) -> TenantMigrationReport:
    """Convenience entry point used by the operational CLI and tests."""
    return await PersonalTenantMigration(session_factory, dry_run=dry_run).run()


async def check_tenant_readiness(
    session_factory: async_sessionmaker[AsyncSession],
    users: AsyncIterator[User] | None = None,
) -> TenantReadinessReport:
    """Check every active Mongo account, including accounts with no documents."""
    report = TenantReadinessReport()
    source = users if users is not None else iter_users()
    active_users = [user async for user in source if user.is_active]
    report.active_accounts = len(active_users)

    async with session_factory() as session:
        memberships = list((await session.scalars(select(TenantMembership))).all())
        tenants = {tenant.id for tenant in (await session.scalars(select(Tenant))).all()}

    by_user: dict[str, list[TenantMembership]] = {}
    for membership in memberships:
        by_user.setdefault(membership.user_id, []).append(membership)
    report.pending_mirror_sync.extend(
        membership.user_id for membership in memberships if membership.mirror_sync_pending
    )

    for user in active_users:
        user_id = str(user.id)
        expected_tenant = personal_tenant_id(user_id)
        user_memberships = by_user.get(user_id, [])
        if not user_memberships:
            report.missing_membership.append(user_id)
            continue
        personal = [m for m in user_memberships if m.tenant_id == expected_tenant]
        if not personal:
            report.missing_personal_tenant.append(user_id)
        elif expected_tenant not in tenants:
            report.missing_tenant.append(user_id)

        account_invalid = False
        for membership in user_memberships:
            if membership.tenant_id not in tenants:
                report.missing_tenant.append(user_id)
                account_invalid = True
            try:
                normalize_legal_roles(membership.legal_roles)
            except (TypeError, ValueError):
                account_invalid = True
        if account_invalid:
            report.invalid_membership.append(user_id)

        # Mongo is a mirror of the last selected membership, not a readiness
        # authority.  Only flag a mirror tenant that points outside SQL; a
        # valid team tenant is an acceptable last-active value.
        if user.tenant_id is not None and user.tenant_id not in {
            membership.tenant_id for membership in user_memberships
        }:
            report.mirror_mismatch.append(user_id)
        if (
            user_id not in report.pending_mirror_sync
            and user_id not in report.mirror_mismatch
            and not account_invalid
            and personal
            and expected_tenant in tenants
        ):
            report.valid_accounts += 1
    report.pending_mirror_sync = sorted(set(report.pending_mirror_sync))
    report.missing_tenant = sorted(set(report.missing_tenant))
    report.invalid_membership = sorted(set(report.invalid_membership))
    report.missing_personal_tenant = sorted(set(report.missing_personal_tenant))
    return report
