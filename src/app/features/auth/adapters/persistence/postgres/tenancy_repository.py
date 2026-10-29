"""Persistence and service boundary for tenant-scoped legal membership.

PostgreSQL is the authority for tenant membership and legal roles.  The
MongoDB ``User.legal_roles`` field is deliberately a denormalised mirror used
by the authentication hot path; callers must update it through
``TenantMembershipService`` so a failed mirror update is observable and
reconcilable rather than silently becoming a second source of truth.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from returns.result import Failure, Success
from sqlalchemy import select
from sqlalchemy.exc import SQLAlchemyError

from app.features.auth.domain.errors import (
    AuthAuthorizationError,
    AuthConflictError,
    AuthInfrastructureError,
    AuthNotFoundError,
)
from app.features.auth.domain.tenancy import LegalRole, normalize_legal_role
from app.features.auth.domain.tenancy_models import Tenant, TenantMembership
from app.observability.logging import logger

if TYPE_CHECKING:
    from collections.abc import Iterable

    from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

    from app.features.auth.adapters.persistence.mongodb.user_store import AuthAccountStore
    from app.features.auth.domain.errors import AuthResult


def normalize_legal_roles(roles: Iterable[LegalRole | str]) -> tuple[LegalRole, ...]:
    """Normalize role values into a stable, de-duplicated tuple."""
    normalized: list[LegalRole] = []
    for role in roles:
        value = normalize_legal_role(role)
        if value not in normalized:
            normalized.append(value)
    return tuple(normalized or [LegalRole.ANALYST])


class TenantMembershipRepository:
    """SQL-only repository; transaction ownership remains with the service."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def get_tenant(self, tenant_id: str) -> Tenant | None:
        return await self._session.get(Tenant, tenant_id)

    async def get_membership(self, tenant_id: str, user_id: str) -> TenantMembership | None:
        return await self._session.get(TenantMembership, (tenant_id, user_id))

    async def list_memberships_for_user(self, user_id: str) -> list[TenantMembership]:
        """Return every SQL membership for ``user_id``.

        The account document's ``tenant_id`` is a denormalised last-active
        hint.  Selection therefore starts from this relation and never from
        Mongo, otherwise a stale mirror can silently cross a tenant boundary.
        """
        result = await self._session.scalars(
            select(TenantMembership)
            .where(TenantMembership.user_id == user_id)
            .order_by(TenantMembership.tenant_id)
        )
        return list(result.all())

    async def ensure_tenant(self, tenant_id: str, *, display_name: str) -> Tenant:
        tenant = await self.get_tenant(tenant_id)
        if tenant is not None:
            return tenant
        tenant = Tenant(id=tenant_id, display_name=display_name, self_review_allowed=True)
        self._session.add(tenant)
        await self._session.flush()
        return tenant

    async def ensure_membership(
        self,
        tenant_id: str,
        user_id: str,
        *,
        legal_roles: Iterable[LegalRole | str],
        role_revision: int,
    ) -> tuple[TenantMembership, bool]:
        """Create a membership if absent; never overwrite SQL authority.

        Returns ``(membership, created)``.  Existing roles/revisions are
        preserved so rerunning a migration cannot clobber a tenant-admin edit.
        """
        membership = await self.get_membership(tenant_id, user_id)
        if membership is not None:
            return membership, False
        normalized = normalize_legal_roles(legal_roles)
        membership = TenantMembership(
            tenant_id=tenant_id,
            user_id=user_id,
            legal_role=normalized[0].value,
            legal_roles=[role.value for role in normalized],
            role_revision=max(0, role_revision),
        )
        self._session.add(membership)
        await self._session.flush()
        return membership, True

    async def set_membership_roles(
        self,
        tenant_id: str,
        user_id: str,
        *,
        legal_roles: Iterable[LegalRole | str],
        expected_revision: int | None = None,
    ) -> TenantMembership:
        membership = await self.get_membership(tenant_id, user_id)
        if membership is None:
            message = "tenant membership does not exist"
            raise LookupError(message)
        normalized = normalize_legal_roles(legal_roles)
        if expected_revision is not None and membership.role_revision != expected_revision:
            message = "tenant membership revision is stale"
            raise ValueError(message)
        values = [role.value for role in normalized]
        if membership.legal_roles != values or membership.legal_role != values[0]:
            membership.legal_roles = values
            membership.legal_role = values[0]
            membership.role_revision += 1
            await self._session.flush()
        return membership

    async def set_self_review_allowed(self, tenant_id: str, *, allowed: bool) -> Tenant:
        tenant = await self.get_tenant(tenant_id)
        if tenant is None:
            message = "tenant does not exist"
            raise LookupError(message)
        tenant.self_review_allowed = allowed
        await self._session.flush()
        return tenant

    async def list_pending_mirror_sync(self, *, limit: int) -> list[TenantMembership]:
        """Return a bounded batch of SQL memberships needing Mongo repair."""
        result = await self._session.scalars(
            select(TenantMembership)
            .where(TenantMembership.mirror_sync_pending.is_(True))
            .order_by(TenantMembership.created_at)
            .limit(limit)
        )
        return list(result.all())

    async def mark_mirror_sync_pending(self, tenant_id: str, user_id: str, *, error: str) -> bool:
        membership = await self.get_membership(tenant_id, user_id)
        if membership is None:
            return False
        membership.mirror_sync_pending = True
        membership.mirror_sync_error = error[:500]
        await self._session.flush()
        return True

    async def clear_mirror_sync_pending(
        self, tenant_id: str, user_id: str, *, role_revision: int
    ) -> bool:
        """Clear a marker only if the SQL authority has not changed meanwhile."""
        membership = await self.get_membership(tenant_id, user_id)
        if membership is None or membership.role_revision != role_revision:
            return False
        membership.mirror_sync_pending = False
        membership.mirror_sync_error = None
        await self._session.flush()
        return True


class TenantMembershipService:
    """Authoritative SQL membership service with explicit Mongo mirror sync."""

    def __init__(
        self,
        session_factory: async_sessionmaker[AsyncSession],
        account_store: AuthAccountStore | None = None,
    ) -> None:
        self._session_factory = session_factory
        self._account_store = account_store

    async def get_live_authorization(
        self,
        *,
        tenant_id: str,
        user_id: str,
    ) -> AuthResult[tuple[tuple[LegalRole, ...], bool, int]]:
        """Read the current SQL membership and tenant policy atomically.

        This is the read-side authority for long-lived requests and HITL
        resumes. Mongo account fields are a mirror only; missing SQL rows or a
        database failure fail closed rather than falling back to a personal
        tenant or stale token/account roles.
        """
        try:
            async with self._session_factory() as session:
                repository = TenantMembershipRepository(session)
                tenant = await repository.get_tenant(tenant_id)
                membership = await repository.get_membership(tenant_id, user_id)
        except SQLAlchemyError as exc:
            exc.add_note(
                f"tenant_id={tenant_id}, user_id={user_id}, operation=get_live_authorization"
            )
            logger.bind(tenant_id=tenant_id, user_id=user_id).exception(
                "tenant_authorization_read_failed"
            )
            return Failure(
                AuthInfrastructureError(
                    message="Tenant authorization read failed",
                    details={"operation": "get_live_authorization"},
                    source="tenant_membership_service",
                )
            )
        if tenant is None or membership is None:
            return Failure(
                AuthNotFoundError(
                    message="Tenant membership not found",
                    details={"tenant_id": tenant_id, "user_id": user_id},
                    source="tenant_membership_service",
                )
            )
        try:
            roles = normalize_legal_roles(membership.legal_roles)
        except (TypeError, ValueError) as exc:
            return Failure(
                AuthConflictError(
                    message="Tenant membership contains an invalid legal role",
                    details={"tenant_id": tenant_id, "user_id": user_id, "error": str(exc)},
                    source="tenant_membership_service",
                )
            )
        return Success((roles, bool(tenant.self_review_allowed), membership.role_revision))

    async def resolve_tenant(
        self,
        *,
        user_id: str,
        requested_tenant_id: str | None = None,
    ) -> AuthResult[str]:
        """Resolve the active tenant from SQL memberships.

        A single membership is unambiguous.  Multiple memberships require a
        selector supplied by the request/session boundary and that selector is
        checked against SQL.  Mongo's mirror is deliberately not consulted.
        """
        try:
            async with self._session_factory() as session:
                memberships = await TenantMembershipRepository(session).list_memberships_for_user(
                    user_id
                )
        except SQLAlchemyError as exc:
            exc.add_note(f"user_id={user_id}, operation=resolve_tenant")
            logger.bind(user_id=user_id).exception("tenant_selection_read_failed")
            return Failure(
                AuthInfrastructureError(
                    message="Tenant selection read failed",
                    details={"operation": "resolve_tenant"},
                    source="tenant_membership_service",
                )
            )
        tenant_ids = tuple(membership.tenant_id for membership in memberships)
        if requested_tenant_id is not None:
            if requested_tenant_id not in tenant_ids:
                return Failure(
                    AuthAuthorizationError(
                        message="Requested tenant is not an active membership",
                        details={"tenant_id": requested_tenant_id, "user_id": user_id},
                        source="tenant_membership_service",
                    )
                )
            return Success(requested_tenant_id)
        if len(tenant_ids) == 1:
            return Success(tenant_ids[0])
        if not tenant_ids:
            return Failure(
                AuthNotFoundError(
                    message="No tenant membership found",
                    details={"user_id": user_id},
                    source="tenant_membership_service",
                )
            )
        return Failure(
            AuthConflictError(
                message="Tenant selector is required for multiple memberships",
                details={"user_id": user_id, "membership_count": len(tenant_ids)},
                source="tenant_membership_service",
            )
        )

    async def update_roles(
        self,
        *,
        tenant_id: str,
        user_id: str,
        legal_roles: Iterable[LegalRole | str],
        expected_revision: int | None = None,
    ) -> AuthResult[tuple[LegalRole, ...]]:
        normalized = normalize_legal_roles(legal_roles)
        membership_revision = 0
        try:
            async with self._session_factory() as session, session.begin():
                membership = await TenantMembershipRepository(session).set_membership_roles(
                    tenant_id,
                    user_id,
                    legal_roles=normalized,
                    expected_revision=expected_revision,
                )
                result = tuple(LegalRole(role) for role in membership.legal_roles)
                membership_revision = membership.role_revision
        except LookupError:
            return Failure(
                AuthNotFoundError(
                    message="Tenant membership not found",
                    details={"tenant_id": tenant_id, "user_id": user_id},
                    source="tenant_membership_service",
                )
            )
        except (TypeError, ValueError) as exc:
            return Failure(
                AuthConflictError(
                    message="Tenant membership revision is stale",
                    details={"tenant_id": tenant_id, "user_id": user_id, "error": str(exc)},
                    source="tenant_membership_service",
                )
            )
        except SQLAlchemyError as exc:
            exc.add_note(f"tenant_id={tenant_id}, user_id={user_id}, operation=update_roles")
            logger.bind(tenant_id=tenant_id, user_id=user_id).exception(
                "tenant_membership_update_failed"
            )
            return Failure(
                AuthInfrastructureError(
                    message="Tenant membership update failed",
                    details={"operation": "update_roles"},
                    source="tenant_membership_service",
                )
            )

        mirror = await self._sync_mongo_mirror(
            user_id=user_id,
            tenant_id=tenant_id,
            legal_roles=result,
            role_revision=membership_revision,
        )
        if isinstance(mirror, Failure):
            return Failure(mirror.failure())
        return Success(result)

    async def set_self_review_allowed(self, *, tenant_id: str, allowed: bool) -> AuthResult[bool]:
        try:
            async with self._session_factory() as session, session.begin():
                tenant = await TenantMembershipRepository(session).set_self_review_allowed(
                    tenant_id, allowed=allowed
                )
        except LookupError:
            return Failure(
                AuthNotFoundError(
                    message="Tenant not found",
                    details={"tenant_id": tenant_id},
                    source="tenant_membership_service",
                )
            )
        except SQLAlchemyError as exc:
            exc.add_note(f"tenant_id={tenant_id}, operation=set_self_review_allowed")
            return Failure(
                AuthInfrastructureError(
                    message="Tenant setting update failed",
                    details={"operation": "set_self_review_allowed"},
                    source="tenant_membership_service",
                )
            )
        return Success(tenant.self_review_allowed)

    async def _sync_mongo_mirror(
        self,
        *,
        user_id: str,
        tenant_id: str,
        legal_roles: tuple[LegalRole, ...],
        role_revision: int,
    ) -> AuthResult[None]:
        if self._account_store is None:
            return Success(None)
        result = await self._account_store.set_tenant_binding(
            user_id,
            tenant_id=tenant_id,
            legal_roles=legal_roles,
            # New adapters consume the explicit tenant revision. Keep the
            # alias during rolling deploys for older account-store workers;
            # neither value is sourced from the platform role revision.
            tenant_role_revision=role_revision,
            role_revision=role_revision,
        )
        if isinstance(result, Failure):
            logger.bind(tenant_id=tenant_id, user_id=user_id).error(
                "tenant_membership_mongo_mirror_stale"
            )
            await self._record_mirror_failure(
                tenant_id=tenant_id,
                user_id=user_id,
                error=result.failure().message,
            )
            return Failure(result.failure())
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
            logger.bind(tenant_id=tenant_id, user_id=user_id).exception(
                "tenant_membership_mirror_marker_clear_failed"
            )
        return Success(None)

    async def _record_mirror_failure(self, *, tenant_id: str, user_id: str, error: str) -> None:
        """Persist a retry marker after SQL membership commit has succeeded."""
        try:
            async with self._session_factory() as session, session.begin():
                await TenantMembershipRepository(session).mark_mirror_sync_pending(
                    tenant_id, user_id, error=error
                )
        except SQLAlchemyError as exc:
            # The original mirror error remains the user-visible failure.  A
            # marker write failure is logged loudly so operators can repair it.
            exc.add_note(
                f"tenant_id={tenant_id}, user_id={user_id}, operation=mark_mirror_sync_pending"
            )
            logger.bind(tenant_id=tenant_id, user_id=user_id).exception(
                "tenant_membership_mirror_marker_write_failed"
            )

    async def reconcile_pending_mirrors(self, *, limit: int = 100) -> dict[str, int]:
        """Retry SQL-committed membership mirrors without changing SQL authority."""
        if limit < 1:
            message = "limit must be positive"
            raise ValueError(message)
        if self._account_store is None:
            return {"scanned": 0, "succeeded": 0, "failed": 0}
        async with self._session_factory() as session:
            pending = await TenantMembershipRepository(session).list_pending_mirror_sync(
                limit=limit
            )
        succeeded = 0
        failed = 0
        for membership in pending:
            try:
                roles = normalize_legal_roles(membership.legal_roles)
            except (TypeError, ValueError) as exc:
                await self._record_mirror_failure(
                    tenant_id=membership.tenant_id,
                    user_id=membership.user_id,
                    error=str(exc),
                )
                failed += 1
                continue
            result = await self._account_store.set_tenant_binding(
                membership.user_id,
                tenant_id=membership.tenant_id,
                legal_roles=roles,
                tenant_role_revision=membership.role_revision,
                role_revision=membership.role_revision,
            )
            if isinstance(result, Failure):
                await self._record_mirror_failure(
                    tenant_id=membership.tenant_id,
                    user_id=membership.user_id,
                    error=result.failure().message,
                )
                failed += 1
                continue
            try:
                async with self._session_factory() as session, session.begin():
                    cleared = await TenantMembershipRepository(session).clear_mirror_sync_pending(
                        membership.tenant_id,
                        membership.user_id,
                        role_revision=membership.role_revision,
                    )
                if cleared:
                    succeeded += 1
                else:
                    failed += 1
            except SQLAlchemyError as exc:
                exc.add_note(
                    f"tenant_id={membership.tenant_id}, user_id={membership.user_id}, operation=reconcile_pending_mirror"
                )
                logger.bind(tenant_id=membership.tenant_id, user_id=membership.user_id).exception(
                    "tenant_membership_mirror_reconcile_failed"
                )
                failed += 1
        return {"scanned": len(pending), "succeeded": succeeded, "failed": failed}
