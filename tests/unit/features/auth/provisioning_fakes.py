"""In-memory SQL doubles for the personal-tenant provisioning tests.

The fakes mirror the exact statement shapes
``PersonalTenantProvisioningRepository`` issues (PostgreSQL ``INSERT ... ON
CONFLICT DO NOTHING`` upserts and one counted select) so unit tests can assert
commit/rollback/idempotency semantics without a live database.  Writes are
staged per transaction and only merged into the store on a successful commit,
so simulated commit failures leave no state behind — the same contract the
real ``session.begin()`` context gives the provisioner.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from sqlalchemy import Insert, Select
from sqlalchemy.exc import SQLAlchemyError

from app.features.auth.domain.tenancy_models import Tenant, TenantMembership

if TYPE_CHECKING:
    from typing import Any


def _insert_values(statement: Insert) -> dict[str, Any]:
    return {
        column.key: getattr(value, "value", value)
        for column, value in statement._values.items()
    }


class FakeExecuteResult:
    """Minimal Result facade for the statements the fakes support."""

    def __init__(self, *, rowcount: int = 1, scalar: Any = None) -> None:
        self._rowcount = rowcount
        self._scalar = scalar

    @property
    def rowcount(self) -> int:
        return self._rowcount

    def scalar_one(self) -> Any:
        assert self._scalar is not None, "scalar_one requested without a scalar"
        return self._scalar


class FakeTransaction:
    def __init__(self, session: FakeProvisioningSession) -> None:
        self._session = session

    async def __aenter__(self) -> FakeTransaction:
        return self

    async def __aexit__(self, exc_type: object, exc: object, tb: object) -> None:
        await self._session._finish(exc_type is None)


class FakeProvisioningSession:
    """AsyncSession double staging writes until the transaction commits."""

    def __init__(self, store: FakeTenantSQLStore) -> None:
        self.store = store
        self._staged_tenants: dict[str, Tenant] = {}
        self._staged_memberships: dict[tuple[str, str], TenantMembership] = {}

    async def __aenter__(self) -> FakeProvisioningSession:
        return self

    async def __aexit__(self, exc_type: object, exc: object, tb: object) -> None:
        # A plain session exit closes the session without committing: staged
        # writes are discarded and the simulated commit failure is reserved
        # for the transaction boundary opened by ``begin()``.
        self._staged_tenants.clear()
        self._staged_memberships.clear()

    def begin(self) -> FakeTransaction:
        return FakeTransaction(self)

    async def _finish(self, success: bool) -> None:
        if success and self.store.fail_next_commit:
            self.store.fail_next_commit = False
            self._staged_tenants.clear()
            self._staged_memberships.clear()
            commit_error = SQLAlchemyError("simulated PostgreSQL commit failure")
            raise commit_error
        if success:
            self.store.tenants.update(self._staged_tenants)
            self.store.memberships.update(self._staged_memberships)
        self._staged_tenants.clear()
        self._staged_memberships.clear()

    async def execute(self, statement: object) -> FakeExecuteResult:
        if self.store.yield_between_statements:
            # Give the event loop a chance to interleave concurrent provisioners.
            await asyncio.sleep(0)
        if isinstance(statement, Insert):
            values = _insert_values(statement)
            table = statement.table.name
            if table == Tenant.__tablename__:
                tenant_id = str(values["id"])
                if tenant_id not in self._all_tenants():
                    self._staged_tenants[tenant_id] = Tenant(
                        id=tenant_id,
                        display_name=str(values["display_name"]),
                        self_review_allowed=bool(values["self_review_allowed"]),
                    )
                    return FakeExecuteResult(rowcount=1)
                return FakeExecuteResult(rowcount=0)
            if table == TenantMembership.__tablename__:
                key = (str(values["tenant_id"]), str(values["user_id"]))
                if key not in self._all_memberships():
                    self._staged_memberships[key] = TenantMembership(
                        tenant_id=key[0],
                        user_id=key[1],
                        legal_role=str(values["legal_role"]),
                        legal_roles=list(values["legal_roles"]),
                        role_revision=int(values["role_revision"]),
                        mirror_sync_pending=bool(values["mirror_sync_pending"]),
                    )
                    return FakeExecuteResult(rowcount=1)
                return FakeExecuteResult(rowcount=0)
            insert_target_error = AssertionError(f"unexpected insert target {table}")
            raise insert_target_error
        if isinstance(statement, Select):
            if self.store.fail_select:
                read_error = SQLAlchemyError("simulated PostgreSQL read failure")
                raise read_error
            user_id = getattr(statement.whereclause.right, "value", None)
            assert isinstance(user_id, str)
            count = sum(1 for _, uid in self._all_memberships() if uid == user_id)
            return FakeExecuteResult(scalar=count)
        statement_error = AssertionError(f"unexpected statement type {type(statement).__name__}")
        raise statement_error

    async def get(self, model: type, key: object) -> Tenant | TenantMembership | None:
        if model is Tenant:
            assert isinstance(key, str)
            return self._all_tenants().get(key)
        assert isinstance(key, tuple)
        return self._all_memberships().get((key[0], key[1]))

    async def scalars(self, statement: object) -> FakeScalarResult:
        """Emulate ``session.scalars(select(...))`` for the read paths the
        ``TenantMembershipRepository`` uses (membership listing, reconcile batch)."""
        if self.store.yield_between_statements:
            await asyncio.sleep(0)
        if self.store.fail_select:
            read_error = SQLAlchemyError("simulated PostgreSQL read failure")
            raise read_error
        model = statement.column_descriptions[0]["entity"]
        if model is Tenant:
            rows: list[object] = list(self._all_tenants().values())
            return FakeScalarResult(rows)
        rows = list(self._all_memberships().values())
        for key, value in _where_conditions(getattr(statement, "whereclause", None)).items():
            if key == "user_id":
                rows = [row for row in rows if row.user_id == value]
            elif key == "mirror_sync_pending":
                rows = [row for row in rows if row.mirror_sync_pending is value]
            else:
                filter_error = AssertionError(f"unsupported fake filter on {key}")
                raise filter_error
        limit = getattr(statement, "_limit", None)
        if limit is not None:
            rows = rows[: int(getattr(limit, "value", limit))]
        return FakeScalarResult(rows)

    async def flush(self) -> None:
        if self.store.yield_between_statements:
            await asyncio.sleep(0)

    def _all_tenants(self) -> dict[str, Tenant]:
        return {**self.store.tenants, **self._staged_tenants}

    def _all_memberships(self) -> dict[tuple[str, str], TenantMembership]:
        return {**self.store.memberships, **self._staged_memberships}


class FakeScalarResult:
    def __init__(self, rows: list[object]) -> None:
        self._rows = rows

    def all(self) -> list[object]:
        return list(self._rows)


def _where_conditions(whereclause: object) -> dict[str, object]:
    """Extract ``column == literal`` pairs from simple SQLAlchemy where clauses."""
    conditions: dict[str, object] = {}
    if whereclause is None:
        return conditions
    clauses = getattr(whereclause, "clauses", None)
    expressions = list(clauses) if clauses is not None else [whereclause]
    for expression in expressions:
        left = getattr(expression, "left", None)
        right = getattr(expression, "right", None)
        key = getattr(left, "key", None)
        if key is None:
            where_error = AssertionError(f"unsupported fake where expression: {expression!r}")
            raise where_error
        conditions[key] = getattr(right, "value", None)
    return conditions


class FakeTenantSQLStore:
    """State backing :class:`FakeProvisioningSession`."""

    def __init__(self) -> None:
        self.tenants: dict[str, Tenant] = {}
        self.memberships: dict[tuple[str, str], TenantMembership] = {}
        self.fail_next_commit = False
        self.fail_select = False
        self.yield_between_statements = False

    def factory(self) -> FakeProvisioningSession:
        return FakeProvisioningSession(self)
