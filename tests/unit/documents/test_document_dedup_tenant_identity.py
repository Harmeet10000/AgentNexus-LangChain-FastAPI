"""Document deduplication identity is ``(tenant_id, content_hash)`` — task 3.2.

The old ``(user_id, content_hash)`` unique constraint made two teammates
uploading the same contract into two documents inside one tenant, and let one
tenant's upload collide with another's only by accident of user ids. Migration
``0029_tenant_ownership_and_legal_memberships`` drops
``uq_documents_user_content_hash`` and creates ``uq_documents_tenant_content_hash``;
the dedup lookup resolves by tenant, retaining ``user_id`` solely for audit in
error details.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, cast
from uuid import uuid4

import pytest
from returns.result import Failure, Success
from sqlalchemy.dialects import postgresql

from app.features.documents.adapters.persistence.postgres.document_repository import (
    DocumentRepository,
)
from app.features.documents.domain.document import UnifiedDocument
from app.features.documents.domain.errors import DocumentNotFoundError

if TYPE_CHECKING:
    from typing import Any

    from sqlalchemy import Select

_REPO_ROOT = Path(__file__).resolve().parents[3]
_MIGRATION_SOURCE = (
    _REPO_ROOT / "src/alembic/versions/0029_tenant_ownership_and_legal_memberships.py"
).read_text(encoding="utf-8")

_TENANT = "tenant-acme"
_HASH = "a" * 64


class _DocumentResultSession:
    """Fake session answering ``scalar_one_or_none`` from compiled binds.

    A document exists for ``existing_tenant``; every other tenant (same
    content hash) reads as absent — the constraint-level identity rendered at
    the query level.
    """

    def __init__(self, existing_tenant: str) -> None:
        self.existing_tenant = existing_tenant
        self.statements: list[Select] = []
        self._row: Any | None = None

    async def execute(self, statement: Select, *_args: object) -> Any:
        self.statements.append(statement)

        class _Result:
            def __init__(self, row: UnifiedDocument | None) -> None:
                self._row = row

            def scalar_one_or_none(self) -> UnifiedDocument | None:
                return self._row

        compiled = statement.compile(dialect=postgresql.dialect())
        tenant = compiled.params.get("tenant_id_1") or compiled.params.get("tenant_id")
        row: UnifiedDocument | None = None
        if tenant == self.existing_tenant:
            # One row per tenant: the storage-level identity the unique
            # constraint enforces, so repeat lookups resolve the same document.
            if self._row is None:
                self._row = cast(
                    "Any",
                    SimpleDocument(
                        id=uuid4(),
                        tenant_id=self.existing_tenant,
                        user_id="uploader-1",
                        content_hash=_HASH,
                    ),
                )
            row = self._row
        return _Result(row)


class SimpleDocument:
    """Stand-in for the ORM entity the session would hydrate."""

    def __init__(self, **kwargs: Any) -> None:
        self.__dict__.update(kwargs)


def _compiled_binds(statement: Select) -> dict[str, Any]:
    return dict(statement.compile(dialect=postgresql.dialect()).params)


@pytest.mark.asyncio
async def test_dedup_lookup_predicate_is_tenant_and_hash_never_user() -> None:
    session = _DocumentResultSession(existing_tenant=_TENANT)
    repo = DocumentRepository(cast("Any", session))

    result = await repo.get_document_by_user_hash(
        user_id="uploader-1", tenant_id=_TENANT, content_hash=_HASH
    )

    assert isinstance(result, Success)
    statement = session.statements[-1]
    rendered = str(statement.compile(dialect=postgresql.dialect()))
    where_clause = rendered.split("WHERE", 1)[1]
    assert "tenant_id" in where_clause
    assert "content_hash" in where_clause
    assert "user_id" not in where_clause, "uploader identity must not be a dedup predicate"


@pytest.mark.asyncio
async def test_teammates_sharing_a_tenant_resolve_one_document() -> None:
    """Both users' lookups bind the same tenant + hash: one dedup identity."""
    session = _DocumentResultSession(existing_tenant=_TENANT)
    repo = DocumentRepository(cast("Any", session))

    first = await repo.get_document_by_user_hash(
        user_id="teammate-a", tenant_id=_TENANT, content_hash=_HASH
    )
    second = await repo.get_document_by_user_hash(
        user_id="teammate-b", tenant_id=_TENANT, content_hash=_HASH
    )

    assert isinstance(first, Success)
    assert isinstance(second, Success)
    assert first.unwrap() is second.unwrap(), "one tenant, one document identity"
    binds_a = _compiled_binds(session.statements[0])
    binds_b = _compiled_binds(session.statements[1])
    assert binds_a["tenant_id_1"] == binds_b["tenant_id_1"] == _TENANT
    assert binds_a["content_hash_1"] == binds_b["content_hash_1"] == _HASH
    # user_id is retained for audit only — it differs between the callers
    # without changing the resolved document.
    assert binds_a.get("user_id_1") is None
    assert binds_b.get("user_id_1") is None


@pytest.mark.asyncio
async def test_different_tenants_get_separate_identities_for_identical_content() -> None:
    """The same hash in another tenant does not resolve this tenant's document."""
    session = _DocumentResultSession(existing_tenant=_TENANT)
    repo = DocumentRepository(cast("Any", session))

    own = await repo.get_document_by_user_hash(
        user_id="uploader-1", tenant_id=_TENANT, content_hash=_HASH
    )
    foreign = await repo.get_document_by_user_hash(
        user_id="uploader-1", tenant_id="tenant-other", content_hash=_HASH
    )

    assert isinstance(own, Success)
    assert isinstance(foreign, Failure)
    error = foreign.failure()
    assert isinstance(error, DocumentNotFoundError)
    # user_id survives as audit detail, never as scope.
    assert error.details["user_id"] == "uploader-1"
    binds = _compiled_binds(session.statements[-1])
    assert binds["tenant_id_1"] == "tenant-other"


def test_migration_drops_the_user_constraint_and_creates_the_tenant_constraint() -> None:
    assert 'op.drop_constraint("uq_documents_user_content_hash", "documents", type_="unique")' in (
        _MIGRATION_SOURCE
    )
    assert (
        "op.create_unique_constraint(\n"
        '        "uq_documents_tenant_content_hash", "documents", ["tenant_id", "content_hash"]\n'
        "    )" in _MIGRATION_SOURCE
    )
    # Tenant read paths get their indexes in the same migration.
    assert 'op.create_index("ix_documents_tenant_id", "documents", ["tenant_id"])' in (
        _MIGRATION_SOURCE
    )
    assert (
        'op.create_index("ix_chunks_tenant_document", "chunks", ["tenant_id", "document_id"])'
        in (_MIGRATION_SOURCE)
    )


def test_domain_model_declares_only_the_tenant_unique_constraint() -> None:
    constraint_names = {constraint.name for constraint in UnifiedDocument.__table__.constraints}
    assert "uq_documents_tenant_content_hash" in constraint_names
    assert "uq_documents_user_content_hash" not in constraint_names
