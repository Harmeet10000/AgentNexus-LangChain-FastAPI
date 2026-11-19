"""Query-level two-tenant isolation over every DocumentRepository read — task 3.3.

Each read predicate is driven twice — once for tenant A, once for tenant B —
with the *same* document/chunk ids probed cross-tenant. A predicate that forgot
its tenant scope would compile the same statement for both callers; the suite
fails the moment any statement binds, or renders, the other tenant. Allowlist
enforcement is exercised on the chunk fetch (evidence reads), which must bind
the caller's own allowlist and never the other tenant's document ids.
"""

from __future__ import annotations

import inspect
from typing import TYPE_CHECKING, cast

import pytest
from returns.result import Success
from sqlalchemy import TextClause
from sqlalchemy.dialects import postgresql

from app.features.documents.adapters.persistence.postgres.document_repository import (
    DocumentRepository,
    build_search_filter_params,
)

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable
    from typing import Any


_TENANT_A = "personal:alice"
_TENANT_B = "personal:bob"
_DOC_A = "aaaaaaaa-1111-1111-1111-111111111111"
_DOC_B = "bbbbbbbb-2222-2222-2222-222222222222"
_CHUNK_A = "cccccccc-1111-1111-1111-111111111111"
_EMBEDDING = [0.1] * 4

_READ_PREDICATES: tuple[
    tuple[str, Callable[[DocumentRepository, str], Awaitable[Any]], str], ...
] = (
    # (name, call, probed id that must never leak the other tenant)
    (
        "get_document_by_id",
        lambda repo, tenant: repo.get_document_by_id(
            user_id="alice", tenant_id=tenant, document_id=_DOC_A
        ),
        _DOC_B,
    ),
    (
        "get_document_by_user_hash",
        lambda repo, tenant: repo.get_document_by_user_hash(
            user_id="alice", tenant_id=tenant, content_hash="a" * 64
        ),
        _DOC_B,
    ),
    (
        "fetch_status",
        lambda repo, tenant: repo.fetch_status(
            user_id="alice", tenant_id=tenant, document_id=_DOC_A
        ),
        _DOC_B,
    ),
    (
        "fetch_structural_trees",
        lambda repo, tenant: repo.fetch_structural_trees(
            user_id="alice", tenant_id=tenant, document_ids=[_DOC_A]
        ),
        _DOC_B,
    ),
    (
        "resolve_scope_versions",
        lambda repo, tenant: repo.resolve_scope_versions(tenant_id=tenant, document_ids=[_DOC_A]),
        _DOC_B,
    ),
    (
        "resolve_scope_contracts",
        lambda repo, tenant: repo.resolve_scope_contracts(tenant_id=tenant, document_ids=[_DOC_A]),
        _DOC_B,
    ),
    (
        "find_document_ids_by_metadata",
        lambda repo, tenant: repo.find_document_ids_by_metadata(
            user_id="alice",
            tenant_id=tenant,
            jurisdictions=["India"],
            document_kinds=[],
            matters=[],
            limit=10,
        ),
        _DOC_B,
    ),
    (
        "list_document_ids_for_tenant",
        lambda repo, tenant: repo.list_document_ids_for_tenant(tenant_id=tenant),
        _DOC_B,
    ),
    (
        "bm25_search",
        lambda repo, tenant: repo.bm25_search(
            user_id="alice",
            tenant_id=tenant,
            query="indemnity",
            candidate_limit=5,
            filter_params=build_search_filter_params(metadata_filter={"document_ids": [_DOC_A]}),
        ),
        _DOC_B,
    ),
    (
        "vector_search",
        lambda repo, tenant: repo.vector_search(
            user_id="alice",
            tenant_id=tenant,
            embedding=_EMBEDDING,
            candidate_limit=5,
            filter_params=build_search_filter_params(metadata_filter={"document_ids": [_DOC_A]}),
        ),
        _DOC_B,
    ),
    (
        "trigram_search",
        lambda repo, tenant: repo.trigram_search(
            user_id="alice",
            tenant_id=tenant,
            query="indemnity",
            candidate_limit=5,
            filter_params=build_search_filter_params(metadata_filter={"document_ids": [_DOC_A]}),
        ),
        _DOC_B,
    ),
    (
        "fetch_chunks_by_ids",
        lambda repo, tenant: repo.fetch_chunks_by_ids(
            [_CHUNK_A],
            tenant_id=tenant,
            allowed_doc_ids=[_DOC_A],
        ),
        _DOC_B,
    ),
)


class _IsolationCaptureSession:
    """Records every executed statement plus its runtime binds."""

    def __init__(self) -> None:
        self.executions: list[tuple[Any, dict[str, Any] | None]] = []

    async def execute(self, statement: Any, params: dict[str, Any] | None = None) -> Any:
        self.executions.append((statement, dict(params) if params is not None else None))

        class _Rows:
            def mappings(self) -> _Rows:
                return self

            def all(self) -> list[dict[str, Any]]:
                return []

            def one_or_none(self) -> dict[str, Any] | None:
                return None

            def scalar_one_or_none(self) -> None:
                return None

        return _Rows()


def _bound_values(execution: tuple[Any, dict[str, Any] | None]) -> list[Any]:
    statement, params = execution
    if isinstance(statement, TextClause):
        return list((params or {}).values())
    compiled = statement.compile(dialect=postgresql.dialect())
    return list(compiled.params.values())


def _rendered(execution: tuple[Any, dict[str, Any] | None]) -> str:
    statement, _params = execution
    if isinstance(statement, TextClause):
        return str(statement)
    return str(statement.compile(dialect=postgresql.dialect()))


@pytest.mark.parametrize(("name", "call", "foreign_id"), _READ_PREDICATES)
@pytest.mark.asyncio
async def test_read_predicate_binds_only_the_calling_tenant(
    name: str,
    call: Callable[[DocumentRepository, str], Awaitable[Any]],
    foreign_id: str,
) -> None:
    executions_by_tenant: dict[str, list[tuple[Any, dict[str, Any] | None]]] = {}
    for tenant in (_TENANT_A, _TENANT_B):
        session = _IsolationCaptureSession()
        repo = DocumentRepository(cast("Any", session))
        await call(repo, tenant)
        executions_by_tenant[tenant] = session.executions

        assert session.executions, f"{name} issued no statement"
        # A statement with NO tenant bind at all used to be skipped silently —
        # which is exactly the regression this suite exists to catch (a dropped
        # tenant predicate removes the bind too). Only session-scoped pragmas
        # (DiskANN's SET LOCAL) may carry no scope binds; every other statement
        # must bind the caller's tenant, and each call must check at least one
        # real read.
        checked_reads = 0
        for execution in session.executions:
            rendered = _rendered(execution)
            if "SET LOCAL" in rendered:
                continue  # session-scoped pragma — carries no per-call scope binds
            values = _bound_values(execution)
            checked_reads += 1
            assert tenant in values, f"{name} did not bind the caller's tenant for {tenant}"
            other = _TENANT_B if tenant == _TENANT_A else _TENANT_A
            assert other not in values, f"{name} bound the other tenant for {tenant}"
            assert other not in rendered, f"{name} rendered the other tenant for {tenant}"
            assert foreign_id not in rendered, f"{name} rendered the other tenant's document"
        assert checked_reads > 0, (
            f"{name}: every issued statement was skipped as scope-free — "
            "the read's tenant predicate is missing entirely"
        )


def test_chunk_fetch_tenant_scope_is_required_not_optional() -> None:
    """The hydration read can never run without a tenant predicate again."""
    parameter = inspect.signature(DocumentRepository.fetch_chunks_by_ids).parameters["tenant_id"]
    assert parameter.kind is inspect.Parameter.KEYWORD_ONLY
    assert parameter.default is inspect.Parameter.empty


@pytest.mark.asyncio
async def test_chunk_fetch_probed_with_foreign_ids_binds_caller_scope_only() -> None:
    """Same chunk ids probed from both tenants: each statement pins its own
    tenant and its own allowlist — cross-tenant probes match no rows."""
    executions_by_tenant: dict[str, list[tuple[Any, dict[str, Any] | None]]] = {}
    for tenant, own_doc in ((_TENANT_A, _DOC_A), (_TENANT_B, _DOC_B)):
        session = _IsolationCaptureSession()
        repo = DocumentRepository(cast("Any", session))
        result = await repo.fetch_chunks_by_ids(
            [_CHUNK_A],
            tenant_id=tenant,
            allowed_doc_ids=[own_doc],
        )
        assert isinstance(result, Success)
        executions_by_tenant[tenant] = session.executions

    statement_a, _ = executions_by_tenant[_TENANT_A][-1]
    statement_b, _ = executions_by_tenant[_TENANT_B][-1]
    binds_a = dict(statement_a.compile(dialect=postgresql.dialect()).params)
    binds_b = dict(statement_b.compile(dialect=postgresql.dialect()).params)

    def _flat(values: dict[str, Any]) -> list[Any]:
        flattened: list[Any] = []
        for value in values.values():
            if isinstance(value, (list, tuple)):
                flattened.extend(value)
            else:
                flattened.append(value)
        return flattened

    assert binds_a != binds_b, "cross-tenant probes must compile distinct scopes"
    flat_a, flat_b = _flat(binds_a), _flat(binds_b)
    assert _TENANT_A in flat_a
    assert _TENANT_B not in flat_a
    assert _TENANT_B in flat_b
    assert _TENANT_A not in flat_b
    assert _DOC_A in flat_a
    assert _DOC_B not in flat_a
    assert _DOC_B in flat_b
    assert _DOC_A not in flat_b
    # The allowlist predicate is part of the statement shape, not a post-filter.
    rendered_a = str(statement_a.compile(dialect=postgresql.dialect()))
    assert "ANY" in rendered_a
    assert "document_id" in rendered_a
