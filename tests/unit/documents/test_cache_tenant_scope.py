"""Tenant identity and document/pipeline versions are part of every cache key.

The harden change (3.10) requires that no cached search or answer survives a
document version change, a pipeline version change, or a tenant boundary: a
stale-version reuse would serve evidence extracted from superseded source
text, and a cross-tenant reuse would serve another tenant's corpus.
"""

import pytest

import app.features.documents.application.queries.documents as queries_documents
from app.features.documents.api.http.dto import UnifiedSearchRequest
from app.features.documents.application.queries.documents import (
    _build_answer_cache_key,
    _build_cache_key,
)


def test_search_cache_key_is_tenant_scoped() -> None:
    payload = UnifiedSearchRequest(query="same legal question")

    first = _build_cache_key("documents:search", payload, user_id="tenant-a")
    second = _build_cache_key("documents:search", payload, user_id="tenant-b")

    assert first != second


def test_answer_cache_key_is_tenant_scoped() -> None:
    common = {
        "query": "same legal question",
        "doc_ids_filter": [],
        "jurisdiction": None,
        "contract_type": None,
        "clause_type": None,
        "require_graphiti_verified": False,
    }

    first = _build_answer_cache_key(user_id="tenant-a", **common)
    second = _build_answer_cache_key(user_id="tenant-b", **common)

    assert first != second


# ---------------------------------------------------------------------------
# 3.10 — stale-version isolation over BOTH cache windows
# ---------------------------------------------------------------------------


def test_search_cache_key_changes_when_a_document_version_changes() -> None:
    payload = UnifiedSearchRequest(query="same legal question")
    common = {
        "user_id": "user-a",
        "tenant_id": "tenant-a",
        "allowed_doc_ids": ("doc-1",),
    }

    version_one = _build_cache_key(
        "documents:search",
        payload,
        document_versions=[("doc-1", 1, "documents-v1")],
        **common,
    )
    version_two = _build_cache_key(
        "documents:search",
        payload,
        document_versions=[("doc-1", 2, "documents-v1")],
        **common,
    )

    assert version_one != version_two


def test_answer_cache_key_changes_when_a_document_version_changes() -> None:
    common = {
        "query": "same legal question",
        "doc_ids_filter": [],
        "jurisdiction": None,
        "contract_type": None,
        "clause_type": None,
        "require_graphiti_verified": False,
        "user_id": "user-a",
        "tenant_id": "tenant-a",
        "allowed_doc_ids": ("doc-1",),
    }

    version_one = _build_answer_cache_key(
        document_versions=[("doc-1", 1, "documents-v1")], **common
    )
    version_two = _build_answer_cache_key(
        document_versions=[("doc-1", 2, "documents-v1")], **common
    )

    assert version_one != version_two


@pytest.mark.parametrize("builder", ["search", "answer"])
def test_cache_keys_change_when_the_pipeline_version_changes(
    monkeypatch: pytest.MonkeyPatch, builder: str
) -> None:
    """The pipeline version participates in every key: chunks produced by an
    older extraction pipeline must never serve a cached answer/search."""
    payload = UnifiedSearchRequest(query="same legal question")

    def build() -> str:
        if builder == "search":
            return _build_cache_key(
                "documents:search",
                payload,
                user_id="user-a",
                tenant_id="tenant-a",
            )
        return _build_answer_cache_key(
            query="same legal question",
            doc_ids_filter=[],
            jurisdiction=None,
            contract_type=None,
            clause_type=None,
            require_graphiti_verified=False,
            user_id="user-a",
            tenant_id="tenant-a",
        )

    monkeypatch.setattr(queries_documents, "CANONICAL_PIPELINE_VERSION", "documents-v1")
    first = build()
    monkeypatch.setattr(queries_documents, "CANONICAL_PIPELINE_VERSION", "documents-v2")
    second = build()

    assert first != second


def test_answer_cache_key_changes_when_a_document_leaves_the_allowed_scope() -> None:
    """A scope narrowing must not reuse a broader cached answer (3.10)."""
    common = {
        "query": "same legal question",
        "doc_ids_filter": [],
        "jurisdiction": None,
        "contract_type": None,
        "clause_type": None,
        "require_graphiti_verified": False,
        "user_id": "user-a",
        "tenant_id": "tenant-a",
        "document_versions": [("doc-1", 1, None), ("doc-2", 1, None)],
    }

    wide = _build_answer_cache_key(allowed_doc_ids=("doc-1", "doc-2"), **common)
    narrowed = _build_answer_cache_key(allowed_doc_ids=("doc-1",), **common)

    assert wide != narrowed
