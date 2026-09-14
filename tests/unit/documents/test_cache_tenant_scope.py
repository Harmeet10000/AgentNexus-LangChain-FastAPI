"""Tenant identity is part of every document-response cache key."""

from app.features.documents.dto import UnifiedSearchRequest
from app.features.documents.service import _build_answer_cache_key, _build_cache_key


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
