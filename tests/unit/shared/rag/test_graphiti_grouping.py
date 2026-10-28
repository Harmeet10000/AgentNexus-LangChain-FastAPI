"""Canonical Graphiti scope identity tests."""

from __future__ import annotations

import pytest

from app.retrieval.graphiti.grouping import (
    build_graphiti_group_id,
    parse_graphiti_group_id,
)


def test_group_id_round_trips_the_full_scope_tuple() -> None:
    group_id = build_graphiti_group_id(
        tenant_id="tenant/acme",
        document_id="doc:42",
        document_version=7,
        contract_index=3,
    )

    assert set(group_id) <= set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-")
    assert parse_graphiti_group_id(group_id).tenant_id == "tenant/acme"
    assert parse_graphiti_group_id(group_id).document_id == "doc:42"
    assert parse_graphiti_group_id(group_id).document_version == 7
    assert parse_graphiti_group_id(group_id).contract_index == 3


def test_contract_and_version_changes_never_share_a_group() -> None:
    base = build_graphiti_group_id(
        tenant_id="tenant-a", document_id="doc-a", document_version=1, contract_index=0
    )
    other_contract = build_graphiti_group_id(
        tenant_id="tenant-a", document_id="doc-a", document_version=1, contract_index=1
    )
    other_version = build_graphiti_group_id(
        tenant_id="tenant-a", document_id="doc-a", document_version=2, contract_index=0
    )

    assert len({base, other_contract, other_version}) == 3


def test_tenant_and_document_isolation_are_encoded() -> None:
    first = build_graphiti_group_id(
        tenant_id="tenant-a", document_id="doc-a", document_version=1
    )
    different_tenant = build_graphiti_group_id(
        tenant_id="tenant-b", document_id="doc-a", document_version=1
    )
    different_document = build_graphiti_group_id(
        tenant_id="tenant-a", document_id="doc-b", document_version=1
    )

    assert len({first, different_tenant, different_document}) == 3


def test_group_id_rejects_non_canonical_provider_values() -> None:
    with pytest.raises(ValueError, match="canonical"):
        parse_graphiti_group_id("tenant:tenant-a:document:doc-a")
