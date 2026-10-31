"""Ingest request size bounds: the DoS fence, not style.

Every accepted body becomes an outbox row, a broker message, and O(content)
metered embedding calls. An unbounded body converts one anonymous (well:
authenticated) POST into storage, queue, and provider spend — so the content
length and the serialized metadata size are capped before validation succeeds.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.features.search.api.http.dto import (
    MAX_INGEST_CONTENT_CHARS,
    MAX_INGEST_METADATA_BYTES,
    SearchIngestRequest,
)


def _ingest(**overrides: object) -> SearchIngestRequest:
    values: dict[str, object] = {"title": "Site page", "content": "site body"}
    values.update(overrides)
    return SearchIngestRequest(**values)


def test_content_within_the_cap_is_accepted() -> None:
    request = _ingest(content="a" * (MAX_INGEST_CONTENT_CHARS // 2))
    assert len(request.content) == MAX_INGEST_CONTENT_CHARS // 2


def test_content_over_the_cap_is_rejected() -> None:
    with pytest.raises(ValidationError, match="content"):
        _ingest(content="a" * (MAX_INGEST_CONTENT_CHARS + 1))


def test_metadata_within_the_cap_is_accepted() -> None:
    request = _ingest(doc_metadata={"lang": "en", "section": "pricing"})
    assert request.doc_metadata == {"lang": "en", "section": "pricing"}


def test_metadata_over_the_serialized_cap_is_rejected() -> None:
    with pytest.raises(ValidationError, match="doc_metadata"):
        _ingest(doc_metadata={"blob": "x" * (MAX_INGEST_METADATA_BYTES + 1)})


def test_deeply_nested_metadata_is_caught_by_the_serialized_cap() -> None:
    """Depth alone would dodge a shallow check; the byte cap catches it."""
    nested: dict[str, object] = {"leaf": True}
    for _ in range(500):
        nested = {"n": nested}
    with pytest.raises(ValidationError, match="doc_metadata"):
        _ingest(doc_metadata=nested)


def test_empty_metadata_is_accepted() -> None:
    assert _ingest().doc_metadata == {}
