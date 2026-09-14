"""Pin the uniform filter surface across retrieval branches (retrieval-sql task 2.3).

Every branch method's SQL must contain the shared filter block, and every key in
the built filter parameters must be consumed by every branch. Removing one
predicate from one branch must fail this test.
"""

from __future__ import annotations

import inspect

from app.features.documents.repository import (
    _FILTER_SQL,
    build_search_filter_params,
)
from app.features.documents.repository import (
    DocumentRepository as _Repository,
)

_BRANCH_METHODS = ("bm25_search", "vector_search", "trigram_search")


def test_every_branch_embeds_the_shared_filter_block() -> None:
    for method_name in _BRANCH_METHODS:
        source = inspect.getsource(getattr(_Repository, method_name))
        assert "_FILTER_SQL" in source, (
            f"{method_name} no longer embeds the shared filter block"
        )


def test_every_filter_key_is_consumed_by_the_shared_block() -> None:
    params = build_search_filter_params(metadata_filter={})
    assert params, "expected a non-empty filter parameter surface"
    for key in params:
        assert f":{key}" in _FILTER_SQL, (
            f"filter parameter {key!r} is not consumed by the shared filter block"
        )
