"""All retrieval callers share the hosted reranker configuration."""

import inspect

import pytest

from app.shared.langgraph_layer.retrieval_kb.reranker import (
    HostedReranker,
    get_configured_reranker,
    get_shared_reranker,
)


@pytest.fixture(autouse=True)
def _cold_singleton():
    get_shared_reranker.cache_clear()
    yield
    get_shared_reranker.cache_clear()


def test_accessor_returns_one_hosted_instance() -> None:
    shared = get_shared_reranker()
    assert isinstance(shared, HostedReranker)
    assert shared is get_shared_reranker()
    assert get_configured_reranker() is shared


def test_no_local_model_constructor_remains_at_call_sites() -> None:
    from app.features.documents import service as documents_service
    from app.shared.langgraph_layer.retrieval_kb import nodes as retrieval_nodes

    source = inspect.getsource(documents_service) + inspect.getsource(retrieval_nodes)
    assert "CrossEncoder" not in source
