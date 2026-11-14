"""All retrieval callers share the hosted reranker configuration."""

import inspect

import pytest

from app.config.settings import Settings, get_settings
from app.features.documents.retrieval.reranking.reranker import (
    HostedReranker,
    get_configured_reranker,
    get_shared_reranker,
)


@pytest.fixture(autouse=True)
def _cold_singleton():
    get_shared_reranker.cache_clear()
    get_settings.cache_clear()
    yield
    get_shared_reranker.cache_clear()
    get_settings.cache_clear()


def test_accessor_returns_one_hosted_instance() -> None:
    shared = get_shared_reranker()
    assert isinstance(shared, HostedReranker)
    assert shared is get_shared_reranker()
    assert get_configured_reranker() is shared


def test_rerank_flag_defaults_on() -> None:
    # Pristine env: the served-path gate is ON per the recorded rerank
    # measurement (see the settings comment); this test locks the default so
    # an accidental flip is a visible decision, not a silent one.
    assert Settings(_env_file=None).DOCUMENTS_RERANK_ENABLED is True


def test_flag_off_accessor_returns_none(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DOCUMENTS_RERANK_ENABLED", "false")
    assert get_configured_reranker() is None


def test_no_local_model_constructor_remains_at_call_sites() -> None:
    from app.features.documents.application.queries import documents as documents_service
    from app.features.documents.retrieval.graph import nodes as retrieval_nodes

    source = inspect.getsource(documents_service) + inspect.getsource(retrieval_nodes)
    assert "CrossEncoder" not in source
