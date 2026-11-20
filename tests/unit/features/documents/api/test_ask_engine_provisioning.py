"""Fail-closed provisioning of the ask endpoints' retrieval engine (task 4.2).

When the compile-once retrieval engine is not provisioned in the serving
process, both ask endpoints return a typed unavailability rather than falling
back to any other answering path. The app fixture is hand-built for the same
reason ``test_unprovisioned_graph_fails_closed.py`` builds its own: the real
factory opens connections; the registration surface under test is two
dependency overrides away from exact.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING, cast

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.features.documents.api.http import dependencies as documents_dependencies
from app.features.documents.api.http.router import router
from app.features.documents.application.queries.documents import DocumentQueryService
from app.middleware import register_exception_handlers
from app.utils import ServiceUnavailableException

if TYPE_CHECKING:
    from collections.abc import Iterator
    from typing import Any

pytestmark = pytest.mark.unit

_ENGINE_CAPABILITY = "retrieval_engine"


class _StubRepo:
    """Never reached on the unprovisioned path; present for construction."""


def _engineless_service() -> DocumentQueryService:
    return DocumentQueryService(
        cast("Any", _StubRepo()),
        cast("Any", lambda: None),
        None,
        None,
    )


def _authorization() -> Any:
    """A minimal server-owned capability: the user owns one document.

    An unfiltered request must still resolve to a non-empty tenant allowlist
    (the real ``narrow_document_ids`` returns the tenant's document ids): an
    empty allowlist is the explicit empty-scope deny, which returns the
    grounded fallback envelope — a different contract from the unprovisioned
    engine this test pins.
    """
    return SimpleNamespace(
        user_id="user-1",
        tenant_id="tenant-a",
        narrow_document_ids=lambda requested: tuple(dict.fromkeys(requested)) or ("doc-a",),
    )


@pytest.fixture
def client() -> Iterator[TestClient]:
    app = FastAPI()
    register_exception_handlers(app)

    # The Redis-backed rate limiter reads app.state.redis and registers its
    # GCRA script on it; an inert double keeps the limiter transparent.
    class _InertRedis:
        def register_script(self, _script: Any) -> Any:
            async def _run(*_args: Any, **_kwargs: Any) -> tuple[bool, int]:
                return (True, 0)

            return _run

    app.state.redis = _InertRedis()
    app.include_router(router)
    # Override keys are the dependency callables, not the ``Annotated`` aliases:
    # FastAPI matches override keys against the callable inside ``Depends``,
    # so an alias key silently never applies (verified against fastapi 0.143)
    # and the real dependency graph — which opens a database session this
    # fixture does not have — would answer with a 500 before the service runs.
    app.dependency_overrides[documents_dependencies.get_document_query_service] = (
        _engineless_service
    )
    app.dependency_overrides[documents_dependencies.get_document_authorization] = _authorization
    with TestClient(app, raise_server_exceptions=False) as test_client:
        yield test_client


@pytest.mark.parametrize("path", ["/search/ask", "/legal/ask"])
def test_ask_endpoints_fail_closed_without_a_provisioned_engine(
    client: TestClient, path: str
) -> None:
    response = client.post(path, json={"query": "what termination rights apply?"})

    assert response.status_code == 503, (
        f"expected a typed service-unavailable from {path}, got {response.status_code}"
    )
    body: dict[str, Any] = response.json()
    assert body["success"] is False
    assert body["error"]["data"] == {"capability": _ENGINE_CAPABILITY}


async def test_the_service_raises_the_typed_exception_rather_than_returning_none() -> None:
    """Pinned at the function level as well as through HTTP, including the
    scoped (non-empty) request that would otherwise reach the engine."""
    from app.features.documents.api.http.dto import UnifiedAskRequest

    service = _engineless_service()
    with pytest.raises(ServiceUnavailableException) as caught:
        await service.ask(
            user_id="user-1",
            tenant_id="tenant-a",
            allowed_doc_ids=["doc-a"],
            payload=UnifiedAskRequest(query="what termination rights apply?"),
            require_graphiti_verified=False,
        )
    assert caught.value.status_code == 503
    assert caught.value.data == {"capability": _ENGINE_CAPABILITY}
