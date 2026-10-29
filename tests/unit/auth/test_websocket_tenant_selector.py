"""Browser-safe WebSocket tenant selector tests."""

import pytest
from fastapi.datastructures import Headers, QueryParams

from app.features.auth.api.http.dependencies import _websocket_tenant_selector
from app.utils import ForbiddenException


class _WebSocket:
    def __init__(self, *, header: str | None = None, query: str | None = None) -> None:
        self.headers = Headers({"x-tenant-id": header} if header is not None else {})
        self.query_params = QueryParams({"tenant_id": query} if query is not None else {})


def test_websocket_selector_accepts_browser_query_parameter() -> None:
    assert _websocket_tenant_selector(_WebSocket(query="tenant-a")) == "tenant-a"


def test_websocket_selector_requires_matching_header_and_query() -> None:
    with pytest.raises(ForbiddenException, match="Conflicting tenant selectors"):
        _websocket_tenant_selector(_WebSocket(header="tenant-a", query="tenant-b"))
