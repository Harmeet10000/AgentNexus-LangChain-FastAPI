"""Build provenance surfacing: /health must name the code it serves.

The 2026-11-15 smoke (F3) found `git_sha: "unknown"` on every payload of a
native (non-docker) deployment, making the served commit unreconstructable
while branches were switched underneath the running service. Resolution order:
baked build metadata (`GIT_SHA` env/docker build-arg), then the import-time
git read, then a truthful "unknown".
"""

from __future__ import annotations

import pytest

from app.config import get_settings
from app.features.health.adapters.inbound.http import build_info
from app.features.health.adapters.inbound.http.build_info import resolve_git_sha
from app.kernel.result import DependencyHealth

pytestmark = pytest.mark.unit


@pytest.fixture
def _fresh_git_cache() -> None:
    build_info._GIT_SHA_CACHE.clear()
    yield
    build_info._GIT_SHA_CACHE.clear()


def test_baked_build_metadata_wins(monkeypatch: pytest.MonkeyPatch) -> None:
    """A docker build bakes GIT_SHA via --build-arg; it beats everything."""
    monkeypatch.setenv("GIT_SHA", "ef28bec")
    get_settings.cache_clear()
    try:
        assert resolve_git_sha() == "ef28bec"
    finally:
        get_settings.cache_clear()


def test_unbaked_deployments_fall_back_to_the_import_time_git_read(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Native systemd-run deployments get the working-tree short SHA."""
    monkeypatch.setenv("GIT_SHA", "unknown")
    monkeypatch.setattr(build_info, "_cached_git_sha", lambda: "dd3abfe")
    get_settings.cache_clear()
    try:
        assert resolve_git_sha() == "dd3abfe"
    finally:
        get_settings.cache_clear()


def test_without_git_the_sha_stays_unknown(monkeypatch: pytest.MonkeyPatch) -> None:
    """No build pipeline and no git — the honest absence, not a fabricated sha."""
    monkeypatch.setenv("GIT_SHA", "unknown")
    monkeypatch.setattr(build_info, "_cached_git_sha", lambda: None)
    get_settings.cache_clear()
    try:
        assert resolve_git_sha() == "unknown"
    finally:
        get_settings.cache_clear()


def test_git_read_runs_once_and_is_cached() -> None:
    """The subprocess executes at import, never in the request path."""
    build_info._GIT_SHA_CACHE["sha"] = "abc1234"
    try:
        assert build_info._cached_git_sha() == "abc1234"
        # A mutated cache is still served from the cache — no re-read.
        build_info._GIT_SHA_CACHE["sha"] = None
        assert build_info._cached_git_sha() is None
    finally:
        build_info._GIT_SHA_CACHE.clear()


@pytest.mark.asyncio
async def test_deep_health_payload_carries_the_resolved_sha(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The served payload is built from the resolver, not the raw setting.

    The deep handler is exercised with its probes stubbed so the assertion is
    about the git_sha field, not dependency reachability.
    """
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from app.features.health.adapters.inbound.http.router import deep_health_router

    async def _healthy(_app: FastAPI) -> DependencyHealth:
        return DependencyHealth.ok("stub", 0.0)

    monkeypatch.setenv("GIT_SHA", "unknown")
    monkeypatch.setattr(build_info, "_cached_git_sha", lambda: "dd3abfe")
    monkeypatch.setattr(
        "app.features.health.adapters.inbound.http.router.ALL_PROBES",
        tuple(_healthy for _ in range(7)),
    )
    get_settings.cache_clear()

    app = FastAPI()
    app.include_router(deep_health_router)

    try:
        with TestClient(app) as client:
            body = client.get("/health").json()
    finally:
        get_settings.cache_clear()

    data = body["data"]
    assert data["git_sha"] == "dd3abfe"
    assert data["status"] == "healthy"


def test_root_route_uses_the_shared_resolver() -> None:
    """F3 residual: GET / must not report git_sha "unknown" while /health knows.

    The root route closure reuses the exact resolver the deep-health payload
    uses (never the raw ``settings.GIT_SHA``), so provenance cannot disagree
    between the two endpoints on a native boot. Executed as an AST assertion:
    a full ``create_app()`` import costs minutes of heavy-model imports, and
    the suite's error-envelope convention already inspects ``main.py`` this way.
    """
    import ast
    from pathlib import Path

    source = Path("src/app/main.py").read_text(encoding="utf-8")
    tree = ast.parse(source)

    root_body: list[ast.stmt] | None = None
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "root":
            root_body = node.body
    assert root_body is not None, "root route handler not found in main.py"

    rendered = ast.unparse(ast.Module(body=root_body, type_ignores=[]))
    assert "resolve_git_sha()" in rendered
    assert "settings.GIT_SHA" not in rendered

    # And the resolver is the shared build_info seam, not a local copy.
    assert (
        "from .features.health.adapters.inbound.http.build_info import resolve_git_sha"
        in source
        or "from app.features.health.adapters.inbound.http.build_info import resolve_git_sha"
        in source
    )
