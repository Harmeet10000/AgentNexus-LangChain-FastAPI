"""Closure tests for the global exception-handler installation contract.

History this file exists to prevent: ``register_exception_handlers`` shipped for
MONTHS with three of the four handler branches unreachable — ``setdefault``
silently accepts an omitted registration, so deleting a row was invisible. The
docstring in :func:`app.middleware.global_exception_handler.exception_handler_bindings`
tells the story; these tests turn that prose into CI enforcement:

1. every binding row lands on ``app.exception_handlers`` — deleting a row fails;
2. the ``HTTPException`` key is Starlette's class object, which *replaces*
   FastAPI's pre-seeded default (registering the wrong class re-opens the bug;
   the key identity assertion is the tripwire);
3. MRO dispatch resolves each error family back to the global handler — the
   exact walk ``ExceptionMiddleware`` performs, asserted without an HTTP round
   trip, so a branch rendered unreachable fails here, not in the request path;
4. one request-path smoke test proves the dispatched handler answers with the
   project envelope, not a registry-shaped but functionally dead mapping.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import pytest
from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError, WebSocketRequestValidationError
from fastapi.testclient import TestClient
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.middleware import register_exception_handlers
from app.middleware.global_exception_handler import (
    exception_handler_bindings,
    global_exception_handler,
)
from app.utils import ServiceUnavailableException

pytestmark = pytest.mark.unit

if TYPE_CHECKING:
    from collections.abc import Callable


def _bare_registered_app() -> FastAPI:
    """A FastAPI app with FastAPI's pre-seeded defaults, plus our registrations."""
    app = FastAPI()
    register_exception_handlers(app)
    return app


def _dispatch(app: FastAPI, exc: Exception) -> Callable[..., object] | None:
    """Replicate ExceptionMiddleware's MRO walk over the handler registry."""
    return next(
        (
            app.exception_handlers[base]
            for base in type(exc).__mro__
            if base in app.exception_handlers
        ),
        None,
    )


def test_every_binding_row_is_installed_on_the_registry() -> None:
    app = _bare_registered_app()

    for exc_class, handler in exception_handler_bindings():
        assert app.exception_handlers.get(exc_class) is handler, (
            f"{exc_class}: the global handler is not installed for this key — "
            f"that branch is unreachable again"
        )


def test_starlette_key_replaces_fastapi_default_entry() -> None:
    bare = FastAPI()
    default_handler = bare.exception_handlers.get(StarletteHTTPException)
    assert default_handler is not None, "FastAPI no longer pre-seeds HTTPException"
    assert default_handler is not global_exception_handler

    app = _bare_registered_app()

    assert app.exception_handlers[StarletteHTTPException] is global_exception_handler


@pytest.mark.parametrize(
    ("build_exc", "label"),
    [
        (lambda: Exception("boom"), "catch-all"),
        (lambda: StarletteHTTPException(status_code=405), "starlette/http"),
        (
            lambda: ServiceUnavailableException("PostgreSQL startup failed"),
            "api-exception family",
        ),
        (lambda: RequestValidationError([], body=b"{}"), "validation"),
    ],
)
def test_mro_dispatch_resolves_every_family_to_the_global_handler(
    build_exc: Callable[[], Exception], label: str
) -> None:
    app = _bare_registered_app()
    exc = build_exc()

    resolved = _dispatch(app, exc)

    assert resolved is global_exception_handler, (
        f"{label}: MRO dispatch stops before reaching the global handler — "
        f"the branch is unreachable for this family"
    )


def test_websocket_validation_error_keeps_fastapi_default() -> None:
    """Deliberate exception: a websocket scope cannot carry an HTTP envelope."""
    app = _bare_registered_app()
    bare = FastAPI()

    assert (
        app.exception_handlers[WebSocketRequestValidationError]
        is bare.exception_handlers[WebSocketRequestValidationError]
    ), "the global handler was registered over FastAPI's websocket default"


def test_raised_api_exception_answers_with_the_project_envelope() -> None:
    """Request-path smoke: the dispatched handler actually renders the envelope."""
    app = _bare_registered_app()

    @app.get("/boom")
    async def boom() -> None:
        incident = "upstream gone"
        raise ServiceUnavailableException(incident)

    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.get("/boom")

    assert response.status_code == 503
    body = cast("dict[str, object]", response.json())
    assert body["success"] is False
