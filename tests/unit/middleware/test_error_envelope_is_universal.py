"""The project's error envelope is the API's actual contract, on every error path.

`src/app/utils/response_type.py` defines one response shape — ``{success, statusCode, request,
message, data, error}`` — and `src/app/middleware/global_exception_handler.py` has four branches that
build it. Until `register_exception_handlers` existed, three of those four branches were unreachable
and the fourth could not emit a body, so the shape was documentation rather than contract. These tests
exist so that can never quietly become true again.

**Why every test here goes through a `TestClient` and never calls the handler directly.** The defect
these tests close was invisible to any test that called `global_exception_handler(request, exc)` — the
function was always correct. What was wrong was whether Starlette would ever *call* it, which is
decided by the exception-handler registry and an MRO walk inside `ExceptionMiddleware`. A direct
function call skips exactly the machinery that was broken. That is why the defect survived, and why a
unit test of the handler function would be worse than no test: it would report green over a dead
branch.

**Why the app is assembled here rather than imported from `app.main`.** Importing the real factory
costs ~6s (cognee, graphiti, the model layer) and executes `create_app()` at module scope. The two
things that decide the behaviour under test are reproduced exactly instead:

* `register_exception_handlers` — the same function `main.py` calls, so the registry is production's by
  construction, not by imitation. `test_the_handler_registry_is_the_one_the_application_ships` pins its
  contents against the registry a real `create_app()` was observed to hold.
* `RequestStateLoggingMiddleware` — required, not decoration. It sets the `request_state` and
  `execution_path` ContextVars the handler and `http_error` read. Omit it and every test here fails
  with a bodiless 500, which is precisely the bug documented on
  `test_an_unexpected_exception_arrives_with_a_body`.

`test_the_application_factory_registers_through_register_exception_handlers` covers the remaining gap
— that `main.py` really routes through this function — by parsing its source, so no import cost is
paid for it.
"""

from __future__ import annotations

import ast
import importlib.util
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
import starlette.exceptions
from fastapi import FastAPI, HTTPException
from fastapi.exceptions import RequestValidationError, WebSocketRequestValidationError
from fastapi.testclient import TestClient
from pydantic import BaseModel

from app.middleware import RequestStateLoggingMiddleware, register_exception_handlers
from app.utils import ErrorCode, UnauthorizedException

if TYPE_CHECKING:
    from collections.abc import Iterator
    from typing import Any

pytestmark = pytest.mark.unit

#: Every key the envelope promises, in the serialized (camelCase alias) form a client sees.
_ENVELOPE_KEYS = {"success", "statusCode", "request", "message", "data", "error"}

#: Raised by the branch-4 route. A module-level constant rather than an inline literal so the
#: message is one thing, named once, and ruff's EM101 stays satisfied without a suppression.
_UNEXPECTED = "nobody planned for this"


class _Payload(BaseModel):
    quantity: int


def _build_app() -> FastAPI:
    """A FastAPI app whose error-handling stack equals the real one. See the module docstring."""
    app = FastAPI()
    app.add_middleware(RequestStateLoggingMiddleware)
    register_exception_handlers(app)

    @app.get("/branch-1-api-exception")
    async def _api_exception() -> None:
        raise UnauthorizedException(detail="token rejected")

    @app.post("/branch-2-validation")
    async def _validation(payload: _Payload) -> dict[str, int]:
        return {"quantity": payload.quantity}

    @app.get("/branch-3-fastapi-http-exception")
    async def _fastapi_http_exception() -> None:
        raise HTTPException(status_code=409, detail="already claimed")

    @app.get("/branch-3-starlette-http-exception")
    async def _starlette_http_exception() -> None:
        # Not the same class as the import above: `fastapi.HTTPException` subclasses this one, and
        # FastAPI keys its default registry entry on *this* class. Raised directly here so the
        # registration is proven against the key object it actually has to displace.
        raise starlette.exceptions.HTTPException(status_code=409, detail="already claimed")

    @app.get("/branch-4-unexpected")
    async def _unexpected() -> None:
        raise RuntimeError(_UNEXPECTED)

    @app.get("/no-body-status")
    async def _no_body_status() -> None:
        raise HTTPException(status_code=304)

    @app.get("/success")
    async def _success() -> dict[str, str]:
        return {"plain": "body"}

    return app


@pytest.fixture
def client() -> Iterator[TestClient]:
    # `raise_server_exceptions=False` so an unhandled exception is observable as the response a
    # client would receive. With it left on, branch 4 re-raises into the test (Starlette's
    # `ServerErrorMiddleware` always re-raises after responding) and the assertion can never see
    # the body — which is the only thing branch 4 was getting wrong.
    with TestClient(_build_app(), raise_server_exceptions=False) as test_client:
        yield test_client


def _assert_is_envelope(body: dict[str, Any], *, status_code: int, error_code: str) -> None:
    assert set(body) == _ENVELOPE_KEYS, f"not the project envelope: {sorted(body)}"
    assert body["success"] is False
    assert body["statusCode"] == status_code
    assert body["error"] is not None
    assert body["error"]["code"] == error_code
    # FastAPI's default handlers return `{"detail": ...}`. Its absence is the whole point of the
    # change and is asserted positively, not left to the key-set comparison above, so a future
    # relaxation of that comparison cannot let `detail` back in unnoticed.
    assert "detail" not in body


# --------------------------------------------------------------------------------------
# The four branches of global_exception_handler, each driven through HTTP
# --------------------------------------------------------------------------------------


def test_an_api_exception_arrives_as_the_project_envelope(client: TestClient) -> None:
    """Branch 1 — the family every feature raises, and the branch that was dead the longest.

    `APIException` inherits from `HTTPException`, which FastAPI pre-registers, so before
    `register_exception_handlers` the MRO walk stopped at FastAPI's handler and this body was
    `{"detail": {"message": ..., "error_code": ...}}`.
    """
    response = client.get("/branch-1-api-exception")

    assert response.status_code == 401
    _assert_is_envelope(response.json(), status_code=401, error_code=ErrorCode.UNAUTHORIZED)
    assert response.json()["message"] == "token rejected"


def test_an_api_exceptions_own_headers_survive_the_envelope(client: TestClient) -> None:
    """`UnauthorizedException` sets `WWW-Authenticate`, and RFC 9110 requires it on a 401.

    The handler forwards `exc.headers` when it builds the response. Routing the family to a
    different handler is exactly the kind of change that drops such a detail, so it is pinned
    rather than assumed.
    """
    assert client.get("/branch-1-api-exception").headers["www-authenticate"] == "Bearer"


def test_a_real_validation_failure_arrives_as_the_project_envelope(client: TestClient) -> None:
    """Branch 2 — driven by an actual bad request body, not a hand-built exception.

    FastAPI raises `RequestValidationError` from inside its own request handler, and pre-registers a
    handler for it by the same `setdefault` that hid branch 1. Posting a non-integer is what proves
    the registration displaced it on the real code path.
    """
    response = client.post("/branch-2-validation", json={"quantity": "not-a-number"})

    assert response.status_code == 422
    body = response.json()
    _assert_is_envelope(body, status_code=422, error_code=ErrorCode.VALIDATION_ERROR)
    assert body["error"]["data"]["errors"][0]["field"] == "body → quantity"


def test_a_plain_fastapi_http_exception_arrives_as_the_project_envelope(client: TestClient) -> None:
    """Branch 3, raised as `fastapi.HTTPException` — what application code normally imports."""
    response = client.get("/branch-3-fastapi-http-exception")

    assert response.status_code == 409
    _assert_is_envelope(response.json(), status_code=409, error_code="HTTP_409")
    assert response.json()["message"] == "already claimed"


def test_a_plain_starlette_http_exception_arrives_as_the_project_envelope(
    client: TestClient,
) -> None:
    """Branch 3 again, via the *other* `HTTPException` class — the trap in this change.

    `fastapi.exceptions.HTTPException` and `starlette.exceptions.HTTPException` are distinct class
    objects and both sit in the MRO. FastAPI keys its default entry on Starlette's, so registering
    Starlette's replaces that entry and covers both; registering FastAPI's subclass instead would
    add a more-derived key, leave FastAPI's entry in place, and let anything raising Starlette's
    class straight through to `{"detail": ...}`. Starlette's own router raises that class, so this
    is a live path, not a hypothetical. This test is the difference between the two choices.
    """
    response = client.get("/branch-3-starlette-http-exception")

    assert response.status_code == 409
    _assert_is_envelope(response.json(), status_code=409, error_code="HTTP_409")


def test_an_unexpected_exception_arrives_with_a_body(client: TestClient) -> None:
    """Branch 4 — registered all along, and still unable to answer.

    `add_exception_handler(Exception, ...)` puts the handler on `ServerErrorMiddleware`, outside
    `RequestStateLoggingMiddleware` — whose `finally` block resets the `request_state` and
    `execution_path` ContextVars as the exception passes back up. The handler then read an unset
    ContextVar, raised `LookupError` inside `ServerErrorMiddleware`'s own `except`, and the client
    got a 500 with **no headers and a zero-byte body**. So this asserts the body's content, not just
    the status: `assert status_code == 500` passed throughout the defect.
    """
    response = client.get("/branch-4-unexpected")

    assert response.status_code == 500
    assert response.content, "bodiless 500: the handler raised on an unset ContextVar"
    _assert_is_envelope(
        response.json(), status_code=500, error_code=ErrorCode.INTERNAL_SERVER_ERROR
    )


# --------------------------------------------------------------------------------------
# Boundaries: what the registration must *not* change
# --------------------------------------------------------------------------------------


def test_a_status_code_that_forbids_a_body_gets_no_body(client: TestClient) -> None:
    """The envelope stops where HTTP says a body may not go.

    FastAPI's `http_exception_handler` skipped the body for 1xx/204/304. Displacing it moved that
    obligation into `_json_error_response`; without the guard a 304 would ship a JSON envelope and a
    `Content-Length`, which a real ASGI server rejects as a protocol error even though `TestClient`
    tolerates it.
    """
    response = client.get("/no-body-status")

    assert response.status_code == 304
    assert response.content == b""


def test_a_successful_response_is_untouched(client: TestClient) -> None:
    """The positive control.

    Every other test here asserts a body shape on an error. Without this, a registration that
    somehow enveloped *successful* responses too would look like a complete success.
    """
    response = client.get("/success")

    assert response.status_code == 200
    assert response.json() == {"plain": "body"}


def test_websocket_validation_errors_are_left_with_fastapi(client: TestClient) -> None:
    """A deliberate exclusion, asserted so it reads as a decision rather than an oversight.

    `WebSocketRequestValidationError` is a sibling of `RequestValidationError`, not a subclass, so
    registering the latter does not capture it. It must stay with FastAPI's handler: that handler
    closes the socket with a code, and a websocket scope has no way to send an HTTP JSON envelope.
    """
    handler = client.app.exception_handlers[WebSocketRequestValidationError]

    assert handler.__name__ == "websocket_request_validation_exception_handler"
    assert not issubclass(WebSocketRequestValidationError, RequestValidationError)


# --------------------------------------------------------------------------------------
# The wiring itself
# --------------------------------------------------------------------------------------


def test_the_handler_registry_is_the_one_the_application_ships(client: TestClient) -> None:
    """Pins the registry by *key object*, which is where this change can silently go wrong.

    Keys are compared as ``module.qualname`` on purpose. Both `HTTPException` classes have the same
    ``__name__``, so a name-only comparison — the obvious way to write this — passes whichever one is
    registered, and cannot tell the working registration from the broken one.

    The expected mapping was read off a real `create_app()` instance (see the module docstring for
    why that call is not made here).
    """
    registered = {
        f"{key.__module__}.{key.__qualname__}": handler.__name__
        for key, handler in client.app.exception_handlers.items()
    }

    assert registered == {
        "builtins.Exception": "global_exception_handler",
        "starlette.exceptions.HTTPException": "global_exception_handler",
        "fastapi.exceptions.RequestValidationError": "global_exception_handler",
        "app.utils.exceptions.APIException": "global_exception_handler",
        "fastapi.exceptions.WebSocketRequestValidationError": (
            "websocket_request_validation_exception_handler"
        ),
    }


def test_the_application_factory_registers_through_register_exception_handlers() -> None:
    """The one thing a hand-built app cannot show: that `main.py` uses this wiring.

    Read from source with `ast` rather than by importing `app.main`, which would cost ~6s and run
    `create_app()` as an import side effect. `find_spec` locates the module without executing it.

    A direct `add_exception_handler` call in the factory is treated as a failure, not a warning: that
    is the shape the defect had, and a partial registration alongside this call is worse than either
    — it would look deliberate.
    """
    spec = importlib.util.find_spec("app.main")
    assert spec is not None
    assert spec.origin is not None
    tree = ast.parse(Path(spec.origin).read_text(encoding="utf-8"))

    factory = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "create_app"
    )
    called = {
        node.func.attr if isinstance(node.func, ast.Attribute) else getattr(node.func, "id", "")
        for node in ast.walk(factory)
        if isinstance(node, ast.Call)
    }

    assert "register_exception_handlers" in called, (
        "create_app() no longer registers exception handlers through "
        "register_exception_handlers; the error envelope is unreachable again"
    )
    assert "add_exception_handler" not in called, (
        "create_app() registers an exception handler directly; every registration belongs in "
        "register_exception_handlers, where the MRO mechanism is documented"
    )
