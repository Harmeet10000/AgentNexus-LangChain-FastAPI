"""C10: the unprovisioned ingestion graph must fail closed, not fall over.

C10 is a recorded **non-goal** plus the check that keeps it true. Nothing here provisions the shared
graph, and nothing should: ingestion runs in the queue worker process, which never executes the
application lifespan, so a per-process graph was never shared application state to begin with (D17).
The router that would consume this dependency is mounted in neither `v1` nor `v2`, so no
service-unavailable surface actually ships today. What is under test is the dependency's own
behaviour if it is ever reached.

**Why the app is hand-built rather than `create_app()`.** The real factory imports the full stack
(cognee, graphiti, the model layer) and its lifespan opens connections. The registration that matters
is reproducible exactly: the real app's handler registry holds precisely `HTTPException`,
`RequestValidationError`, `WebSocketRequestValidationError`, and `Exception` — the first three are
FastAPI's own defaults on any `FastAPI()`, and the fourth is the one line `main.py:110` adds. A bare
app plus that one registration is therefore not an approximation of the real registry; it is the same
registry. `test_the_handler_registry_matches_the_real_application` asserts that rather than trusting
it.

**Why no test here asserts the project's response envelope.** C10's Proof asks for the standard error
envelope, and it is currently **unreachable for every exception in the `APIException` family** — a
finding well outside this change's scope and recorded in `tasks.md` rather than fixed here.
`test_the_project_envelope_is_still_unreachable_tripwire` pins the cause so the day it is fixed shows
up as a failing tripwire with instructions, not as a silent divergence between the Proof and reality.
**Why this file has no `from __future__ import annotations`.** With it, ruff's type-checking rules
demand that `IngestionGraphDep` move into a `TYPE_CHECKING` block, because the alias appears only in
an annotation. Obeying that breaks the app at import time: FastAPI resolves every endpoint's type
hints to build its dependency graph, so an `Annotated[..., Depends(...)]` alias has to exist at
runtime. `src/app/features/ingestion/dependencies.py` omits the future import for the same reason, so
omitting it here matches the module under test rather than suppressing a rule.
"""

from collections.abc import Iterator
from typing import TYPE_CHECKING

import pytest
from fastapi import FastAPI
from fastapi.exception_handlers import http_exception_handler
from fastapi.testclient import TestClient
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.features.ingestion.dependencies import IngestionGraphDep, get_ingestion_graph
from app.middleware import global_exception_handler
from app.utils import APIException, ErrorCode, ServiceUnavailableException

if TYPE_CHECKING:
    # Type-only even without the future import: the sole use is a *local variable* annotation, and
    # PEP 526 local annotations are never evaluated at runtime. `Iterator` stays a runtime import
    # because a function's return annotation is evaluated at definition time.
    from typing import Any

pytestmark = pytest.mark.unit

_PATH = "/ingest-probe"
_CAPABILITY = "ingestion_graph"

#: A stand-in for a compiled graph. Its only job is to be *not* `None`, so that the positive control
#: below proves the failures elsewhere in this file come from absence rather than from a dependency
#: that always refuses.
_A_GRAPH = object()


def _build_app() -> FastAPI:
    """A FastAPI app whose exception-handler registry equals the real one. See the module docstring."""
    app = FastAPI()
    app.add_exception_handler(exc_class_or_status_code=Exception, handler=global_exception_handler)

    @app.get(_PATH)
    async def _probe(graph: IngestionGraphDep) -> dict[str, bool]:
        return {"graph_present": graph is not None}

    return app


@pytest.fixture
def app() -> FastAPI:
    return _build_app()


@pytest.fixture
def client(app: FastAPI) -> Iterator[TestClient]:
    # `raise_server_exceptions=False` so that an unhandled exception is observable as the 500 a
    # client would receive, instead of being re-raised into the test. Without it, the regression
    # this file exists for surfaces as an `AttributeError` traceback rather than as a status code,
    # and the assertion could not distinguish "failed closed" from "fell over".
    with TestClient(app, raise_server_exceptions=False) as c:
        yield c


# --------------------------------------------------------------------------------------
# The dependency fails closed
# --------------------------------------------------------------------------------------


def test_an_unprovisioned_graph_yields_service_unavailable_not_a_server_error(
    client: TestClient,
) -> None:
    """The mandatory C10 claim, in the form the defect actually took.

    The attribute is not set to `None` — it is never **set**. Starlette's state object raises on an
    unknown attribute, so the previous direct read raised before the `is None` test could run: the
    guard that looks like it produces a 503 produced an unhandled attribute error and a 500. The
    assertion is on 503 *and* explicitly not 500, because "some error happened" is exactly the
    conflation this task removes.
    """
    response = client.get(_PATH)

    assert response.status_code == 503, (
        f"expected a typed service-unavailable, got {response.status_code}; "
        "a 500 means the state read raised before the guard could run"
    )


def test_the_failure_names_the_capability_that_is_missing(client: TestClient) -> None:
    """Naming it structurally, not in prose.

    The capability travels as `data`, which `APIException` folds into its detail object, so a caller
    can branch on it without parsing a sentence. Asserted through the shape the *default* handler
    produces — see the envelope tripwire below for why that is the shape and not the project's.
    """
    body: dict[str, Any] = client.get(_PATH).json()

    detail = body["detail"]
    assert detail["error_code"] == ErrorCode.SERVICE_UNAVAILABLE
    assert detail["data"] == {"capability": _CAPABILITY}


def test_a_graph_explicitly_set_to_absent_also_fails_closed(app: FastAPI) -> None:
    """The one branch that already worked, pinned so the fix did not trade one case for the other.

    `getattr` with a default collapses "never provisioned" and "provisioned as absent" into the same
    outcome deliberately: from a caller's position they are the same condition, and keeping them
    apart is what produced two different status codes for one situation.
    """
    app.state.ingestion_graph = None

    with TestClient(app, raise_server_exceptions=False) as client:
        assert client.get(_PATH).status_code == 503


def test_a_provisioned_graph_is_handed_through(app: FastAPI) -> None:
    """The positive control.

    Without this, every test above would still pass against a dependency hard-coded to refuse, and
    the file would prove only that a constant is a constant.
    """
    app.state.ingestion_graph = _A_GRAPH

    with TestClient(app) as client:
        response = client.get(_PATH)

    assert response.status_code == 200
    assert response.json() == {"graph_present": True}


async def test_the_dependency_raises_the_typed_exception_rather_than_returning_none() -> None:
    """Asserted at the function level as well as through HTTP.

    The HTTP tests above route through a handler, so they would also pass if the dependency returned
    `None` and something downstream objected. This pins that the dependency itself refuses.
    """

    class _RequestWithNoState:
        app = FastAPI()

    with pytest.raises(ServiceUnavailableException) as caught:
        await get_ingestion_graph(_RequestWithNoState())  # ty: ignore[invalid-argument-type]

    assert caught.value.status_code == 503
    assert caught.value.data == {"capability": _CAPABILITY}


# --------------------------------------------------------------------------------------
# C10's other two Proofs, and the envelope gap
# --------------------------------------------------------------------------------------


def test_the_handler_registry_matches_the_real_application(app: FastAPI) -> None:
    """Justifies building the app by hand instead of calling the real factory.

    If FastAPI's defaults change, or `main.py` registers something new, this fails and the module
    docstring's claim stops being true — which is the point of asserting it rather than stating it.
    """
    registered = {
        getattr(key, "__name__", str(key)): handler.__name__
        for key, handler in app.exception_handlers.items()
    }

    assert registered == {
        "HTTPException": "http_exception_handler",
        "RequestValidationError": "request_validation_exception_handler",
        "WebSocketRequestValidationError": "websocket_request_validation_exception_handler",
        "Exception": "global_exception_handler",
    }


def test_the_project_envelope_is_still_unreachable_tripwire(app: FastAPI) -> None:
    """**A tripwire, not a specification.** It asserts a defect, and must fail when that is fixed.

    C10's second Proof asks for the standard error envelope. No exception in the `APIException`
    family can produce it: `add_exception_handler(Exception, ...)` installs a 500-only net on the
    outermost middleware, while every other class is resolved by walking the raised exception's MRO
    against the registry — and FastAPI pre-registers `HTTPException`, which `APIException` inherits
    from. The walk therefore stops three classes early, and the elaborate `APIException` branch in
    the project's own handler never executes for any request.

    Fixing that means registering the family explicitly, which changes the body of **every** error
    response in the application — auth, users, billing, documents — and is not a decision an
    ingestion change gets to make. So it is recorded and pinned here.

    **When this test fails, the gap has been closed.** Delete this test and restore C10's envelope
    Proof; do not adjust the assertion to keep it green.
    """
    selected = next(
        app.exception_handlers[cls]
        for cls in ServiceUnavailableException.__mro__
        if cls in app.exception_handlers
    )

    assert issubclass(ServiceUnavailableException, (APIException, StarletteHTTPException))
    assert selected is http_exception_handler, (
        "the project's own handler now receives APIException: the envelope gap is closed, so "
        "delete this tripwire and restore C10's standard-envelope Proof"
    )
