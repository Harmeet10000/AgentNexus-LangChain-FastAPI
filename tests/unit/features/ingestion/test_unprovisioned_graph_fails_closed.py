"""C10: the unprovisioned ingestion graph must fail closed, not fall over.

C10 is a recorded **non-goal** plus the check that keeps it true. Nothing here provisions the shared
graph, and nothing should: ingestion runs in the queue worker process, which never executes the
application lifespan, so a per-process graph was never shared application state to begin with (D17).
The router that would consume this dependency is mounted in neither `v1` nor `v2`, so no
service-unavailable surface actually ships today. What is under test is the dependency's own
behaviour if it is ever reached.

**Why the app is hand-built rather than `create_app()`.** The real factory imports the full stack
(cognee, graphiti, the model layer) and its lifespan opens connections. The registration that matters
is reproducible exactly, because it is a single function: `register_exception_handlers`, the same one
`create_app()` calls. A bare app plus that call is therefore not an approximation of the real registry;
it is the same registry. `test_the_handler_registry_matches_the_real_application` asserts that rather
than trusting it. `RequestStateLoggingMiddleware` is added for the same reason — it sets the
ContextVars the handler and `http_error` read, and without it every enveloped response here would be
a bodiless 500.

**The envelope is now asserted here, and used to be pinned as unreachable.** C10's Proof asks for the
standard error envelope. When this file was written the envelope was unreachable for the entire
`APIException` family, and `test_the_project_envelope_is_still_unreachable_tripwire` recorded the
cause so the day it was fixed would show up as a failing tripwire rather than as a silent divergence
between the Proof and reality. That day came: the tripwire is now
`test_the_project_envelope_is_reachable_for_the_api_exception_family`, asserting the same mechanism
with the opposite expectation, and C10's envelope Proof is live in
`test_the_failure_names_the_capability_that_is_missing`.
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
from fastapi.testclient import TestClient
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.features.ingestion.dependencies import IngestionGraphDep, get_ingestion_graph
from app.middleware import (
    RequestStateLoggingMiddleware,
    global_exception_handler,
    register_exception_handlers,
)
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
    app.add_middleware(RequestStateLoggingMiddleware)
    register_exception_handlers(app)

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
    """Naming it structurally, not in prose — and this is C10's standard-envelope Proof.

    The capability travels as `data`, which `APIException` folds into `detail["data"]`, so a caller
    can branch on it without parsing a sentence. **The path to it changed.** Under FastAPI's default
    handler the body was `{"detail": {"message", "error_code", "data"}}`; now that the family reaches
    the project's own handler, the same value is at `error.data` inside the standard envelope, and
    `error_code` has become `error.code`. Both the envelope and the capability are asserted here, so
    a regression in either one fails a test whose name says what was lost.
    """
    response = client.get(_PATH)
    body: dict[str, Any] = response.json()

    assert set(body) == {"success", "statusCode", "request", "message", "data", "error"}
    assert body["success"] is False
    assert body["statusCode"] == 503
    assert "detail" not in body, "FastAPI's default handler is back; the envelope is unreachable"

    error = body["error"]
    assert error["code"] == ErrorCode.SERVICE_UNAVAILABLE
    assert error["data"] == {"capability": _CAPABILITY}


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

    Compared against a reference registry produced by the *same* function `main.py` calls, rather
    than against a hand-copied list of class names. Two reasons. Both `HTTPException` classes share
    a `__name__`, so the name-keyed comparison this test used to make could not tell the working
    registration from the broken one. And a literal list has to be edited every time the registration
    surface changes, which is how the docstring's claim would quietly stop being true — the failure
    mode this test exists to prevent. The by-key-object literal lives once, in
    `tests/unit/middleware/test_error_envelope_is_universal.py`.
    """
    reference = FastAPI()
    register_exception_handlers(reference)

    assert app.exception_handlers == reference.exception_handlers
    # Guard against the reference itself being empty or FastAPI's untouched defaults: if
    # `register_exception_handlers` ever became a no-op, the equality above would still hold.
    assert app.exception_handlers[APIException] is global_exception_handler


def test_the_project_envelope_is_reachable_for_the_api_exception_family(app: FastAPI) -> None:
    """The mechanism that makes the registration load-bearing. **Was a tripwire asserting the defect.**

    C10's second Proof asks for the standard error envelope. For most of this application's life no
    exception in the `APIException` family could produce it, and the reason is worth keeping written
    down, because nothing about the code makes it visible:
    `add_exception_handler(Exception, ...)` installs a 500-only net on the *outermost* middleware,
    while every other class is resolved by walking the raised exception's MRO against the registry on
    the *innermost* one — and FastAPI pre-registers `HTTPException`, which `APIException` inherits
    from. The walk therefore stopped three classes early, and the elaborate `APIException` branch in
    the project's own handler never executed for any request. Because FastAPI installs its entries
    with `setdefault`, omitting the registration produced no error and no warning.

    `register_exception_handlers` closes that by registering the family explicitly, which changed the
    body of **every** error response in the application — auth, users, billing, documents. That was
    too large a decision for an ingestion change to make, so this test spent its first life as
    `test_the_project_envelope_is_still_unreachable_tripwire`, asserting that the selected handler
    *was* FastAPI's, and instructing whoever closed the gap to invert it. This is that inversion.

    Asserted at the level of handler *selection*, not response body: the body is covered elsewhere,
    and what is uniquely worth pinning here is the MRO walk itself — the step that was silently
    wrong and that no response assertion explains.
    """
    selected = next(
        app.exception_handlers[cls]
        for cls in ServiceUnavailableException.__mro__
        if cls in app.exception_handlers
    )

    assert issubclass(ServiceUnavailableException, (APIException, StarletteHTTPException))
    assert selected is global_exception_handler, (
        "the MRO walk no longer reaches the project's handler for the APIException family: the "
        "envelope gap has reopened and C10's standard-envelope Proof is void. Check that "
        "register_exception_handlers still registers APIException and starlette's HTTPException"
    )
