"""Opt-in access to the real task application, for tests about real registration.

The suite-wide conftest puts ``MagicMock()`` into ``sys.modules`` for the task
application module and for the task package, to keep unrelated unit tests from
paying for either import. A ``MagicMock`` has no ``__path__``, so under those
entries the declaring modules are not merely mocked — they are unimportable, and
the import machinery reports the package as not being a package at all. Anything
asserting that a name is registered, or that a module is listed for the worker,
therefore cannot be written at all while the entries are in place.

Rather than remove them and make every other unit test pay, this fixture lifts
them for the duration of one test module and puts the originals back afterwards.
It is deliberately not autouse: a module that wants the real application asks for
it, and the sibling module in this directory that wants the mocks keeps them.

Four details that are easy to get wrong here:

* the registry module has to be lifted too, even though nothing stubs it. If it
  was imported while the application module was a mock, it closed over a mock
  application, and a spy installed on the real one would never be consulted.
* the shared utility package has to be lifted as well, and only at its top level.
  Another unit test in this suite replaces it at import time with a two-attribute
  proxy whose logger is an async mock, and never puts the real one back. A module
  imported into that state binds a logger whose ``bind`` returns a coroutine, so
  the first diagnostic a refusal writes raises an attribute error instead — and
  half the declaring modules cannot be imported at all, because the proxy does
  not carry the exception classes they import. Only the top-level name is lifted
  so the already-imported submodules keep their identity: re-executing the
  package's re-export shim then rebinds the same classes rather than second
  copies of them, which matters because tests assert on exception identity.
* the task-name module is deliberately *not* lifted. It imports nothing from the
  application, so one copy serves everybody — which is what lets a test patch an
  entry in the declaring-module mapping and have the registry observe the patch.
* teardown drops every module imported through the lifted names before restoring
  the originals. Leaving the real modules loaded would leak the real application
  into whichever module runs next, which is the failure this fixture is avoiding
  in the other direction.
"""

import importlib
import sys
from types import SimpleNamespace

import pytest

_LIFTED_TREES = (
    "app.connections.celery",
    "tasks",
)

_LIFTED_NAMES = ("app.utils",)


def _is_lifted(module_name: str) -> bool:
    if module_name in _LIFTED_NAMES:
        return True
    return any(module_name == root or module_name.startswith(f"{root}.") for root in _LIFTED_TREES)


@pytest.fixture(scope="module")
def real_celery():
    """Yield the real task application and registry module, restoring the stubs after."""
    saved = {name: module for name, module in sys.modules.items() if _is_lifted(name)}
    for name in saved:
        del sys.modules[name]

    try:
        yield SimpleNamespace(
            app=importlib.import_module("app.connections.celery").celery_app,
            registry=importlib.import_module("app.connections.celery"),
        )
    finally:
        for name in [name for name in sys.modules if _is_lifted(name)]:
            del sys.modules[name]
        sys.modules.update(saved)
