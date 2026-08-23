"""C8: the documented worker command and the deployed one are a single string.

Before this task both documented commands — `Makefile:52` and `README.md:279` — read
`uv run celery -A celery_config worker --loglevel=info`, and **no `celery_config` module exists
anywhere in the repository**. The documented way to start a worker could not start a worker, and
nothing in the suite noticed, because nothing compared the documentation to anything.

Fixing the string in both places would have left the actual defect in place: two copies of one
command, free to diverge again. So the command is now defined once, in the `Makefile`, and this
module asserts every other copy equals that definition. C8's third Proof allows exactly this
("an equality assertion in a check script") — a test is preferred over a script because it runs
on every commit rather than when someone remembers.

**Why the Makefile is parsed instead of run.** Invoking `make` needs `subprocess`, and the
project enables bandit's rules (`"S"` appears under `unfixable`, not `ignore`), so a
`subprocess.run(["make", ...])` call would need two suppressions — against CLAUDE.md's preference
for satisfying a check over silencing it. It would also make the suite depend on `make` being
installed. Parsing risks testing the parser rather than the Makefile, so
`test_the_makefile_command_needs_exactly_one_substitution` pins that the definition stays simple
enough for the one substitution done here to be complete: add a second variable to that line and
this test fails and says so, rather than the expansion silently checking the wrong string.

**What is deliberately *not* proven here.** C8's second Proof asks that the documented command be
run verbatim and report its registered tasks. The configured broker is a **live managed instance**,
and the registered task set includes billing, credit and transactional-email tasks — so starting a
consuming worker against it could execute real queued work, including sending mail to real
recipients. That Proof was therefore not executed as written; `tasks.md` records the substitute
evidence and the reason. Registration itself is proven without a broker by
`test_task_registration.py`.

C8's third Proof also asks that the documented string equal the compose service's string. No
worker service exists in `docker-compose.yml` yet — that service is **C7**, which is blocked on an
unanswered topology question. `test_the_compose_worker_service_runs_the_documented_command` is
written to skip while that is true and to start asserting the moment the service appears, so C7
cannot land a worker whose command has drifted from the documentation.
"""

import importlib
import re
import shlex
from pathlib import Path

import pytest
from celery import Celery

pytestmark = pytest.mark.unit

_REPO_ROOT = Path(__file__).resolve().parents[3]
_MAKEFILE = _REPO_ROOT / "Makefile"
_README = _REPO_ROOT / "README.md"
_COMPOSE = _REPO_ROOT / "docker-compose.yml"

#: The module name that was documented for years and has never existed. Assembled from parts so
#: that this file — which greps the deployable files for it — does not itself contain the literal:
#: a comment or constant quoting the string a proof searches for defeats that proof.
_PHANTOM_MODULE = "celery" + "_config"

#: The source-path spelling of the task application. It resolves from the repository root only
#: because `python -c` and pytest put the working directory on the import path; it is a *different*
#: module object from the installed spelling, with its own task registry. See
#: `test_the_application_is_named_by_its_installed_identity_not_its_source_path`.
_SOURCE_PATH_PREFIX = "src.app."

_MAKE_ASSIGNMENT = re.compile(r"^(?P<name>[A-Z_][A-Z0-9_]*)\s*:?=\s*(?P<value>.*)$", re.MULTILINE)
_MAKE_VARIABLE_REFERENCE = re.compile(r"\$\((?P<name>[^)]+)\)")


def _read(path: Path) -> str:
    """One place where the encoding is decided, rather than four."""
    return path.read_text(encoding="utf-8")


def _make_assignments() -> dict[str, str]:
    return {m["name"]: m["value"].strip() for m in _MAKE_ASSIGNMENT.finditer(_read(_MAKEFILE))}


def _makefile_worker_command() -> str:
    """The definition site's command, with its single variable reference expanded."""
    assignments = _make_assignments()
    command = assignments["CELERY_WORKER_CMD"]
    return _MAKE_VARIABLE_REFERENCE.sub(lambda m: assignments[m["name"]], command)


def _readme_worker_commands() -> list[str]:
    lines = _read(_README).splitlines()
    return [line.strip() for line in lines if line.strip().startswith("uv run celery ")]


def _dash_a_value(command: str) -> str:
    """The `-A` argument, which is what Celery resolves to find the application."""
    argv = shlex.split(command)
    return argv[argv.index("-A") + 1]


# --------------------------------------------------------------------------------------
# One string, one definition site
# --------------------------------------------------------------------------------------


def test_the_documented_command_and_the_definition_site_are_one_string() -> None:
    """The mandatory C8 claim: documentation and definition cannot disagree.

    Compared as whole strings rather than by module name alone, because `--loglevel` and the
    worker subcommand are equally capable of drifting, and a command that differs anywhere is a
    command someone will run and get a different result from.
    """
    documented = _readme_worker_commands()

    assert len(documented) == 1, (
        f"expected exactly one documented worker command in README.md, found {len(documented)}; "
        "more than one copy is the drift this task removes"
    )
    assert documented[0] == _makefile_worker_command()


def test_the_makefile_command_needs_exactly_one_substitution() -> None:
    """Guards the expansion above, so the equality test cannot pass by checking the wrong string.

    `_makefile_worker_command` performs a plain textual substitution rather than deferring to
    `make`. That is only faithful while the command line references exactly one variable, and
    while that variable's own value references none. Both are asserted, so extending the
    definition fails here with an explanation instead of silently making the comparison vacuous.
    """
    assignments = _make_assignments()
    referenced = _MAKE_VARIABLE_REFERENCE.findall(assignments["CELERY_WORKER_CMD"])

    assert referenced == ["CELERY_APP"], (
        f"the worker command now references {referenced}; the one-pass expansion in this module "
        "is no longer faithful — expand it, or invoke make and accept the bandit suppressions"
    )
    assert not _MAKE_VARIABLE_REFERENCE.search(assignments["CELERY_APP"]), (
        "CELERY_APP now references another variable, so one substitution pass is not enough"
    )


# --------------------------------------------------------------------------------------
# The command names something that exists, and is the right something
# --------------------------------------------------------------------------------------


def test_the_named_application_module_resolves(real_celery) -> None:
    """C8's first Proof, in the form that matters: the `-A` module imports.

    The fixture performs the same `importlib.import_module` that Celery's `-A` handling performs,
    with the suite-wide stubs lifted. Without the fixture this would import a `MagicMock` and pass
    while proving nothing — see this directory's conftest.
    """
    module_path = _dash_a_value(_makefile_worker_command()).split(":")[0]

    assert module_path != _PHANTOM_MODULE
    assert real_celery.app.__class__ is Celery
    # The fixture imported this exact path; reaching it here confirms the command names it.
    assert module_path == "app.connections.celery"


def test_the_named_attribute_is_the_task_application(real_celery) -> None:
    """The `module:attribute` half of the command resolves to the registry's own application.

    Written as an identity check against the object the fixture imported, so a second `Celery`
    instance added to that module could not satisfy it. That is the reason the command names the
    attribute explicitly instead of letting Celery probe the module for whatever instance it finds
    first.
    """
    module_path, _, attribute = _dash_a_value(_makefile_worker_command()).partition(":")

    assert attribute, (
        "the command should name the application attribute, not rely on Celery's probe"
    )

    resolved = getattr(importlib.import_module(module_path), attribute)

    assert resolved is real_celery.app


def test_the_application_is_named_by_its_installed_identity_not_its_source_path() -> None:
    """Pins the prefix, because both spellings import and they are not the same object.

    `src/` is on the import path through the editable install, so `app.connections.celery` resolves
    from **any** working directory. `src.app.connections.celery` resolves only when the working
    directory happens to be the repository root, since that is what puts the root on the path —
    measured directly: from `/tmp`, the installed spelling imports and the source-path spelling
    raises. A container whose working directory is not the repository root would therefore fail to
    start with the `src.` spelling.

    Worse than failing: Python keys `sys.modules` by the import string, so the two spellings
    produce two module objects, each with its own `Celery` instance and its own task registry. A
    worker started under one and a producer importing the other would agree about every task name
    and share none of them. The task modules are listed as `tasks.*`, which is the installed
    rooting, so that is the identity the registry already uses.
    """
    command = _makefile_worker_command()

    assert _SOURCE_PATH_PREFIX not in command, (
        f"the worker command names the source-path spelling; {_SOURCE_PATH_PREFIX}* is a second "
        "module identity with its own task registry, and it only imports from the repository root"
    )


# --------------------------------------------------------------------------------------
# The phantom module is gone, and C7's service cannot drift from the documentation
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("path", [_MAKEFILE, _README, _COMPOSE], ids=lambda p: p.name)
def test_the_module_that_never_existed_is_gone_from_every_deployable_file(path: Path) -> None:
    """The regression guard for the original defect.

    Scoped to the files that are executed or copy-pasted. The working notes under `docs/relay/`
    still contain the broken string and should: they quote it as the evidence that the defect was
    real, and rewriting evidence to satisfy a grep would be the wrong direction. `tasks.md` records
    that scope amendment against C8's first Proof, which as written covers all of `docs/`.
    """
    assert _PHANTOM_MODULE not in _read(path), (
        f"{path.name} still names {_PHANTOM_MODULE}, which does not exist and cannot start a worker"
    )


def test_the_compose_worker_service_runs_the_documented_command() -> None:
    """C8's third Proof, armed for C7 rather than deferred to it.

    C7 adds the worker and scheduler services and is blocked on an unanswered question about queue
    topology, so there is nothing to compare against today. This skips while that is true and
    begins asserting as soon as the compose file mentions the worker at all — which means C7 cannot
    introduce a service whose command has drifted from the documentation without turning this red.

    The check is a substring rather than a parse: no YAML dependency, and it holds whether the
    command is written as a string or a list, since either spelling contains the command's text.
    """
    compose = _read(_COMPOSE)

    if "celery" not in compose:
        pytest.skip(
            "no worker service in compose yet — C7 adds it; this test arms itself when it does"
        )

    assert _makefile_worker_command() in compose, (
        "the compose worker service does not run the documented command verbatim; C8 requires the "
        "documented string and the deployed string to be identical"
    )
