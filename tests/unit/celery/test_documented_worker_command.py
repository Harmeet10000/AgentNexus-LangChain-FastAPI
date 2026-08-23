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

C8's third Proof also asks that the documented string equal the compose service's string. C7 has
since added the worker and scheduler services, so `test_the_compose_worker_service_runs_the_documented
_command` no longer skips: the base command must appear in the compose file verbatim, and it does,
as the shared prefix of both worker services' commands.

**What C7 changed here, and why.** C7 introduced a second worker command (a dedicated ingestion
queue) and a scheduler command, so `uv run celery ` now appears three times in the README rather
than once. `test_the_documented_command_and_the_definition_site_are_one_string`'s
`len(documented) == 1` could not survive that, and weakening it to `>= 1` would have let an
undocumented or drifted command through. It is generalised instead, per the two rules that assertion
was protecting: the set of documented commands must equal the set the `Makefile` defines — so the
count is still pinned, to the definition site's own count rather than to the literal 1 — and every
celery command anywhere, documented or defined or deployed, must name the application by the same
`-A` value. That is strictly more than the original asserted: three exact strings instead of one, in
three files instead of two.
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

#: Every command the deployment runs, by its `Makefile` variable name. Listed rather than discovered
#: by pattern, so that renaming one of them fails here — loudly — instead of quietly shrinking what
#: this module compares.
_CELERY_COMMAND_VARIABLES = (
    "CELERY_DEFAULT_WORKER_CMD",
    "CELERY_INGESTION_WORKER_CMD",
    "CELERY_BEAT_CMD",
)

#: A bound on the fixed-point expansion below. The derived commands reference a variable that itself
#: references one, so a single pass is not enough for them — but an unbounded loop over a Makefile
#: with a self-referential assignment would hang the suite instead of failing it.
_MAX_EXPANSION_PASSES = 8

_MAKE_ASSIGNMENT = re.compile(r"^(?P<name>[A-Z_][A-Z0-9_]*)\s*:?=\s*(?P<value>.*)$", re.MULTILINE)
_MAKE_VARIABLE_REFERENCE = re.compile(r"\$\((?P<name>[^)]+)\)")

#: Compose is matched rather than parsed, deliberately: no YAML dependency, and the commands are
#: written as plain strings precisely so a text comparison is sufficient.
_COMPOSE_CELERY_COMMAND = re.compile(
    r"^\s*command:\s*(?P<command>uv run celery .*?)\s*$", re.MULTILINE
)


def _read(path: Path) -> str:
    """One place where the encoding is decided, rather than four."""
    return path.read_text(encoding="utf-8")


def _make_assignments() -> dict[str, str]:
    return {m["name"]: m["value"].strip() for m in _MAKE_ASSIGNMENT.finditer(_read(_MAKEFILE))}


def _makefile_worker_command() -> str:
    """The base worker command, with its single variable reference expanded.

    Deliberately still a single pass, so that
    `test_the_makefile_command_needs_exactly_one_substitution` keeps meaning what it says about this
    one definition. The derived commands use the fixed-point expander below.
    """
    assignments = _make_assignments()
    command = assignments["CELERY_WORKER_CMD"]
    return _MAKE_VARIABLE_REFERENCE.sub(lambda m: assignments[m["name"]], command)


def _expand(value: str, assignments: dict[str, str]) -> str:
    """Substitute until nothing changes, so a command built from a command resolves."""
    for _ in range(_MAX_EXPANSION_PASSES):
        expanded = _MAKE_VARIABLE_REFERENCE.sub(lambda m: assignments[m["name"]], value)
        if expanded == value:
            return value
        value = expanded
    return value


def _makefile_celery_commands() -> dict[str, str]:
    assignments = _make_assignments()
    return {name: _expand(assignments[name], assignments) for name in _CELERY_COMMAND_VARIABLES}


def _is_worker_command(command: str) -> bool:
    """Distinguish a worker command from the scheduler's, by the subcommand it names."""
    return "worker" in shlex.split(command)


def _readme_celery_commands() -> list[str]:
    lines = _read(_README).splitlines()
    return [line.strip() for line in lines if line.strip().startswith("uv run celery ")]


def _compose_celery_commands() -> list[str]:
    return [m["command"] for m in _COMPOSE_CELERY_COMMAND.finditer(_read(_COMPOSE))]


def _dash_a_value(command: str) -> str:
    """The `-A` argument, which is what Celery resolves to find the application."""
    argv = shlex.split(command)
    return argv[argv.index("-A") + 1]


# --------------------------------------------------------------------------------------
# One string, one definition site
# --------------------------------------------------------------------------------------


def test_the_documented_command_and_the_definition_site_are_one_string() -> None:
    """The mandatory C8 claim, generalised by C7: documentation and definition cannot disagree.

    Compared as whole strings rather than by module name alone, because `--loglevel`, the queue
    selection and the worker subcommand are equally capable of drifting, and a command that differs
    anywhere is a command someone will run and get a different result from.

    Set equality in both directions is what replaced C8's `len(documented) == 1`. It still pins the
    count — to however many commands the definition site defines — and it additionally catches a
    documented command that no longer exists, which a count alone would not.
    """
    documented = _readme_celery_commands()
    defined = _makefile_celery_commands()

    assert len(documented) == len(defined), (
        f"README.md documents {len(documented)} celery commands and the Makefile defines "
        f"{len(defined)}; every command the deployment runs is documented, and nothing else is"
    )
    assert set(documented) == set(defined.values())


def test_every_celery_command_anywhere_names_the_same_application() -> None:
    """The one property that must hold of a celery command no matter what it is for.

    A worker, a scheduler, and any future subcommand each have their own flags, so whole-string
    equality cannot be asked of all of them together. What can: they must all resolve to the same
    task application. Two `-A` values in one deployment means two task registries, and a producer
    and consumer that agree about every task name while sharing none of them.
    """
    expected = _dash_a_value(_makefile_worker_command())
    everywhere = (
        _readme_celery_commands()
        + list(_makefile_celery_commands().values())
        + _compose_celery_commands()
    )

    assert everywhere, "no celery command found in any of the three files — the search is broken"

    for command in everywhere:
        assert _dash_a_value(command) == expected, (
            f"{command!r} names a different application than the definition site's {expected!r}"
        )


def test_every_deployed_celery_command_is_a_definition_site_command() -> None:
    """Compose holds copies because YAML cannot reference a Makefile variable — so pin the copies.

    Stricter than the substring check further down, which only asks that the base command appear
    somewhere. This asks that every celery command compose runs is exactly one of the defined ones,
    which is what stops a queue selection or a concurrency figure being edited in compose alone.
    """
    deployed = _compose_celery_commands()
    defined = _makefile_celery_commands()

    assert len(deployed) == len(defined), (
        f"docker-compose.yml runs {len(deployed)} celery commands and the Makefile defines "
        f"{len(defined)}; a service without a definition is a command nothing checks"
    )
    assert set(deployed) == set(defined.values())


def test_the_makefile_command_needs_exactly_one_substitution() -> None:
    """Guards the base expansion, so the equality tests cannot pass by checking the wrong string.

    `_makefile_worker_command` performs a single textual substitution rather than deferring to
    `make`. That is only faithful while the base command line references exactly one variable, and
    while that variable's own value references none. Both are asserted, so extending the base
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


def test_the_derived_commands_expand_completely() -> None:
    """The same guard for the fixed-point expander: nothing may be left unexpanded.

    `_expand` gives up after a bounded number of passes rather than looping forever, so a Makefile
    it cannot resolve would otherwise leave a literal `$(...)` in the string and every comparison
    above would still pass — comparing two equally-unexpanded strings.
    """
    for name, command in _makefile_celery_commands().items():
        assert not _MAKE_VARIABLE_REFERENCE.search(command), (
            f"{name} still holds an unexpanded reference after {_MAX_EXPANSION_PASSES} passes: "
            f"{command!r}"
        )


def test_exactly_one_of_the_defined_commands_is_the_scheduler() -> None:
    """Pins the worker/scheduler split this module's `-A`-only comparison relies on.

    Without it, `_is_worker_command` returning False for everything would leave
    `test_every_celery_command_anywhere_names_the_same_application` as the only check on any of
    them, and the whole-string comparisons would still pass while proving less than they read as.
    """
    commands = _makefile_celery_commands().values()
    workers = [command for command in commands if _is_worker_command(command)]
    schedulers = [command for command in commands if not _is_worker_command(command)]

    assert len(workers) == 2, f"expected two worker commands, found {workers}"
    assert len(schedulers) == 1, f"expected one scheduler command, found {schedulers}"


def test_every_worker_command_selects_its_queues_explicitly() -> None:
    """`-Q` is not tidiness, and this is the assertion that says so.

    A worker started without `-Q` consumes **every** queue the application declares — measured, not
    assumed — which includes the dead-letter queue. Such a worker re-runs precisely the messages
    that were parked there for a human to look at, and it looks completely healthy while doing it.
    Two workers over disjoint queues is also the whole mechanism that stops minutes-long ingestion
    from delaying sub-second billing work, and it collapses to nothing the moment a `-Q` is dropped.
    """
    for command in _makefile_celery_commands().values():
        if not _is_worker_command(command):
            continue
        assert "-Q" in shlex.split(command), (
            f"{command!r} starts a worker without naming its queues, so it consumes all of them, "
            "the dead-letter queue included"
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
    """C8's third Proof, armed for C7 and now satisfied by it.

    Written while C7 was blocked on an unanswered topology question, so it skipped while the compose
    file did not mention the worker and began asserting the moment it did. C7 has since added two
    worker services and a scheduler, so this is live: the base command must appear verbatim, which it
    does as the shared prefix of both worker commands.

    The skip branch is kept rather than deleted. It is the honest behaviour if the services are ever
    removed again — there is nothing to compare — and `test_every_deployed_celery_command_is_a_
    definition_site_command` is the check that would then go red, which is the right place for that
    failure to surface.

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
