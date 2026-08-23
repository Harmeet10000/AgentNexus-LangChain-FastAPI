"""Task registration is explicit, not an import side effect.

The ingestion task was registered before this suite existed, but only because the
task package's initialiser imports it and importing any listed sibling imports
that initialiser first. These tests pin the replacement: the module list itself
names every module that declares a task, so tidying the initialiser cannot stop a
dispatched name from being registered.

No worker is started anywhere here. The task application is interrogated by
importing it, which is what a deployment's readiness check does too. The real
application arrives through the ``real_celery`` fixture because the suite-wide
conftest replaces it, and this file's whole subject is what the real one holds.
"""

import pytest

from app.connections.celery_task_names import (
    DOCUMENTS_INGEST,
    PAGEINDEX_INGEST,
    TASK_DECLARING_MODULES,
)


@pytest.fixture(scope="module")
def all_declaring_modules_imported(real_celery):
    """Import every declaring module once, and hand back the registry that recorded them.

    Module-scoped because this is the expensive fixture in the suite: the
    ingestion modules pull the document-converter stack, so the first import of
    the full list costs seconds. The dispatch helper never does this — it imports
    the one module that declares the name it was handed — but a test asking
    whether *every* declared name is backed has no cheaper way to find out.
    """
    registry = real_celery.registry.CeleryTaskRegistry
    for task_name in TASK_DECLARING_MODULES:
        registry.ensure_declared_module_imported(task_name)
    return real_celery


def test_every_declaring_module_is_named_in_the_include_list(real_celery):
    """The two lists are kept as two lists on purpose, so this is the seam.

    Deriving the include list from the mapping would make it unreadable at the one
    place an operator looks; deriving the mapping from the include list would lose
    which name each module declares. So they are written independently and this
    asserts they agree — in both directions, because a module listed but declaring
    nothing is as much a mistake as a declaring module left out.
    """
    assert set(TASK_DECLARING_MODULES.values()) == set(real_celery.app.conf.include)


@pytest.mark.parametrize("task_name", [DOCUMENTS_INGEST, PAGEINDEX_INGEST])
def test_dispatched_task_modules_are_listed_explicitly(task_name, real_celery):
    """The dispatched names' own modules must be listed, not reached through a sibling."""
    assert TASK_DECLARING_MODULES[task_name] in real_celery.app.conf.include


def test_the_ingestion_module_is_listed_rather_than_reached_through_the_initialiser(real_celery):
    """The regression this whole task exists to prevent, stated as one assertion.

    Before, this module was absent from the list and arrived only because the task
    package's initialiser imported it. Removing that import — which an unrelated
    tidy-up would reasonably do — left dispatch succeeding into a queue whose
    worker had never heard of the name.
    """
    assert TASK_DECLARING_MODULES[DOCUMENTS_INGEST] in real_celery.app.conf.include


def test_every_declared_task_name_has_a_registered_payload_model(all_declaring_modules_imported):
    """Nothing dispatchable may be left with no model to be checked against.

    The dispatch helper now refuses an unregistered name outright, so a name
    declared without a model is not a soft gap that logs a warning — it is a name
    that cannot be dispatched at all. This is the check that keeps the two facts
    from drifting apart quietly.
    """
    registry = all_declaring_modules_imported.registry.CeleryTaskRegistry

    missing = sorted(set(TASK_DECLARING_MODULES) - registry.registered_names())

    assert missing == []


def test_every_declared_task_name_is_bound_on_the_task_application(all_declaring_modules_imported):
    """Importing the declaring module must actually bind the name Celery routes on.

    A registered payload model and a bound task are two different facts: the first
    is what the dispatching side checks against, the second is what a worker can
    answer. A name with only the first dispatches cleanly and is then discarded.
    """
    bound = all_declaring_modules_imported.app.tasks

    unbound = sorted(name for name in TASK_DECLARING_MODULES if name not in bound)

    assert unbound == []


def test_declared_but_unimplemented_task_is_registered_and_fails_explicitly(real_celery):
    """A deferred task must fail with its own diagnostic, not an unknown-name one.

    Leaving it out of the module list would have produced the worse of the two
    failures: the dispatch succeeds, the worker rejects a name it has never seen,
    and the only evidence is work that never happened.
    """
    real_celery.registry.CeleryTaskRegistry.ensure_declared_module_imported(PAGEINDEX_INGEST)

    assert PAGEINDEX_INGEST in real_celery.app.tasks

    # The arguments satisfy the declared signature and are otherwise irrelevant:
    # the body raises before it looks at either of them.
    with pytest.raises(NotImplementedError):
        real_celery.app.tasks[PAGEINDEX_INGEST].run(file_path="unread.pdf", user_id="u1")


def test_the_typed_email_reference_module_is_not_listed(real_celery):
    """It declares the live email names; listing it lets import order pick the winner."""
    assert "tasks.auth_email_tasks_typed" not in real_celery.app.conf.include
