"""Role interface invariants for the canonical worker runtime."""

from __future__ import annotations

import subprocess  # noqa: S404 — trusted local interpreter import smoke tests
import sys

import pytest

from app.config import get_settings
from app.shared.jobs.names import SEARCH_INGEST
from app.worker.celery.app import celery_app
from app.worker.celery.cli import build_worker_argv
from app.worker.celery.completeness import (
    check_worker_profile_completeness,
    ensure_worker_profile_completeness,
)
from app.worker.celery.profiles import profile_for_role


@pytest.mark.parametrize("role", ["default", "ingestion", "crawler"])
def test_role_profile_owns_process_shape(role: str) -> None:
    """The profile mirrors the CONFIGURED concurrency for its role.

    The expected value comes from settings, not a literal: the deployment env
    may deliberately override concurrency (e.g. the 7.5 GiB dev machine runs
    2/1/1 to stay inside its memory budget — see .env.development). Pinning
    the literal here would make every intentional env override read as a
    regression; the code defaults are pinned separately below.
    """
    profile = profile_for_role(role)
    settings = get_settings()
    queue = profile.queue
    expected_concurrency = {
        "default": settings.CELERY_DEFAULT_CONCURRENCY,
        "ingestion": settings.CELERY_INGESTION_CONCURRENCY,
        "crawler": settings.CELERY_CRAWLER_CONCURRENCY,
    }[role]
    assert profile.pool == "prefork"
    assert profile.concurrency == expected_concurrency
    assert profile.task_names
    assert profile.task_modules

    command = build_worker_argv(role)
    assert command.count("-Q") == 1
    assert command[command.index("-Q") + 1] == queue
    assert command[command.index("--pool") + 1] == "prefork"
    assert command[command.index("--concurrency") + 1] == str(expected_concurrency)
    assert profile.queue != get_settings().CELERY_DEAD_LETTER_QUEUE


@pytest.mark.parametrize(
    ("setting_name", "default_concurrency"),
    [
        ("CELERY_DEFAULT_CONCURRENCY", 8),
        ("CELERY_INGESTION_CONCURRENCY", 2),
        ("CELERY_CRAWLER_CONCURRENCY", 2),
    ],
)
def test_role_concurrency_code_defaults_are_pinned(
    setting_name: str, default_concurrency: int
) -> None:
    """The shipped code defaults are a decision, not an accident: deployment
    env files may override them (profile_for_role follows the configured
    values — see test_role_profile_owns_process_shape), but a change to the
    defaults themselves must be a visible, deliberate diff. Asserted against
    pristine settings (env file disabled); the profile singleton cannot show
    pristine values inside a configured process."""
    from app.config import Settings

    pristine = Settings(_env_file=None)
    assert getattr(pristine, setting_name) == default_concurrency


def test_ingestion_profile_registers_the_search_ingest_job() -> None:
    """tasks.search_ingest rides the ingestion queue and its declaring module
    is in that worker role's explicit import subset (CR MAJOR: a profile gap
    would leave the job queued forever with no consumer)."""
    profile = profile_for_role("ingestion")

    assert SEARCH_INGEST in profile.task_names
    assert "app.features.search.adapters.inbound.worker.tasks" in profile.task_modules


def test_role_command_rejects_duplicate_queue_or_process_flags() -> None:
    for option in (
        "-Q",
        "--queues",
        "--pool",
        "--concurrency",
        "--max-tasks-per-child",
        "-n",
        "--hostname",
    ):
        with pytest.raises(ValueError, match="owns"):
            build_worker_argv("default", (option, "other"))


@pytest.mark.parametrize("role", ["default", "ingestion", "crawler"])
def test_role_command_gives_a_distinct_node_name_per_role(role: str) -> None:
    """F5: every role previously booted as the same `celery@<host>` node.

    Celery's default node name does not include the role, so kombu's pidbox saw each
    successive role boot collide with the previous one ("A node named celery@Aging-Devil is
    already used"). The command now owns a role-scoped hostname; `%h` keeps the host suffix so
    multiple hosts still get distinct names.
    """
    command = build_worker_argv(role)

    assert command.count("-n") == 1
    assert command[command.index("-n") + 1] == f"{role}@%h"


def test_role_node_names_are_mutually_distinct() -> None:
    names = [
        build_worker_argv(role)[build_worker_argv(role).index("-n") + 1]
        for role in ("default", "ingestion", "crawler")
    ]
    assert len(set(names)) == 3


def test_canonical_app_has_one_identity() -> None:
    assert celery_app.conf.task_acks_on_failure_or_timeout is True
    assert celery_app.conf.task_acks_late is True
    assert celery_app.conf.task_reject_on_worker_lost is True


def test_jobs_namespace_registers_once() -> None:
    """The canonical jobs namespace is the only registration path."""

    import importlib

    celery_app.loader.import_default_modules()
    importlib.import_module("app.features.auth.adapters.inbound.worker.tasks")
    names = [name for name in celery_app.tasks if name.startswith("auth.")]
    assert sorted(names) == [
        "auth.pending_revocation_reconcile",
        "auth.send_password_reset_email",
        "auth.send_verification_email",
    ]


@pytest.mark.parametrize("role", ["default", "ingestion", "crawler"])
def test_role_process_imports_only_its_canonical_job_subset(role: str) -> None:
    """A fresh role process must not import unrelated task adapters."""

    script = f"""
import sys
from app.worker.celery.app import create_worker_celery_app
from app.worker.celery.profiles import profile_for_role

app = create_worker_celery_app({role!r})
profile = profile_for_role({role!r})
assert tuple(app.conf.include) == profile.task_modules
assert all(
    module.startswith("app.worker.jobs.")
    or module in {{
        "app.features.auth.adapters.inbound.worker.tasks",
        "app.features.agent_saul.adapters.inbound.worker.tasks",
        "app.features.crawler.adapters.inbound.worker.tasks",
        "app.features.documents.adapters.inbound.worker.ingestion_tasks",
        "app.features.documents.adapters.inbound.worker.maintenance_tasks",
        "app.features.search.adapters.inbound.worker.tasks",
    }}
    for module in app.conf.include
)
app.loader.import_default_modules()
expected = set(profile.task_modules)
loaded = {{name for name in expected if name in sys.modules}}
assert expected <= loaded
assert loaded <= expected
assert profile.task_names <= set(app.tasks)
"""
    result = subprocess.run(  # noqa: S603 — literal script with trusted interpreter
        [sys.executable, "-c", script], capture_output=True, text=True, timeout=120, check=False
    )
    assert result.returncode == 0, result.stderr


def test_cli_import_does_not_construct_an_app_before_role_selection() -> None:
    script = """
import sys
import app.worker.celery.cli
assert "app.worker.celery.app" not in sys.modules
"""
    result = subprocess.run(  # noqa: S603 — literal script with trusted interpreter
        [sys.executable, "-c", script], capture_output=True, text=True, timeout=60, check=False
    )
    assert result.returncode == 0, result.stderr


def test_beat_rejects_short_and_long_schedule_overrides() -> None:
    from app.worker.celery import cli

    for option in ("--schedule", "--schedule=/tmp/other", "-s", "-s/tmp/other"):
        with pytest.raises(ValueError, match="schedule/-s"):
            cli.beat([option])


@pytest.mark.parametrize(
    "name",
    [
        "app.worker.app",
        "app.worker.cli",
        "app.worker.capability_runtime",
        "app.worker.completeness",
        "app.worker.crawler",
        "app.worker.documents",
        "app.worker.observability",
        "app.worker.reliability",
        "app.worker.replay_outbox",
        "app.worker.runtime",
        "app.worker.schedules",
        "app.worker.signals",
        "app.worker.task",
        "app.worker.task_modules",
        "app.worker.tasks",
        "app.worker.tasks.agent_saul",
        "app.worker.tasks.auth",
        "app.worker.tasks.billing",
        "app.worker.tasks.crawler",
        "app.worker.tasks.credits",
        "app.worker.tasks.documents",
    ],
)
def test_obsolete_worker_modules_are_absent(name: str) -> None:
    import importlib.util

    try:
        spec = importlib.util.find_spec(name)
    except ModuleNotFoundError:
        spec = None
    assert spec is None


@pytest.mark.parametrize("role", ["default", "ingestion", "crawler"])
def test_real_worker_pre_readiness_startup_is_clean_in_a_fresh_process(role: str) -> None:
    """Celery framework builtins must not make a role worker fail startup."""

    script = f"""
from app.worker.celery.app import create_worker_celery_app
from app.worker.celery.profiles import profile_for_role

app = create_worker_celery_app({role!r})
profile = profile_for_role({role!r})
app.Worker(
    app=app,
    queues=[profile.queue],
    pool=profile.pool,
    concurrency=profile.concurrency,
)
print("startup-ok")
"""
    result = subprocess.run(  # noqa: S603 — literal script with trusted interpreter
        [sys.executable, "-c", script], capture_output=True, text=True, timeout=150, check=False
    )
    assert result.returncode == 0, result.stderr
    assert "startup-ok" in result.stdout


def test_profile_completeness_filters_framework_but_rejects_application_drift() -> None:
    role = "default"
    profile = profile_for_role(role)
    clean = check_worker_profile_completeness(
        role,
        consumed_queues=(profile.queue,),
        task_names=(*profile.task_names, "celery.map", "celery.backend_cleanup"),
        task_modules=(*profile.task_modules, "celery.app.builtins"),
    )
    assert clean.ok

    with pytest.raises(RuntimeError, match="unexpected_task_names"):
        ensure_worker_profile_completeness(
            role,
            consumed_queues=(profile.queue,),
            task_names=(
                *profile.task_names,
                "app.features.documents.adapters.inbound.worker.ingestion_tasks.ingest_document",
            ),
            task_modules=(
                *profile.task_modules,
                "app.features.documents.adapters.inbound.worker.ingestion_tasks",
            ),
        )
