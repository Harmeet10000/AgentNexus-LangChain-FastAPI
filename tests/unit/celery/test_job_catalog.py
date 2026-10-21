"""The immutable catalog/checker interface owns the Celery contract tests."""

from __future__ import annotations

import subprocess  # noqa: S404 — trusted local interpreter import smoke test
import sys

import pytest

from app.shared.jobs.catalog import JOB_CATALOG, JobQueue
from app.shared.jobs.names import AGENT_SAUL_TRANSCRIPT_SWEEP
from app.shared.jobs.payloads import NoKwargsPayload
from app.shared.jobs.topology import task_routes
from app.worker.completeness import check_job_catalog_completeness
from app.worker.schedules import BEAT_SCHEDULE
from app.worker.task_modules import TASK_DECLARING_MODULES, TASK_MODULES


def _clean_report() -> object:
    routes = task_routes()
    queues = {route["queue"] for route in routes.values()}
    return check_job_catalog_completeness(
        declaring_modules=TASK_DECLARING_MODULES,
        task_modules=TASK_MODULES,
        bound_task_names=JOB_CATALOG,
        beat_schedule=BEAT_SCHEDULE,
        declared_routes=routes,
        declared_queues=queues,
    )


def test_catalog_and_entries_are_immutable() -> None:
    definition = JOB_CATALOG[AGENT_SAUL_TRANSCRIPT_SWEEP]

    with pytest.raises(TypeError):
        JOB_CATALOG["tasks.new"] = definition  # type: ignore[index]
    with pytest.raises((TypeError, AttributeError, ValueError)):
        definition.queue = JobQueue.CRAWLER  # type: ignore[misc]


def test_catalog_carries_transcript_payload_and_worker_metadata() -> None:
    definition = JOB_CATALOG[AGENT_SAUL_TRANSCRIPT_SWEEP]

    assert definition.payload_model is NoKwargsPayload
    assert definition.queue is JobQueue.DEFAULT
    assert definition.declaring_module == "app.worker.jobs.agent_saul"
    assert AGENT_SAUL_TRANSCRIPT_SWEEP in task_routes()
    assert {entry["task"] for entry in BEAT_SCHEDULE.values()} <= set(JOB_CATALOG)


def test_producer_catalog_import_does_not_load_worker_modules() -> None:
    script = """
import sys
from app.shared.jobs.catalog import JOB_CATALOG
assert len(JOB_CATALOG) == 15
assert not any(name == "app.worker" or name.startswith("app.worker.") for name in sys.modules)
"""
    result = subprocess.run(  # noqa: S603 — literal trusted smoke-test script
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_checker_aggregates_drift_in_one_report() -> None:
    name = AGENT_SAUL_TRANSCRIPT_SWEEP
    declarations = dict(TASK_DECLARING_MODULES)
    declarations.pop(name)
    declarations["tasks.rogue"] = "app.worker.tasks.rogue"
    modules = (*TASK_MODULES, "app.worker.tasks.orphan")
    bound = set(JOB_CATALOG) - {name} | {"tasks.rogue"}
    routes = dict(task_routes())
    routes.pop(name)
    routes["tasks.rogue"] = {"queue": "missing", "routing_key": "missing"}
    beat = dict(BEAT_SCHEDULE)
    beat["rogue"] = {"task": "tasks.rogue"}

    report = check_job_catalog_completeness(
        declaring_modules=declarations,
        task_modules=modules,
        bound_task_names=bound,
        beat_schedule=beat,
        declared_routes=routes,
        declared_queues={route["queue"] for route in task_routes().values()},
    )

    assert not report.ok
    assert name in report.catalog_not_declared
    assert "tasks.rogue" in report.declared_not_catalog
    assert "app.worker.tasks.orphan" in report.listed_modules_without_declarations
    assert name in report.catalog_not_bound
    assert "tasks.rogue" in report.bound_not_catalog
    assert "tasks.rogue" in report.beat_not_catalog
    assert name in report.catalog_not_routed
    assert "tasks.rogue" in report.routes_not_catalog
    assert "tasks.rogue=missing" in report.routes_to_undeclared_queue


def test_current_worker_observations_are_complete() -> None:
    report = _clean_report()

    assert report.ok
    assert report.mismatches == {}
