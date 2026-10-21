"""Worker and beat fail-fast checks for immutable job catalog drift."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from app.shared.jobs.catalog import JOB_CATALOG
from app.shared.jobs.topology import task_routes
from app.worker.app import CatalogValidatingBeat, CatalogValidatingWorker, celery_app
from app.worker.completeness import (
    CatalogCompletenessError,
    JobCatalogDrift,
    check_beat_catalog_completeness,
    check_deployment_queue_coverage,
    check_job_catalog_completeness,
    check_worker_catalog_completeness,
    ensure_worker_catalog_completeness,
)
from app.worker.schedules import BEAT_SCHEDULE
from app.worker.task_modules import TASK_DECLARING_MODULES, TASK_MODULES


def _fake_app(
    *,
    tasks: object | None = None,
    consumed_queues: tuple[str, ...] | None = None,
    extra_queues: tuple[str, ...] = (),
) -> SimpleNamespace:
    routes = task_routes()
    queue_names = {route["queue"] for route in routes.values()} | set(extra_queues)
    queues = tuple(SimpleNamespace(name=name) for name in queue_names)
    return SimpleNamespace(
        conf=SimpleNamespace(
            include=TASK_MODULES,
            beat_schedule=BEAT_SCHEDULE,
            task_routes=routes,
            task_queues=queues,
        ),
        tasks=tasks if tasks is not None else dict.fromkeys(JOB_CATALOG),
        amqp=SimpleNamespace(
            queues=SimpleNamespace(consume_from=consumed_queues or ()),
        ),
    )


def test_loaded_worker_and_beat_observations_are_clean() -> None:
    app = _fake_app()

    assert check_worker_catalog_completeness(app).ok
    assert check_beat_catalog_completeness(app).ok


def test_beat_check_does_not_require_worker_task_binding() -> None:
    app = _fake_app(tasks={})

    beat_report = check_beat_catalog_completeness(app)
    worker_report = check_worker_catalog_completeness(app)

    assert beat_report.ok
    assert worker_report.catalog_not_bound == tuple(sorted(JOB_CATALOG))


def test_celery_framework_tasks_are_exempt_from_application_drift() -> None:
    app = _fake_app(tasks={**dict.fromkeys(JOB_CATALOG), "celery.map": object()})
    app.conf.beat_schedule = {
        **BEAT_SCHEDULE,
        "framework": {"task": "celery.backend_cleanup"},
    }
    app.conf.task_routes = {
        **task_routes(),
        "celery.map": {"queue": "default", "routing_key": "default"},
    }

    report = check_worker_catalog_completeness(app)

    assert report.ok


def test_framework_route_does_not_mask_application_unknown_queue() -> None:
    name = next(iter(JOB_CATALOG))
    app = _fake_app()
    app.conf.task_routes = {
        **task_routes(),
        name: {"queue": "framework-external", "routing_key": "framework-external"},
        "celery.backend_cleanup": {
            "queue": "framework-external",
            "routing_key": "framework-external",
        },
    }

    report = check_worker_catalog_completeness(app)

    assert report.routes_to_undeclared_queue == (f"{name}=framework-external",)
    assert not any("celery.backend_cleanup" in error for error in report.errors)


def test_drift_report_aggregates_every_runtime_category() -> None:
    name = next(iter(JOB_CATALOG))
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
        catalog=JOB_CATALOG,
        declaring_modules=declarations,
        task_modules=modules,
        bound_task_names=bound,
        beat_schedule=beat,
        declared_routes=routes,
        declared_queues={route["queue"] for route in task_routes().values()},
    )

    assert {
        "catalog_not_declared",
        "declared_not_catalog",
        "declaring_modules_not_listed",
        "listed_modules_without_declarations",
        "catalog_not_bound",
        "bound_not_catalog",
        "beat_not_catalog",
        "beat_not_declared",
        "catalog_not_routed",
        "routes_not_catalog",
        "routes_to_undeclared_queue",
    } <= set(report.mismatches)


def test_worker_gate_raises_one_aggregated_error(monkeypatch: pytest.MonkeyPatch) -> None:
    report = JobCatalogDrift(
        catalog_not_declared=("tasks.missing",),
        routes_to_undeclared_queue=("tasks.rogue=missing",),
    )
    monkeypatch.setattr(
        "app.worker.completeness.check_worker_catalog_completeness",
        lambda _app: report,
    )

    with pytest.raises(CatalogCompletenessError) as exc_info:
        ensure_worker_catalog_completeness(_fake_app())

    assert exc_info.value.component == "worker"
    assert exc_info.value.report is report
    assert "catalog_not_declared: tasks.missing" in str(exc_info.value)
    assert "routes_to_undeclared_queue: tasks.rogue=missing" in str(exc_info.value)


def test_worker_and_beat_use_validating_application_classes() -> None:
    assert celery_app.Worker is CatalogValidatingWorker
    assert celery_app.Beat is CatalogValidatingBeat


def test_worker_validates_only_the_queues_it_is_assigned() -> None:
    routes = task_routes()
    default_queue = next(route["queue"] for route in routes.values())

    report = check_worker_catalog_completeness(
        _fake_app(consumed_queues=(default_queue,)),
    )

    assert report.ok


def test_worker_rejects_an_unknown_consumed_queue() -> None:
    report = check_worker_catalog_completeness(
        _fake_app(consumed_queues=("default", "worker-typo")),
    )

    assert report.consumed_queues_not_declared == ("worker-typo",)
    assert report.consumed_queues_without_routes == ("worker-typo",)


def test_worker_rejects_a_declared_but_unrouted_consumed_queue() -> None:
    report = check_worker_catalog_completeness(
        _fake_app(consumed_queues=("default", "orphan"), extra_queues=("orphan",)),
    )

    assert report.consumed_queues_not_declared == ()
    assert report.consumed_queues_without_routes == ("orphan",)


def test_deployment_coverage_requires_the_union_of_worker_selections() -> None:
    report = check_deployment_queue_coverage(
        consumed_queues={"default"},
        required_queues={"default", "ingestion", "crawler"},
        declared_queues={"default", "ingestion", "crawler"},
        routed_queues={"default", "ingestion", "crawler"},
    )

    assert report.required_queues_not_consumed == ("crawler", "ingestion")


def test_deployment_coverage_accepts_disjoint_clean_worker_pools() -> None:
    report = check_deployment_queue_coverage(
        consumed_queues={"default", "ingestion", "crawler"},
        required_queues={"default", "ingestion", "crawler"},
        declared_queues={"default", "ingestion", "crawler", "default.dlq"},
        routed_queues={"default", "ingestion", "crawler"},
        ignored_queues={"default.dlq"},
    )

    assert report.ok


def test_beat_completeness_does_not_require_worker_consumed_queues() -> None:
    report = check_beat_catalog_completeness(
        _fake_app(consumed_queues=("default",)),
    )

    assert report.ok


def test_ci_completeness_command_reports_and_exits_on_missing_queue(
    capsys: pytest.CaptureFixture[str],
) -> None:
    from app.worker import cli

    with pytest.raises(SystemExit) as exc_info:
        cli.completeness(["--consumed-queue", "default"])

    assert exc_info.value.code == 1
    output = capsys.readouterr().out
    assert "Celery catalog queue coverage: drift detected" in output
    assert "required_queues_not_consumed" in output
