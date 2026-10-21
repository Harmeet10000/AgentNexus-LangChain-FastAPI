"""Pure worker-side checks for drift around the immutable job catalog.

The checker accepts observations from a worker application instead of importing
task modules itself.  That makes it safe for tests and readiness tooling to
exercise the complete comparison without turning a producer import into a
worker import.  It deliberately returns every mismatch in one report: fixing
the first missing name should not require a second deploy to reveal the next
one.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, cast

from app.config import get_settings
from app.shared.jobs.catalog import JOB_CATALOG
from app.shared.jobs.topology import task_routes
from app.worker.celery.profiles import profile_for_role
from app.worker.celery.registration import TASK_DECLARING_MODULES, TASK_MODULES

if TYPE_CHECKING:
    from celery import Celery

    from app.shared.jobs.catalog import JobDefinition
    from app.shared.jobs.payloads import CeleryTaskPayload
    from app.worker.celery.profiles import WorkerProfile, WorkerRole


def _names(value: Iterable[str] | Mapping[str, object]) -> frozenset[str]:
    return frozenset(str(name) for name in (value.keys() if isinstance(value, Mapping) else value))


def _scheduled_names(schedule: Mapping[str, Mapping[str, object]]) -> frozenset[str]:
    return frozenset(
        task_name for entry in schedule.values() if isinstance(task_name := entry.get("task"), str)
    )


def _application_names(names: Iterable[str]) -> frozenset[str]:
    """Exclude Celery's built-in canvas/backend tasks from application drift."""
    return frozenset(name for name in names if not name.startswith("celery."))


def _application_routes(
    routes: Mapping[str, Mapping[str, str]],
) -> Mapping[str, Mapping[str, str]]:
    return {name: route for name, route in routes.items() if not name.startswith("celery.")}


def _framework_route_queues(routes: Mapping[str, Mapping[str, str]]) -> frozenset[str]:
    return frozenset(
        route["queue"]
        for name, route in routes.items()
        if name.startswith("celery.") and route.get("queue")
    )


def _route_mismatches(
    routes: Mapping[str, Mapping[str, str]], declared_queues: Iterable[str]
) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    routes = _application_routes(routes)
    queue_names = frozenset(declared_queues)
    missing_queue = tuple(sorted(name for name, route in routes.items() if not route.get("queue")))
    missing_routing_key = tuple(
        sorted(name for name, route in routes.items() if not route.get("routing_key"))
    )
    undeclared_queue = tuple(
        sorted(
            f"{name}={route.get('queue')}"
            for name, route in routes.items()
            if route.get("queue") not in queue_names
        )
    )
    return missing_queue, missing_routing_key, undeclared_queue


def _route_configuration_mismatches(
    routes: Mapping[str, Mapping[str, str]],
    expected_routes: Mapping[str, Mapping[str, str]],
) -> tuple[str, ...]:
    """Describe route metadata that diverges from the catalog-owned topology."""

    application_routes = _application_routes(routes)
    return tuple(
        sorted(
            f"{name}=expected(queue={expected.get('queue')},routing_key={expected.get('routing_key')}),"
            f"actual(queue={actual.get('queue')},routing_key={actual.get('routing_key')})"
            for name, expected in expected_routes.items()
            if (actual := application_routes.get(name)) is not None
            and (
                actual.get("queue") != expected.get("queue")
                or actual.get("routing_key") != expected.get("routing_key")
            )
        )
    )


@dataclass(frozen=True)
class JobCatalogDrift:
    """All independently actionable catalog/worker disagreement categories."""

    catalog_not_declared: tuple[str, ...] = ()
    declared_not_catalog: tuple[str, ...] = ()
    declaring_modules_not_listed: tuple[str, ...] = ()
    listed_modules_without_declarations: tuple[str, ...] = ()
    declaration_module_mismatches: tuple[str, ...] = ()
    catalog_not_bound: tuple[str, ...] = ()
    bound_not_catalog: tuple[str, ...] = ()
    beat_not_catalog: tuple[str, ...] = ()
    beat_not_declared: tuple[str, ...] = ()
    catalog_not_routed: tuple[str, ...] = ()
    routes_not_catalog: tuple[str, ...] = ()
    routes_missing_queue: tuple[str, ...] = ()
    routes_missing_routing_key: tuple[str, ...] = ()
    routes_to_undeclared_queue: tuple[str, ...] = ()
    routes_with_wrong_configuration: tuple[str, ...] = ()
    declared_queues_without_routes: tuple[str, ...] = ()
    consumed_queues_not_declared: tuple[str, ...] = ()
    consumed_queues_without_routes: tuple[str, ...] = ()

    @property
    def mismatches(self) -> Mapping[str, tuple[str, ...]]:
        """Non-empty mismatch categories, suitable for logs and assertions."""

        return MappingProxyType(
            {
                name: values
                for name, values in (
                    ("catalog_not_declared", self.catalog_not_declared),
                    ("declared_not_catalog", self.declared_not_catalog),
                    ("declaring_modules_not_listed", self.declaring_modules_not_listed),
                    (
                        "listed_modules_without_declarations",
                        self.listed_modules_without_declarations,
                    ),
                    ("declaration_module_mismatches", self.declaration_module_mismatches),
                    ("catalog_not_bound", self.catalog_not_bound),
                    ("bound_not_catalog", self.bound_not_catalog),
                    ("beat_not_catalog", self.beat_not_catalog),
                    ("beat_not_declared", self.beat_not_declared),
                    ("catalog_not_routed", self.catalog_not_routed),
                    ("routes_not_catalog", self.routes_not_catalog),
                    ("routes_missing_queue", self.routes_missing_queue),
                    ("routes_missing_routing_key", self.routes_missing_routing_key),
                    ("routes_to_undeclared_queue", self.routes_to_undeclared_queue),
                    ("routes_with_wrong_configuration", self.routes_with_wrong_configuration),
                    ("declared_queues_without_routes", self.declared_queues_without_routes),
                    ("consumed_queues_not_declared", self.consumed_queues_not_declared),
                    ("consumed_queues_without_routes", self.consumed_queues_without_routes),
                )
                if values
            }
        )

    @property
    def errors(self) -> tuple[str, ...]:
        """Flattened, stable diagnostics for CLI/readiness consumers."""

        return tuple(
            f"{category}: {name}" for category, values in self.mismatches.items() for name in values
        )

    @property
    def ok(self) -> bool:
        return not self.mismatches

    @property
    def is_clean(self) -> bool:
        return self.ok


def check_job_catalog_completeness(  # noqa: PLR0914
    *,
    catalog: Mapping[str, JobDefinition[CeleryTaskPayload]] = JOB_CATALOG,
    declaring_modules: Mapping[str, str] | None = None,
    task_modules: Iterable[str] = (),
    bound_task_names: Iterable[str] | Mapping[str, object] = (),
    beat_schedule: Mapping[str, Mapping[str, object]] = MappingProxyType({}),
    declared_routes: Mapping[str, Mapping[str, str]] = MappingProxyType({}),
    declared_queues: Iterable[str] = (),
    expected_routes: Mapping[str, Mapping[str, str]] | None = None,
    ignored_queues: Iterable[str] = (),
    consumed_queues: Iterable[str] | None = None,
    beat_entries: Mapping[str, Mapping[str, object]] | None = None,
    routes: Mapping[str, Mapping[str, str]] | None = None,
    queues: Iterable[str] | None = None,
) -> JobCatalogDrift:
    """Compare worker observations with ``catalog`` and aggregate all drift.

    ``declaring_modules`` is optional only to keep the pure checker convenient
    for producer-side unit tests; a worker completeness invocation should pass
    the actual mapping explicitly.
    """
    declarations = declaring_modules or {}
    if beat_entries is not None:
        beat_schedule = beat_entries
    if routes is not None:
        declared_routes = routes
    if queues is not None:
        declared_queues = queues
    catalog_names = frozenset(catalog)
    declaring_names = frozenset(declarations)
    module_names = frozenset(task_modules)
    bound_names = _names(bound_task_names)
    scheduled_names = _scheduled_names(beat_schedule)
    route_names = frozenset(declared_routes)

    # Celery installs built-in canvas tasks in every app.  They are framework
    # tasks, not application jobs, and therefore are not catalog drift.
    application_bound_names = _application_names(bound_names)
    application_scheduled_names = _application_names(scheduled_names)
    application_route_names = _application_names(route_names)
    declaration_module_mismatches = tuple(
        sorted(
            name
            for name in catalog_names & declaring_names
            if catalog[name].declaring_module != declarations[name]
        )
    )
    route_missing_queue, route_missing_key, route_bad_queue = _route_mismatches(
        declared_routes, declared_queues
    )
    queue_names = frozenset(declared_queues)
    framework_route_queues = _framework_route_queues(declared_routes)
    routed_queue_names = frozenset(
        route.get("queue")
        for route in _application_routes(declared_routes).values()
        if route.get("queue")
    )
    queues_without_routes = tuple(
        sorted(
            queue_names - routed_queue_names - framework_route_queues - frozenset(ignored_queues)
        )
    )
    consumed_queue_names = frozenset(consumed_queues or ()) - frozenset(ignored_queues)
    consumed_queues_not_declared = tuple(
        sorted(consumed_queue_names - queue_names - framework_route_queues)
    )
    consumed_queues_without_routes = tuple(
        sorted(consumed_queue_names - routed_queue_names - framework_route_queues)
    )
    route_configuration_mismatches = (
        _route_configuration_mismatches(declared_routes, expected_routes)
        if expected_routes is not None
        else ()
    )

    return JobCatalogDrift(
        catalog_not_declared=tuple(sorted(catalog_names - declaring_names)),
        declared_not_catalog=tuple(sorted(declaring_names - catalog_names)),
        declaring_modules_not_listed=tuple(sorted(set(declarations.values()) - module_names)),
        listed_modules_without_declarations=tuple(
            sorted(module_names - set(declarations.values()))
        ),
        declaration_module_mismatches=declaration_module_mismatches,
        catalog_not_bound=tuple(sorted(catalog_names - application_bound_names)),
        bound_not_catalog=tuple(sorted(application_bound_names - catalog_names)),
        beat_not_catalog=tuple(sorted(application_scheduled_names - catalog_names)),
        beat_not_declared=tuple(sorted(application_scheduled_names - declaring_names)),
        catalog_not_routed=tuple(sorted(catalog_names - application_route_names)),
        routes_not_catalog=tuple(sorted(application_route_names - catalog_names)),
        routes_missing_queue=route_missing_queue,
        routes_missing_routing_key=route_missing_key,
        routes_to_undeclared_queue=route_bad_queue,
        routes_with_wrong_configuration=route_configuration_mismatches,
        declared_queues_without_routes=queues_without_routes,
        consumed_queues_not_declared=consumed_queues_not_declared,
        consumed_queues_without_routes=consumed_queues_without_routes,
    )


@dataclass(frozen=True)
class QueueCoverageDrift:
    """Deployment-wide queue coverage, independent of any one worker pool."""

    required_queues_not_consumed: tuple[str, ...] = ()
    consumed_queues_not_declared: tuple[str, ...] = ()
    consumed_queues_without_routes: tuple[str, ...] = ()

    @property
    def mismatches(self) -> Mapping[str, tuple[str, ...]]:
        return MappingProxyType(
            {
                name: values
                for name, values in (
                    ("required_queues_not_consumed", self.required_queues_not_consumed),
                    ("consumed_queues_not_declared", self.consumed_queues_not_declared),
                    ("consumed_queues_without_routes", self.consumed_queues_without_routes),
                )
                if values
            }
        )

    @property
    def errors(self) -> tuple[str, ...]:
        return tuple(
            f"{category}: {name}" for category, values in self.mismatches.items() for name in values
        )

    @property
    def ok(self) -> bool:
        return not self.mismatches

    @property
    def is_clean(self) -> bool:
        return self.ok


# Short names make this pure interface easy to consume from worker tests and
# readiness checks without making the implementation part of the contract.
check_catalog_completeness = check_job_catalog_completeness
check_worker_completeness = check_job_catalog_completeness
validate_catalog_completeness = check_job_catalog_completeness


class CatalogCompletenessError(RuntimeError):
    """Worker/beat startup failed because the catalog and runtime drifted."""

    def __init__(self, report: JobCatalogDrift, *, component: str) -> None:
        self.report = report
        self.component = component
        details = "\n".join(f"- {error}" for error in report.errors)
        super().__init__(f"{component} catalog completeness check failed:\n{details}")


def _queue_names(value: object) -> tuple[str, ...]:
    if isinstance(value, Mapping):
        # Celery accepts both ``(Queue(...), ...)`` and ``{"name": Queue}``
        # configuration shapes.  In the mapping form the key is authoritative
        # when the value is a plain dict or another queue-like object without a
        # ``name`` attribute.
        names = [key for key in value if isinstance(key, str)]
        values: Iterable[object] = value.values()
    elif isinstance(value, Iterable) and not isinstance(value, (str, bytes)):
        names = []
        values = value
    else:
        return ()
    names.extend(
        name
        for item in values
        if isinstance(name := (item if isinstance(item, str) else getattr(item, "name", None)), str)
    )
    return tuple(dict.fromkeys(names))


def _mapping(value: object) -> Mapping[str, Mapping[str, object]]:
    if isinstance(value, Mapping):
        return cast("Mapping[str, Mapping[str, object]]", value)
    return {}


def _consumed_queue_names(app: Celery) -> tuple[str, ...]:
    """Observe Celery's post-selection queue set without opening a broker."""
    explicit = getattr(app, "consumed_queues", None)
    if explicit is not None:
        return _queue_names(explicit)
    amqp = getattr(app, "amqp", None)
    queues = getattr(amqp, "queues", None)
    return _queue_names(getattr(queues, "consume_from", ()))


def check_worker_catalog_completeness(app: Celery) -> JobCatalogDrift:
    """Observe one loaded Celery worker app and aggregate every drift class."""
    settings = get_settings()
    return check_job_catalog_completeness(
        catalog=JOB_CATALOG,
        declaring_modules=TASK_DECLARING_MODULES,
        task_modules=tuple(app.conf.include or ()),
        bound_task_names=app.tasks,
        beat_schedule=_mapping(app.conf.beat_schedule),
        declared_routes=cast(
            "Mapping[str, Mapping[str, str]]",
            _mapping(app.conf.task_routes),
        ),
        declared_queues=_queue_names(app.conf.task_queues),
        expected_routes=task_routes(),
        ignored_queues=(settings.CELERY_DEAD_LETTER_QUEUE,),
        consumed_queues=_consumed_queue_names(app),
    )


def check_beat_catalog_completeness(app: Celery) -> JobCatalogDrift:
    """Validate beat's catalog/config surface without requiring worker binding."""
    # Beat needs the declarations, routes, queues, and schedule to publish a
    # valid task, but does not need a worker consumer's bound-task observation.
    # Treat the immutable catalog as its binding set so beat startup stays
    # independent of worker resource provisioning.
    settings = get_settings()
    return check_job_catalog_completeness(
        catalog=JOB_CATALOG,
        declaring_modules=TASK_DECLARING_MODULES,
        task_modules=tuple(app.conf.include or TASK_MODULES),
        bound_task_names=JOB_CATALOG,
        beat_schedule=_mapping(app.conf.beat_schedule),
        declared_routes=cast(
            "Mapping[str, Mapping[str, str]]",
            _mapping(app.conf.task_routes),
        ),
        declared_queues=_queue_names(app.conf.task_queues),
        expected_routes=task_routes(),
        ignored_queues=(settings.CELERY_DEAD_LETTER_QUEUE,),
    )


def check_deployment_queue_coverage(
    *,
    consumed_queues: Iterable[str],
    required_queues: Iterable[str] | None = None,
    declared_queues: Iterable[str] | None = None,
    routed_queues: Iterable[str] | None = None,
    ignored_queues: Iterable[str] = (),
) -> QueueCoverageDrift:
    """Check aggregate queue coverage across all deployed worker pools.

    A single worker is intentionally allowed to consume only its assigned
    queue.  This check is the separate deployment-level invariant: the union
    of all worker selections must cover every queue used by the catalog.
    """
    expected_routes = task_routes()
    if routed_queues is None:
        routed = frozenset(
            route.get("queue") for route in expected_routes.values() if route.get("queue")
        )
    else:
        routed = frozenset(routed_queues)
    ignored = frozenset(ignored_queues)
    required = frozenset(required_queues if required_queues is not None else routed) - ignored
    consumed = frozenset(consumed_queues) - ignored
    declared = (frozenset(declared_queues) if declared_queues is not None else routed) - ignored
    routed -= ignored
    return QueueCoverageDrift(
        required_queues_not_consumed=tuple(sorted(required - consumed)),
        consumed_queues_not_declared=tuple(sorted(consumed - declared)),
        consumed_queues_without_routes=tuple(sorted(consumed - routed)),
    )


def ensure_worker_catalog_completeness(app: Celery) -> None:
    report = check_worker_catalog_completeness(app)
    if not report.ok:
        raise CatalogCompletenessError(report, component="worker")


def ensure_beat_catalog_completeness(app: Celery) -> None:
    report = check_beat_catalog_completeness(app)
    if not report.ok:
        raise CatalogCompletenessError(report, component="beat")


def _application_values(values: Iterable[str]) -> frozenset[str]:
    """Drop Celery framework registrations, retaining every app value."""
    return frozenset(value for value in values if not value.startswith("celery."))


@dataclass(frozen=True, slots=True)
class WorkerProfileDrift:
    """Mismatch between a role profile and the jobs a worker advertises."""

    role: WorkerRole
    expected_queue: str
    consumed_queues: tuple[str, ...]
    missing_task_names: tuple[str, ...] = ()
    unexpected_task_names: tuple[str, ...] = ()
    missing_task_modules: tuple[str, ...] = ()
    unexpected_task_modules: tuple[str, ...] = ()

    @property
    def errors(self) -> tuple[str, ...]:
        values: list[str] = []
        if self.consumed_queues != (self.expected_queue,):
            values.append(
                f"consumed_queues: expected {self.expected_queue}, got {self.consumed_queues}"
            )
        values.extend(f"missing_task_names: {name}" for name in self.missing_task_names)
        values.extend(f"unexpected_task_names: {name}" for name in self.unexpected_task_names)
        values.extend(f"missing_task_modules: {name}" for name in self.missing_task_modules)
        values.extend(f"unexpected_task_modules: {name}" for name in self.unexpected_task_modules)
        return tuple(values)

    @property
    def ok(self) -> bool:
        return not self.errors


def check_worker_profile_completeness(
    role: WorkerRole | str,
    *,
    consumed_queues: Iterable[str] = (),
    task_names: Iterable[str] = (),
    task_modules: Iterable[str] = (),
) -> WorkerProfileDrift:
    """Validate one worker's task subset and queue selection without a broker."""

    profile: WorkerProfile = profile_for_role(role)
    expected_modules = frozenset(profile.task_modules)
    expected_names = profile.task_names
    # Celery installs canvas/backend tasks and ``celery.app.builtins`` on every
    # app. They are framework-owned and intentionally absent from a role
    # profile; application names/modules remain strict so an accidental app
    # import cannot broaden a worker silently.
    actual_names = _application_values(task_names)
    actual_modules = _application_values(task_modules)
    return WorkerProfileDrift(
        role=profile.role,
        expected_queue=profile.queue,
        consumed_queues=tuple(dict.fromkeys(consumed_queues)),
        missing_task_names=tuple(sorted(expected_names - actual_names)),
        unexpected_task_names=tuple(sorted(actual_names - expected_names)),
        missing_task_modules=tuple(sorted(expected_modules - actual_modules)),
        unexpected_task_modules=tuple(sorted(actual_modules - expected_modules)),
    )


def ensure_worker_profile_completeness(
    role: WorkerRole | str,
    *,
    consumed_queues: Iterable[str] = (),
    task_names: Iterable[str] = (),
    task_modules: Iterable[str] = (),
) -> None:
    report = check_worker_profile_completeness(
        role,
        consumed_queues=consumed_queues,
        task_names=task_names,
        task_modules=task_modules,
    )
    if not report.ok:
        message = f"{report.role.value} worker profile is incomplete: {report.errors}"
        raise RuntimeError(message)


__all__ = [
    "CatalogCompletenessError",
    "JobCatalogDrift",
    "QueueCoverageDrift",
    "WorkerProfileDrift",
    "check_beat_catalog_completeness",
    "check_catalog_completeness",
    "check_deployment_queue_coverage",
    "check_job_catalog_completeness",
    "check_worker_catalog_completeness",
    "check_worker_completeness",
    "check_worker_profile_completeness",
    "ensure_beat_catalog_completeness",
    "ensure_worker_catalog_completeness",
    "ensure_worker_profile_completeness",
    "validate_catalog_completeness",
]
