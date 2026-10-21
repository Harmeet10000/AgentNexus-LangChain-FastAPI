"""Installed worker and scheduler entry points."""

import argparse
import sys
from collections.abc import Sequence

from app.config import get_settings
from app.worker.celery.profiles import WorkerRole, profile_for_role
from app.worker.completeness import check_deployment_queue_coverage


def _worker_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Start one role-specific Celery worker")
    parser.add_argument(
        "--role",
        choices=tuple(role.value for role in WorkerRole),
        default=WorkerRole.DEFAULT.value,
        help="Worker profile to run (queue, pool, concurrency and task subset).",
    )
    return parser


def build_worker_argv(role: WorkerRole | str, extra: Sequence[str] = ()) -> tuple[str, ...]:
    """Build one complete Celery worker command from an immutable profile.

    Queue and process-shape flags are owned by the profile.  Rejecting caller
    overrides is intentional: two ``-Q`` flags are parsed differently by
    Celery versions and can silently broaden consumption to the DLQ.
    """

    profile = profile_for_role(role)
    forbidden = {"-Q", "--queues", "--pool", "--concurrency", "--max-tasks-per-child"}
    tokens = tuple(extra)
    for token in tokens:
        option = token.split("=", 1)[0]
        if option in forbidden or (token.startswith("-Q") and len(token) > 2):
            option = "-Q" if token.startswith("-Q") else option
            message = (
                f"worker role {profile.role.value!r} owns {option}; do not pass duplicate flags"
            )
            raise ValueError(message)
    return (
        "worker",
        "--pool",
        profile.pool,
        "--concurrency",
        str(profile.concurrency),
        "--max-tasks-per-child",
        str(profile.max_tasks_per_child),
        "-Q",
        profile.queue,
        *tokens,
    )


def main(argv: Sequence[str] | None = None) -> None:
    """Start a worker selected by ``--role`` with exactly one queue."""

    parser = _worker_parser()
    args, extra = parser.parse_known_args(sys.argv[1:] if argv is None else argv)
    try:
        command = build_worker_argv(args.role, extra)
    except ValueError as exc:
        parser.error(str(exc))
    # Import/composition happens only after argparse has selected the role.
    from app.worker.celery.app import create_worker_celery_app  # noqa: PLC0415

    create_worker_celery_app(args.role).worker_main(list(command))


def beat(argv: Sequence[str] | None = None) -> None:
    """Start the periodic scheduler using the configured persistent schedule file."""

    settings = get_settings()
    args = tuple(sys.argv[1:] if argv is None else argv)
    if any(
        token in {"--schedule", "-s"}
        or token.startswith("--schedule=")
        or (token.startswith("-s") and len(token) > 2)
        for token in args
    ):
        message = "beat owns --schedule/-s; configure CELERY_BEAT_SCHEDULE_FILE instead"
        raise ValueError(message)
    from app.worker.celery.app import create_worker_celery_app  # noqa: PLC0415

    create_worker_celery_app(beat=True).start(
        ["beat", "--schedule", settings.CELERY_BEAT_SCHEDULE_FILE, *args]
    )


def completeness(argv: Sequence[str] | None = None) -> None:
    """Report deployment-wide catalog queue coverage for CI/deploy checks."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--consumed-queue",
        action="append",
        dest="consumed_queues",
        metavar="QUEUE",
        help="Queue consumed by one deployed worker; repeat for the aggregate set.",
    )
    args = parser.parse_args(argv)
    settings = get_settings()
    consumed = args.consumed_queues or []
    report = check_deployment_queue_coverage(
        consumed_queues=consumed,
        declared_queues=(
            settings.CELERY_DEFAULT_QUEUE,
            settings.CELERY_INGESTION_QUEUE,
            settings.CELERY_CRAWLER_QUEUE,
            settings.CELERY_DEAD_LETTER_QUEUE,
        ),
        ignored_queues=(settings.CELERY_DEAD_LETTER_QUEUE,),
    )
    if report.ok:
        sys.stdout.write("Celery catalog queue coverage: clean\n")
        return
    sys.stdout.write("Celery catalog queue coverage: drift detected\n")
    for error in report.errors:
        sys.stdout.write(f"- {error}\n")
    raise SystemExit(1)
