"""Child reset and orderly cleanup; no slow provisioning in init handlers."""

from celery.exceptions import CeleryError
from celery.signals import (
    celeryd_after_setup,
    worker_process_init,
    worker_process_shutdown,
    worker_shutdown,
)
from kombu import Connection
from kombu.exceptions import OperationalError

from app.config import get_settings
from app.features.auth.application.revocation.pending import close_production_resources
from app.observability.logging import logger
from app.shared.jobs.topology import declare_dead_letter_topology
from app.worker.celery.task import ResilientTask
from app.worker.execution import runtime


@celeryd_after_setup.connect
def declare_broker_topology(**_kwargs: object) -> None:
    """Declare the dead-letter topology once per worker start (G6).

    Producers declare before their first publish too; workers declare at
    start so a poison ``Reject(requeue=False)`` is parked even when the
    rejecting worker runs before any producer publish in this broker's
    lifetime.  ``celeryd_after_setup`` fires only when a worker actually
    starts (never at ``Worker`` construction — which unit tests do — nor in
    pool children), and the connection here is the one place a worker boot
    may pay a broker round trip.  Declares are idempotent; a failure never
    blocks the worker start — the next boot retries.
    """
    settings = get_settings()
    try:
        with Connection(settings.RABBITMQ_URL) as connection:
            declare_dead_letter_topology(connection)
        logger.info(
            "Dead-letter topology declared at worker boot",
            exchange=settings.CELERY_DEAD_LETTER_EXCHANGE,
        )
    except (CeleryError, OperationalError, ConnectionError, OSError, TimeoutError) as exc:
        exc.add_note("operation=dead_letter_topology_declaration, stage=celeryd_after_setup")
        logger.warning("Dead-letter topology declaration failed at worker boot; will retry")


@worker_process_init.connect
def initialize_worker(**_kwargs: object) -> None:
    """Reset cached state inherited through prefork without making connections."""
    runtime.reset_after_fork()
    ResilientTask._redis_client = None
    ResilientTask._redis_pid = None


@worker_process_shutdown.connect
@worker_shutdown.connect
def shutdown_worker(**_kwargs: object) -> None:
    """Close reliability Redis on its owning loop, then close that loop."""
    client = ResilientTask._redis_client
    ResilientTask._redis_client = None
    try:
        if client is not None:
            runtime.run(client.aclose())
    except Exception as exc:  # noqa: BLE001 — shutdown must never raise
        exc.add_note("operation=shutdown_worker, resource=reliability_redis")
        logger.bind(error_type=type(exc).__name__).warning("worker_reliability_redis_close_failed")
    finally:
        close_production_resources()
        runtime.close()
