"""Child reset and orderly cleanup; no slow provisioning in init handlers."""

from celery.signals import worker_process_init, worker_process_shutdown, worker_shutdown

from app.features.auth.pending_revocation import close_production_resources
from app.observability.logging import logger
from app.worker.celery.task import ResilientTask
from app.worker.execution import runtime


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
