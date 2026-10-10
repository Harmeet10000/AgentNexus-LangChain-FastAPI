"""Child reset and orderly cleanup; no slow provisioning in init handlers."""

from celery.signals import worker_process_init, worker_process_shutdown, worker_shutdown

from app.utils import logger
from app.worker import runtime
from app.worker.task import ResilientTask


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
    except (OSError, RuntimeError) as exc:
        exc.add_note("operation=shutdown_worker, resource=reliability_redis")
        logger.exception("Could not close worker reliability Redis")
    finally:
        runtime.close()
