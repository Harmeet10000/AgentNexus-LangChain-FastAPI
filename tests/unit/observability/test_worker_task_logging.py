"""G7 regression: task-time worker logging must reach the process stderr.

The 2026-11-15 sweep saw worker BOOT lines in the journal but ZERO lines
during the whole document-ingestion window — no ``Received task``, no
``succeeded``, no ``task_payload_rejected``.  The mechanism: celery's prefork
child redirects ``sys.stderr`` into a ``LoggingProxy`` (level WARNING) BEFORE
the ``worker_process_init`` signal fires, so the loguru sink captured from
``sys.stderr`` wrote back into the stdlib chain, where celery's
``recurse_protection`` swallowed it — a silent black hole.

Two honest observable contracts are pinned here, exactly as the worker
configures them:

1. configuring logging the way the pool child does (with a celery
   ``LoggingProxy`` standing in as ``sys.stderr``) still writes task-time
   lines to the process's original stderr stream (asserted at fd level, the
   stream journald captures);
2. the worker composition owns celery's ``setup_logging`` signal, so the
   main worker process never hijacks the root logger or redirects its
   stdio in the first place.
"""

from __future__ import annotations

import logging
import sys
import weakref

import pytest
from celery.signals import setup_logging as celery_setup_logging
from celery.utils.log import LoggingProxy, get_logger
from loguru import logger as loguru_logger

from app.observability.logging import logger, setup_logging

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def _restore_pipeline() -> None:
    """Leave the settings-driven pipeline in place for other tests."""
    yield
    setup_logging()


def test_task_time_logging_survives_a_proxied_stderr(capfd) -> None:
    """The pool-child scenario: celery proxies stderr before our sink binds."""
    proxy = LoggingProxy(get_logger("celery.redirected"), logging.WARNING)
    real_stderr = sys.stderr
    sys.stderr = proxy
    try:
        # Exactly what initialize_celery_observability runs in the child.
        setup_logging(level="INFO")

        logger.bind(task="g7_probe", task_id="probe-1").info("G7_TASK_BODY_LOG")
        # Celery's own stdlib receipt lines flow through the stdlib bridge.
        get_logger("celery.worker.strategy").info("G7_RECEIVED_TASK")
    finally:
        sys.stderr = real_stderr

    captured = capfd.readouterr().err
    assert "G7_TASK_BODY_LOG" in captured, (
        "task-time loguru lines must reach the original stderr, not vanish "
        "into celery's stdio redirect"
    )
    assert "G7_RECEIVED_TASK" in captured, (
        "celery's stdlib task receipts must be bridged into the same sink"
    )


def test_task_time_logging_reaches_a_current_stderr_sink_without_a_proxy(capfd) -> None:
    """Without a proxy (pytest capture, plain run) the current stderr is used."""
    setup_logging(level="INFO")

    logger.bind(task="g7_probe").info("G7_PLAIN_LOG")

    assert "G7_PLAIN_LOG" in capfd.readouterr().err


def test_worker_composition_owns_celerys_setup_logging_signal() -> None:
    """The signal receiver keeps the main worker from hijacking/redirecting."""
    from app.worker.celery.observability import configure_celery_logging

    receivers = [
        receiver() if isinstance(receiver, weakref.ref) else receiver
        for _, receiver in celery_setup_logging.receivers
    ]
    assert configure_celery_logging in receivers

    # The receiver maps celery's numeric level onto the app's own pipeline
    # instead of accepting celery's handler stack.
    configure_celery_logging(loglevel=logging.WARNING)
    assert loguru_logger._core.handlers, "the loguru pipeline must survive the signal handler"


def test_setup_logging_does_not_bind_a_sink_to_celerys_logging_proxy() -> None:
    """No handler may hold the proxy as its stream — that is the black hole."""
    proxy = LoggingProxy(get_logger("celery.redirected"), logging.WARNING)
    real_stderr = sys.stderr
    sys.stderr = proxy
    try:
        setup_logging(level="INFO")

        streams = [
            getattr(handler._sink, "_stream", None)
            for handler in loguru_logger._core.handlers.values()
        ]
        assert proxy not in streams
    finally:
        sys.stderr = real_stderr
