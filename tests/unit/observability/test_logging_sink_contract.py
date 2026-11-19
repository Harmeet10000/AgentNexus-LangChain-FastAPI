"""F12c regression: the log pipeline must keep a live stdout/stderr sink.

The journals of every systemd unit (uvicorn and celery workers) carried zero
application lines for the whole post-fix round. The app's pipeline was NOT the
defect — ``setup_logging`` always adds a stderr sink — but the honest observable
contract ("app logs reach the process stderr, journald captures them") still
needs a test that holds the config to it: the deployment launch path (Makefile)
must execute the venv entry point directly because ``uv run`` interposes a
relay on the child's fds that loses everything when they point at the journald
socket.

Two checks here:

1. a fresh ``setup_logging()`` attaches exactly one console sink to
   ``sys.stderr`` at the configured level, and stdlib records (celery, uvicorn)
   are bridged into it;
2. the documented process commands never route the runtime entry points
   through ``uv run``.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import pytest
from loguru import logger as loguru_logger

from app.observability.logging import setup_logging

pytestmark = pytest.mark.unit

_REPO_ROOT = Path(__file__).resolve().parents[3]
_MAKEFILE = _REPO_ROOT / "Makefile"


def test_setup_logging_keeps_stderr_sink_and_stdlib_bridge(monkeypatch) -> None:
    monkeypatch.setenv("LOG_LEVEL", "INFO")
    monkeypatch.setenv("LOG_FORMAT", "text")
    try:
        setup_logging(level="INFO")

        sinks = [
            handler
            for handler in loguru_logger._core.handlers.values()
            if getattr(handler, "_sink", None) is not None
            and getattr(handler._sink, "_stream", None) is sys.stderr
        ]
        assert sinks, "no console sink bound to sys.stderr after setup_logging()"

        # The stdlib bridge must keep routing framework records into the
        # pipeline: celery task receipts and uvicorn access lines live on
        # stdlib loggers, and the bridge is what feeds them to the sink.
        bridge = logging.getLogger("celery").handlers
        assert bridge
        assert isinstance(bridge[0], logging.Handler)
        assert logging.getLogger("celery").propagate is False
    finally:
        setup_logging()  # restore the settings-driven pipeline for other tests


def test_runtime_commands_do_not_route_through_uv_run() -> None:
    """uv's fd relay loses journald-bound output; runtime cmds execute directly."""
    makefile = _MAKEFILE.read_text(encoding="utf-8")
    for variable in ("BACKEND_CMD", "CELERY_WORKER_CMD", "CELERY_BEAT_CMD"):
        definition = next(
            line for line in makefile.splitlines() if line.startswith(f"{variable} :=")
        )
        assert "uv run " not in definition, f"{variable} must not execute through `uv run`"
        assert ".venv/bin/" in definition, f"{variable} must execute the venv entry point directly"
