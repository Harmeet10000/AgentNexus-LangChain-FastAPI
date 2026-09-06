"""Regression tests for the logging standard.

Covers the two silent log-loss defects: brace-carrying messages/values
crashing ``console_format`` (KeyError → record dropped), and secret keys
escaping redaction.

Implementation note: these tests NEVER assert on captured stderr. Loguru
binds its sink stream object at import time (the real sys.stderr), so
per-test capsys/capfd assertions are unreliable here — output visibly
reaches the terminal yet ``readouterr()`` returns empty. Instead, brace
safety is tested at the exact crash site (``console_format`` output fed
through ``format_map``, replicating what loguru does with it), and
redaction/traceback through a temporary list sink.
"""

from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace
from typing import TYPE_CHECKING

from app.utils import logger
from app.utils.logger import console_format

if TYPE_CHECKING:
    from typing import Any


def _record(message: str, extra: dict[str, Any]) -> dict[str, Any]:
    return {
        "level": SimpleNamespace(name="INFO"),
        "time": datetime(2026, 9, 5, 12, 0, 0, tzinfo=UTC),
        "message": message,
        "extra": extra,
        "exception": None,
    }


def test_console_format_output_survives_format_map_with_braces() -> None:
    # console_format output is fed through format_map by loguru; any
    # leftover {placeholder} besides {exception} raises KeyError and the
    # record is lost. Reproduced pre-fix with KeyError 'oops'.
    out = console_format(
        _record("test {oops} path", {"details": {"doc_id": "x"}, "flow": "a -> b"})
    )
    formatted = out.format_map({"exception": ""})
    assert "{oops}" in formatted
    assert "doc_id" in formatted


def test_secret_keys_are_redacted() -> None:
    lines: list[str] = []
    handler_id = logger.add(lines.append, format="{message} | {extra}", level="DEBUG")
    try:
        logger.bind(password="hunter2", session_token="abc123", user_id=7).info(
            "Login attempt recorded"
        )
    finally:
        logger.remove(handler_id)
    assert lines, "expected the record to reach the sink"
    assert "*** REDACTED ***" in lines[0]
    assert "hunter2" not in lines[0]
    assert "abc123" not in lines[0]
    assert "user_id" in lines[0]


def _raise_boom() -> None:
    msg = "boom"
    raise ValueError(msg)


def test_exception_attaches_traceback() -> None:
    lines: list[str] = []
    handler_id = logger.add(lines.append, format="{message}\n{exception}", level="DEBUG")
    try:
        try:
            _raise_boom()
        except ValueError:
            logger.bind(operation="regression_probe").exception("Probe failed")
    finally:
        logger.remove(handler_id)
    assert lines, "expected the record to reach the sink"
    assert "Probe failed" in lines[0]
    assert "Traceback" in lines[0]
    assert "ValueError" in lines[0]
