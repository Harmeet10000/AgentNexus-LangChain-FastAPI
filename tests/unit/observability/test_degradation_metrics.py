"""O1 degradation counters and sustained-rise alerts (harden 5.9).

Mirrors the G6 budget-alert test convention: a loguru sink captures the
structured events, the sustained rise fires exactly one alert per period,
and the window resets so the alert is an edge.
"""

from __future__ import annotations

from contextlib import ExitStack
from typing import TYPE_CHECKING

import pytest
from loguru import logger as loguru_logger

from app.config.settings import get_settings
from app.observability import degradation

if TYPE_CHECKING:
    from typing import Any


@pytest.fixture(autouse=True)
def _reset_windows() -> None:
    degradation.reset_degradation_windows()


@pytest.fixture
def _low_threshold(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DOCUMENTS_DEGRADATION_ALERT_THRESHOLD", "3")
    monkeypatch.setenv("DOCUMENTS_DEGRADATION_ALERT_WINDOW_SECONDS", "60")
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def _captured_events() -> tuple[list[dict[str, Any]], ExitStack]:
    events: list[dict[str, Any]] = []
    stack = ExitStack()

    def _sink(message: Any) -> None:
        record = message.record
        if record["message"] == "document_degradation_alert":
            events.append({"event": record["message"], **dict(record["extra"])})

    sink_id = loguru_logger.add(_sink, level="DEBUG", format="{message}")
    stack.callback(loguru_logger.remove, sink_id)
    return events, stack


def test_unknown_counter_is_a_programming_error() -> None:
    with pytest.raises(KeyError):
        degradation.record_degradation("documents.nonsense.total")


@pytest.mark.usefixtures("_low_threshold")
def test_sustained_rise_fires_the_alert_log_once() -> None:
    events, stack = _captured_events()
    with stack:
        for _ in range(3):
            degradation.record_degradation(degradation.NO_EVIDENCE, surface="evidence_search")
        alerts = [event for event in events if event["event"] == "document_degradation_alert"]
        assert len(alerts) == 1
        assert alerts[0]["alert"] == "sustained_degradation"
        assert alerts[0]["counter"] == degradation.NO_EVIDENCE
        assert alerts[0]["surface"] == "evidence_search"
        assert alerts[0]["degradations"] == 3
        # One alert per sustained period: the window reset makes it an edge.
        events.clear()
        for _ in range(3):
            degradation.record_degradation(degradation.NO_EVIDENCE, surface="evidence_search")
        assert len([e for e in events if e["event"] == "document_degradation_alert"]) == 1


@pytest.mark.usefixtures("_low_threshold")
def test_isolated_rise_does_not_fire() -> None:
    events, stack = _captured_events()
    with stack:
        degradation.record_degradation(degradation.INDEX_NOT_READY, surface="evidence_search")
        degradation.record_degradation(degradation.NO_EVIDENCE, surface="evidence_search")
    assert not [e for e in events if e["event"] == "document_degradation_alert"]


@pytest.mark.usefixtures("_low_threshold")
def test_alerts_can_be_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DOCUMENTS_DEGRADATION_ALERTS_ENABLED", "false")
    get_settings.cache_clear()
    try:
        events, stack = _captured_events()
        with stack:
            for _ in range(10):
                degradation.record_degradation(degradation.NO_EVIDENCE, surface="evidence_search")
        assert not [e for e in events if e["event"] == "document_degradation_alert"]
    finally:
        get_settings.cache_clear()
