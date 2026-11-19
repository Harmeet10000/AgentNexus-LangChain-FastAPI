"""No-secrets verification for the structured observability events (harden 9.1).

Design §7: logs carry correlation IDs, tenant/document hashes or IDs permitted
by policy, versions, statuses, retryability, and latency — never raw legal text
or secrets. This suite proves it at the emission seams: a planted legal
sentence and a planted API key never appear in any emitted structured event,
the bind keys stay inside the safe correlation vocabulary, and the access-audit
record is schema-incapable of carrying content at all.
"""

from __future__ import annotations

from contextlib import ExitStack
from types import SimpleNamespace
from typing import TYPE_CHECKING

import pytest
from loguru import logger as loguru_logger
from pydantic import ValidationError

from app.config.settings import get_settings
from app.features.documents.application.access_audit import DocumentAccessAuditEvent
from app.features.documents.retrieval.graph.budgets import (
    StageBudget,
    StageBudgetTracker,
    reset_degradation_window,
)
from app.observability import degradation
from app.resilience.retry import _before_sleep_log

if TYPE_CHECKING:
    from typing import Any

pytestmark = pytest.mark.unit

_LEGAL_TEXT = (
    "The Supplier shall indemnify the Customer against all losses arising "
    "from breach of clause 7.1 of the Services Agreement."
)
_API_KEY = "sk-live-abc123SECRETKEY"
_MARKERS = (_LEGAL_TEXT, _API_KEY)

# Every bind key the audited seams may emit — ids, codes, counts, versions,
# and statuses only (design §7's correlation contract).
_SAFE_KEYS = {
    # degradation.record_degradation + document_degradation_alert
    "counter",
    "degradations",
    "window_seconds",
    "alert",
    "surface",
    "reason",
    "document_class",
    "stage",
    # stage_budget_degradation / retrieval_budget_degradation_alert
    "budget_kind",
    "observed",
    "limit",
    "stages",
    "document_classes",
    # retry_scheduled
    "attempt",
    "waited",
    "outcome",
}

# Content-bearing names the audit record must never declare.
_CONTENT_FIELD_FRAGMENTS = (
    "content",
    "text",
    "quote",
    "snippet",
    "body",
    "secret",
    "token",
    "api_key",
    "password",
)


@pytest.fixture(autouse=True)
def _reset_windows() -> None:
    degradation.reset_degradation_windows()
    reset_degradation_window()
    yield
    degradation.reset_degradation_windows()
    reset_degradation_window()


@pytest.fixture
def _low_threshold(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DOCUMENTS_DEGRADATION_ALERT_THRESHOLD", "2")
    monkeypatch.setenv("DOCUMENTS_DEGRADATION_ALERT_WINDOW_SECONDS", "60")
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def _captured_events() -> tuple[list[dict[str, Any]], ExitStack]:
    """Capture the audited structured events emitted while the stack is open."""
    events: list[dict[str, Any]] = []
    stack = ExitStack()
    audited = {
        "document_degradation_alert",
        "stage_budget_degradation",
        "retrieval_budget_degradation_alert",
        "retry_scheduled",
    }

    def _sink(message: Any) -> None:
        record = message.record
        if record["message"] in audited:
            events.append({"event": record["message"], **dict(record["extra"])})

    sink_id = loguru_logger.add(_sink, level="DEBUG", format="{message}")
    stack.callback(loguru_logger.remove, sink_id)
    return events, stack


def _assert_no_markers(events: list[dict[str, Any]]) -> None:
    rendered = repr(events)
    for marker in _MARKERS:
        assert marker not in rendered, "raw legal text or a secret reached a structured event"


def _assert_safe_keys(events: list[dict[str, Any]]) -> None:
    for event in events:
        unsafe = set(event) - _SAFE_KEYS - {"event"}
        assert not unsafe, f"unsafe correlation keys in structured event: {sorted(unsafe)}"


@pytest.mark.usefixtures("_low_threshold")
def test_degradation_events_carry_only_safe_correlation() -> None:
    events, stack = _captured_events()
    with stack:
        for _ in range(2):
            degradation.record_degradation(
                degradation.NO_EVIDENCE,
                surface="evidence_search",
                reason="graphiti_outage",
                document_class="legal_contract",
            )
        degradation.record_degradation(degradation.INDEX_NOT_READY, surface="evidence_search")

    assert events, "the degradation seams emitted no events to verify"
    assert any(event["event"] == "document_degradation_alert" for event in events)
    _assert_safe_keys(events)
    _assert_no_markers(events)


@pytest.mark.usefixtures("_low_threshold")
def test_budget_events_carry_only_stage_and_class_attribution() -> None:
    events, stack = _captured_events()
    with stack:
        tracker = StageBudgetTracker(
            budgets={"generate": StageBudget(latency_ms=60_000, max_calls=1)},
            alert_threshold=2,
            alert_window_seconds=300,
        )
        for _ in range(2):
            tracker.try_charge("generate", document_class="legal_contract")
        tracker.check_latency("plan", 999_999.0, document_class="legal_contract")

    emitted = {event["event"] for event in events}
    assert "stage_budget_degradation" in emitted
    # The tracker echoes document_class verbatim into the event, so the marker
    # cannot ride THAT input (the no-content rule lives at the callers); the
    # masked seam — a content-bearing exception message — is covered by the
    # retry test below. The marker check here still guards the values the
    # tracker itself synthesizes (stage names, counts, windows).
    _assert_safe_keys(events)
    _assert_no_markers(events)


def test_retry_events_carry_only_attempt_outcome() -> None:
    """A content-bearing exception message must not reach the retry event.

    Greptile wave-3 finding: the planted TimeoutError used to be EMPTY, so
    switching the logger to include ``str(exception)`` would still pass this
    test. The message now carries both markers — realistic, since provider
    errors embed request context — and the event must still name only the
    exception TYPE.
    """
    events, stack = _captured_events()
    with stack:

        def _timeout() -> TimeoutError:
            message = f"{_API_KEY}: upstream quota exceeded while reading {_LEGAL_TEXT}"
            return TimeoutError(message)

        _before_sleep_log(
            SimpleNamespace(
                attempt_number=2,
                seconds_since_start=0.4,
                outcome=SimpleNamespace(failed=True, exception=_timeout),
            )
        )

    assert [event["event"] for event in events] == ["retry_scheduled"]
    assert events[0]["outcome"] == "TimeoutError", (
        "the retry event must attribute the exception by type only"
    )
    _assert_safe_keys(events)
    _assert_no_markers(events)


def test_access_audit_record_is_schema_incapable_of_carrying_content() -> None:
    declared = set(DocumentAccessAuditEvent.model_fields)
    assert declared, "the audit record declared no fields"
    for fragment in _CONTENT_FIELD_FRAGMENTS:
        assert not any(fragment in field for field in declared), (
            f"audit record declares a content-bearing field matching {fragment!r}"
        )

    # extra="forbid": a caller cannot smuggle chunk text or a secret into the
    # accountability record even by accident.
    with pytest.raises(ValidationError):
        DocumentAccessAuditEvent(
            tenant_id="tenant-a",
            principal_id="user-1",
            operation="evidence.read",
            policy="mechanically_verified",
            outcome="ok",
            content=_LEGAL_TEXT,  # ty: ignore[unknown-argument]
        )
    with pytest.raises(ValidationError):
        DocumentAccessAuditEvent(
            tenant_id="tenant-a",
            principal_id="user-1",
            operation="evidence.read",
            policy="mechanically_verified",
            outcome="ok",
            api_key=_API_KEY,  # ty: ignore[unknown-argument]
        )
