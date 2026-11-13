"""Degradation counters and sustained-rise alerts for the legal evidence path.

Harden 5.9 (legal-corpus-retrieval O1): count and expose —
- Graphiti unavailability (`documents.graphiti.unavailable_total`);
- rerank fallback (`documents.retrieval.rerank_fallback_total`);
- grader fallback (`documents.retrieval.grader_fallback_total`);
- `completed_with_warnings` (`documents.ingestion.completed_with_warnings_total`);
- stuck documents (`documents.ingestion.stuck_documents_total`);
- dead-lettered events (`documents.ingestion.dead_letter_events_total`);
- `index_not_ready` (`documents.evidence.index_not_ready_total`);
- `no_evidence` (`documents.evidence.no_evidence_total`).

**Dashboard panels** (Prometheus; one panel per counter, rate over 5m plus an
absolute-value stat): each counter above is one panel query
`rate(<name>[5m])`, correlated by its attributes (`surface`, `reason`,
`document_class` — tenant-safe hashes/IDs permitted by the logging policy,
never legal text). There is no dashboard-as-code in this repository, so the
panel set is this registry: the counters are created eagerly at import so the
Prometheus reader exposes every series even before the first increment.

**Sustained-rise alerts**: a per-counter rolling window; when a counter rises
above `DOCUMENTS_DEGRADATION_ALERT_THRESHOLD` events inside
`DOCUMENTS_DEGRADATION_ALERT_WINDOW_SECONDS`, the structured
`document_degradation_alert` log event fires with the counter, window, and
safe correlation context — the same convention as the G6 budget tracker's
`retrieval_budget_degradation_alert`. One alert per sustained period (the
window resets after firing).
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Any

import opentelemetry.metrics as otel_metrics

from app.config import get_settings
from app.observability.logging import logger

__all__ = [
    "COMPLETED_WITH_WARNINGS",
    "COMPLETED_WITH_WARNINGS_TOTAL",
    "DEAD_LETTER_EVENTS",
    "DEAD_LETTER_EVENTS_TOTAL",
    "GRADER_FALLBACK",
    "GRADER_FALLBACK_TOTAL",
    "GRAPHITI_UNAVAILABLE",
    "GRAPHITI_UNAVAILABLE_TOTAL",
    "INDEX_NOT_READY",
    "INDEX_NOT_READY_TOTAL",
    "NO_EVIDENCE",
    "NO_EVIDENCE_TOTAL",
    "RERANK_FALLBACK",
    "RERANK_FALLBACK_TOTAL",
    "STUCK_DOCUMENTS",
    "STUCK_DOCUMENTS_TOTAL",
    "record_degradation",
    "reset_degradation_windows",
]

_meter = otel_metrics.get_meter("app.documents.evidence")

# The O1 counter registry (hard names; dashboards/alerts key on these).
GRAPHITI_UNAVAILABLE = "documents.graphiti.unavailable_total"
RERANK_FALLBACK = "documents.retrieval.rerank_fallback_total"
GRADER_FALLBACK = "documents.retrieval.grader_fallback_total"
COMPLETED_WITH_WARNINGS = "documents.ingestion.completed_with_warnings_total"
STUCK_DOCUMENTS = "documents.ingestion.stuck_documents_total"
DEAD_LETTER_EVENTS = "documents.ingestion.dead_letter_events_total"
INDEX_NOT_READY = "documents.evidence.index_not_ready_total"
NO_EVIDENCE = "documents.evidence.no_evidence_total"

GRAPHITI_UNAVAILABLE_TOTAL = _meter.create_counter(GRAPHITI_UNAVAILABLE, unit="1")
RERANK_FALLBACK_TOTAL = _meter.create_counter(RERANK_FALLBACK, unit="1")
GRADER_FALLBACK_TOTAL = _meter.create_counter(GRADER_FALLBACK, unit="1")
COMPLETED_WITH_WARNINGS_TOTAL = _meter.create_counter(COMPLETED_WITH_WARNINGS, unit="1")
STUCK_DOCUMENTS_TOTAL = _meter.create_counter(STUCK_DOCUMENTS, unit="1")
DEAD_LETTER_EVENTS_TOTAL = _meter.create_counter(DEAD_LETTER_EVENTS, unit="1")
INDEX_NOT_READY_TOTAL = _meter.create_counter(INDEX_NOT_READY, unit="1")
NO_EVIDENCE_TOTAL = _meter.create_counter(NO_EVIDENCE, unit="1")

_COUNTERS: dict[str, otel_metrics.Counter] = {
    GRAPHITI_UNAVAILABLE: GRAPHITI_UNAVAILABLE_TOTAL,
    RERANK_FALLBACK: RERANK_FALLBACK_TOTAL,
    GRADER_FALLBACK: GRADER_FALLBACK_TOTAL,
    COMPLETED_WITH_WARNINGS: COMPLETED_WITH_WARNINGS_TOTAL,
    STUCK_DOCUMENTS: STUCK_DOCUMENTS_TOTAL,
    DEAD_LETTER_EVENTS: DEAD_LETTER_EVENTS_TOTAL,
    INDEX_NOT_READY: INDEX_NOT_READY_TOTAL,
    NO_EVIDENCE: NO_EVIDENCE_TOTAL,
}


@dataclass
class _Window:
    entries: deque[float] = field(default_factory=deque)


_WINDOWS: dict[str, _Window] = {name: _Window() for name in _COUNTERS}
_WINDOW_MONOTONIC_BASE = time.monotonic()


def _now() -> float:
    return time.monotonic() - _WINDOW_MONOTONIC_BASE


def record_degradation(counter: str, *, amount: int = 1, **attributes: object) -> None:
    """Increment one O1 counter and feed its sustained-rise alert window.

    ``attributes`` carry tenant-safe correlation only (surface, reason,
    document_class, stage). A counter name outside the registry is a
    programming error and raises — silent drops would hide degradation.
    """
    metric = _COUNTERS[counter]
    attrs: dict[str, Any] = dict(attributes)
    metric.add(amount, attrs)
    settings = get_settings()
    if not settings.DOCUMENTS_DEGRADATION_ALERTS_ENABLED:
        return
    window = _WINDOWS[counter]
    threshold = settings.DOCUMENTS_DEGRADATION_ALERT_THRESHOLD
    window_seconds = settings.DOCUMENTS_DEGRADATION_ALERT_WINDOW_SECONDS
    cutoff = _now() - window_seconds
    while window.entries and window.entries[0] < cutoff:
        window.entries.popleft()
    for _ in range(amount):
        window.entries.append(_now())
    if len(window.entries) < threshold:
        return
    logger.bind(
        counter=counter,
        degradations=len(window.entries),
        window_seconds=window_seconds,
        alert="sustained_degradation",
        **attributes,
    ).warning("document_degradation_alert")
    # One alert per sustained period: the window resets so the alert is an
    # edge, and the pilot-kill evaluation remains a separate, later decision.
    window.entries.clear()


def reset_degradation_windows() -> None:
    """Test seam: clear every sustained-rise window."""
    for window in _WINDOWS.values():
        window.entries.clear()
