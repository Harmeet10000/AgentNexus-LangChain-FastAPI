"""Per-stage budget enforcement and the standing degradation alert (G6).

The retrieval loop's own budget machinery, built on the per-stage spans and
the ``stage_usage`` channel the retrieval-engine cutover introduced. Two budget
dimensions per stage (plan, source_identifier, retrieve_fusion, rerank,
post_process, edge_grounding, grade, generate):

- **latency** — a stage's wall-clock duration over its budget;
- **calls** — billable operations (LLM calls, retrieval reads) over the
  per-ask cap, charged *before* the spend so exhaustion stops the run instead
  of letting it continue unbounded.

Budget exhaustion is budget-honest in the validation pipeline's sense: the
caller receives the typed degraded outcome (a :class:`StageWarning` naming the
stage and the budget that fired, surfaced through the ask envelope) — never a
silent overrun and never an exception path. Defaults are deliberately
generous, so enabling budgets is behaviour-preserving on healthy traffic; the
settings maps (``DOCUMENTS_STAGE_BUDGET_LATENCY_MS``,
``DOCUMENTS_STAGE_BUDGET_MAX_CALLS``) override per stage.

The standing alert (task 4.2) is deliberately **earlier than and distinct
from** the harden change's pilot-kill criteria (9.8): sustained degradations
raise a structured ``retrieval_budget_degradation_alert`` log event the
alerting stack can consume, attributable per stage and per document class.
The kill criterion remains a separate, later decision that consumes this
alert as one input.
"""

from __future__ import annotations

import time
from collections import deque
from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict

from app.config import get_settings
from app.observability.logging import logger

if TYPE_CHECKING:
    from collections.abc import Mapping

__all__ = [
    "DEFAULT_STAGE_BUDGETS",
    "STAGE_GENERATE",
    "STAGE_GRADE",
    "StageBudget",
    "StageBudgetTracker",
    "StageBudgetViolation",
    "build_stage_budget_tracker",
    "reset_degradation_window",
]

STAGE_PLAN = "plan"
STAGE_SOURCE_IDENTIFIER = "source_identifier"
STAGE_RETRIEVE_FUSION = "retrieve_fusion"
STAGE_RERANK = "rerank"
STAGE_POST_PROCESS = "post_process"
STAGE_EDGE_GROUNDING = "edge_grounding"
STAGE_GRADE = "grade"
STAGE_GENERATE = "generate"


class StageBudget(BaseModel):
    """One stage's generous-default budget."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    latency_ms: int
    max_calls: int


#: Every stage the loop attributes spend to. Budgets absent from the maps use
#: these defaults — generous by design (the grade loop caps iterations at two,
#: so healthy asks spend one or two calls per stage), which is what makes
#: enforcement-on-by-default behaviour-preserving.
DEFAULT_STAGE_BUDGETS: Mapping[str, StageBudget] = {
    STAGE_PLAN: StageBudget(latency_ms=60_000, max_calls=8),
    STAGE_SOURCE_IDENTIFIER: StageBudget(latency_ms=60_000, max_calls=8),
    STAGE_RETRIEVE_FUSION: StageBudget(latency_ms=60_000, max_calls=8),
    STAGE_RERANK: StageBudget(latency_ms=60_000, max_calls=8),
    STAGE_POST_PROCESS: StageBudget(latency_ms=30_000, max_calls=8),
    STAGE_EDGE_GROUNDING: StageBudget(latency_ms=60_000, max_calls=8),
    STAGE_GRADE: StageBudget(latency_ms=60_000, max_calls=8),
    STAGE_GENERATE: StageBudget(latency_ms=180_000, max_calls=8),
}


class StageBudgetViolation(BaseModel):
    """The typed record of one budget exhaustion, naming stage and budget."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    stage: str
    budget_kind: str  # "calls" | "latency_ms"
    observed: float
    limit: float
    document_class: str = "unknown"


class _WindowEntry(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    at: float
    stage: str
    document_class: str


# Process-level rolling window behind the standing alert. Keyed per tracker
# instance the module creates; tests reset it via `reset_degradation_window`.
_DEGRADATION_WINDOW: deque[_WindowEntry] = deque()
_WINDOW_MONOTONIC_BASE = time.monotonic()


def _now() -> float:
    return time.monotonic() - _WINDOW_MONOTONIC_BASE


class StageBudgetTracker:
    """Per-ask budget accounting across the loop's stages.

    ``charge`` mirrors the validation pipeline's :class:`BudgetLedger` port
    (an adapter for one stage is available via :meth:`ledger_for`): a refused
    charge is an expected answer, not an exception. Latency checks run after a
    stage completes (the span duration is the observation). Every violation is
    logged as a structured ``stage_budget_degradation`` event and feeds the
    process-level sustained-degradation window behind the standing alert.
    """

    def __init__(
        self,
        *,
        budgets: Mapping[str, StageBudget],
        alert_threshold: int,
        alert_window_seconds: int,
    ) -> None:
        self._budgets = dict(budgets)
        self._charged: dict[str, int] = {}
        self._alert_threshold = alert_threshold
        self._alert_window_seconds = alert_window_seconds

    def budget_for(self, stage: str) -> StageBudget:
        return self._budgets.get(stage) or StageBudget(latency_ms=60_000, max_calls=8)

    def charge(self, stage: str, *, calls: int = 1, document_class: str = "unknown") -> bool:
        """Charge ``calls`` against the stage's per-ask cap; False when exhausted."""
        return self.try_charge(stage, calls=calls, document_class=document_class) is None

    def try_charge(
        self, stage: str, *, calls: int = 1, document_class: str = "unknown"
    ) -> StageBudgetViolation | None:
        """Charge the stage; records and returns the violation when exhausted."""
        budget = self.budget_for(stage)
        charged = self._charged.get(stage, 0)
        if charged + calls > budget.max_calls:
            violation = StageBudgetViolation(
                stage=stage,
                budget_kind="calls",
                observed=charged + calls,
                limit=budget.max_calls,
                document_class=document_class,
            )
            self._record(violation)
            return violation
        self._charged[stage] = charged + calls
        return None

    def ledger_for(self, stage: str, *, document_class: str = "unknown") -> _StageLedgerAdapter:
        """A one-stage adapter matching the validation BudgetLedger port shape."""
        return _StageLedgerAdapter(self, stage, document_class)

    def check_latency(
        self, stage: str, duration_ms: float, *, document_class: str = "unknown"
    ) -> StageBudgetViolation | None:
        """Record a latency breach; returns the violation, or None within budget."""
        budget = self.budget_for(stage)
        if duration_ms <= budget.latency_ms:
            return None
        violation = StageBudgetViolation(
            stage=stage,
            budget_kind="latency_ms",
            observed=duration_ms,
            limit=budget.latency_ms,
            document_class=document_class,
        )
        self._record(violation)
        return violation

    def _record(self, violation: StageBudgetViolation) -> None:
        logger.bind(
            stage=violation.stage,
            budget_kind=violation.budget_kind,
            observed=violation.observed,
            limit=violation.limit,
            document_class=violation.document_class,
        ).warning("stage_budget_degradation")
        _DEGRADATION_WINDOW.append(
            _WindowEntry(
                at=_now(),
                stage=violation.stage,
                document_class=violation.document_class,
            )
        )
        self._maybe_alert()

    def _maybe_alert(self) -> None:
        cutoff = _now() - self._alert_window_seconds
        while _DEGRADATION_WINDOW and _DEGRADATION_WINDOW[0].at < cutoff:
            _DEGRADATION_WINDOW.popleft()
        if len(_DEGRADATION_WINDOW) < self._alert_threshold:
            return
        by_stage: dict[str, int] = {}
        by_class: dict[str, int] = {}
        for entry in _DEGRADATION_WINDOW:
            by_stage[entry.stage] = by_stage.get(entry.stage, 0) + 1
            by_class[entry.document_class] = by_class.get(entry.document_class, 0) + 1
        logger.bind(
            degradations=len(_DEGRADATION_WINDOW),
            window_seconds=self._alert_window_seconds,
            stages=by_stage,
            document_classes=by_class,
            alert="standing_degradation",
        ).warning("retrieval_budget_degradation_alert")
        # One alert per sustained period: the window resets so the alert is an
        # edge, and the pilot-kill evaluation remains a separate, later decision.
        _DEGRADATION_WINDOW.clear()


class _StageLedgerAdapter:
    """BudgetLedger-shaped adapter for one stage (validation pipeline parity)."""

    def __init__(self, tracker: StageBudgetTracker, stage: str, document_class: str) -> None:
        self._tracker = tracker
        self._stage = stage
        self._document_class = document_class

    def charge(self) -> bool:
        return self._tracker.charge(self._stage, document_class=self._document_class)


def build_stage_budget_tracker() -> StageBudgetTracker | None:
    """Build the per-ask tracker from settings, or None when budgets are off."""
    settings = get_settings()
    if not settings.DOCUMENTS_STAGE_BUDGETS_ENABLED:
        return None
    budgets: dict[str, StageBudget] = dict(DEFAULT_STAGE_BUDGETS)
    for stage, latency in settings.DOCUMENTS_STAGE_BUDGET_LATENCY_MS.items():
        base = budgets.get(stage) or StageBudget(latency_ms=60_000, max_calls=8)
        budgets[stage] = base.model_copy(update={"latency_ms": int(latency)})
    for stage, max_calls in settings.DOCUMENTS_STAGE_BUDGET_MAX_CALLS.items():
        base = budgets.get(stage) or StageBudget(latency_ms=60_000, max_calls=8)
        budgets[stage] = base.model_copy(update={"max_calls": int(max_calls)})
    return StageBudgetTracker(
        budgets=budgets,
        alert_threshold=settings.DOCUMENTS_STAGE_BUDGET_ALERT_THRESHOLD,
        alert_window_seconds=settings.DOCUMENTS_STAGE_BUDGET_ALERT_WINDOW_SECONDS,
    )


def reset_degradation_window() -> None:
    """Test seam: clear the process-level sustained-degradation window."""
    _DEGRADATION_WINDOW.clear()
