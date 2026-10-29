"""Fail-closed, checkpoint-friendly budgets for Agent Saul runs.

The ledger is part of ``LegalAgentState`` rather than a process-global counter.
That matters on resume: LangGraph replays a checkpoint in another worker, so
an in-memory counter would reset and make HITL a cost-amplification primitive.
"""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class BudgetDimension(StrEnum):
    MODEL_CALLS = "model_calls"
    RETRIEVAL_CALLS = "retrieval_calls"
    TOOL_CALLS = "tool_calls"
    DEEP_RESEARCH_FANOUT = "deep_research_fanout"
    TOKEN_USAGE = "token_usage"  # noqa: S105 — budget dimension, not a credential
    WALL_CLOCK = "wall_clock"


class BudgetLimits(BaseModel, frozen=True):
    model_config = ConfigDict(extra="forbid")

    model_calls: int = Field(default=32, ge=0)
    retrieval_calls: int = Field(default=30, ge=0)
    tool_calls: int = Field(default=50, ge=0)
    deep_research_fanout: int = Field(default=4, ge=0)
    token_usage: int = Field(default=100_000, ge=0)
    wall_clock_seconds: int = Field(default=1_800, ge=1)


class BudgetExhaustedError(RuntimeError):
    """Typed internal signal translated into a terminal graph state."""

    def __init__(self, dimension: BudgetDimension, message: str) -> None:
        super().__init__(message)
        self.dimension = dimension


class BudgetLedger(BaseModel, frozen=True):
    """Persisted usage and immutable limits for one Saul run.

    ``wall_clock`` is active execution time, not time spent waiting for a
    human.  The active segment and the accumulated elapsed value are both
    checkpoint data, so moving a suspended run to another worker cannot reset
    the budget or charge an arbitrarily long HITL pause.
    """

    model_config = ConfigDict(extra="forbid")

    limits: BudgetLimits = Field(default_factory=BudgetLimits)
    started_at: datetime = Field(default_factory=lambda: datetime.now(tz=UTC))
    # ``None`` means the run is suspended.  A legacy checkpoint has no
    # ``active_started_at``; it is treated as active from ``started_at`` until
    # its first pause/resume cycle.
    active_started_at: datetime | None = None
    active_elapsed_seconds: float = Field(default=0.0, ge=0.0)
    suspended: bool = False
    model_calls: int = Field(default=0, ge=0)
    retrieval_calls: int = Field(default=0, ge=0)
    tool_calls: int = Field(default=0, ge=0)
    deep_research_fanout: int = Field(default=0, ge=0)
    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)
    exhausted: bool = False
    exhausted_dimension: BudgetDimension | None = None

    @property
    def token_usage(self) -> int:
        return self.input_tokens + self.output_tokens

    @staticmethod
    def _utc(value: datetime) -> datetime:
        if value.tzinfo is None:
            return value.replace(tzinfo=UTC)
        return value

    def pause(self, *, now: datetime | None = None) -> BudgetLedger:
        """Checkpoint the active segment before waiting for a human.

        Calling ``pause`` more than once is idempotent.  It intentionally does
        not reset ``active_elapsed_seconds``; that value is the anti-bypass
        fence carried through every checkpoint.
        """
        if self.suspended:
            return self
        observed = self._utc(now or datetime.now(tz=UTC))
        active_started = self._utc(self.active_started_at or self.started_at)
        segment = max(0.0, (observed - active_started).total_seconds())
        return self.model_copy(
            update={
                "active_started_at": None,
                "active_elapsed_seconds": self.active_elapsed_seconds + segment,
                "suspended": True,
            }
        )

    def resume(self, *, now: datetime | None = None) -> BudgetLedger:
        """Start a new active segment without resetting prior elapsed time."""
        if self.active_started_at is not None:
            return self
        observed = self._utc(now or datetime.now(tz=UTC))
        return self.model_copy(update={"active_started_at": observed, "suspended": False})

    def consume(
        self,
        dimension: BudgetDimension,
        amount: int = 1,
        *,
        input_tokens: int = 0,
        output_tokens: int = 0,
        now: datetime | None = None,
    ) -> BudgetLedger:
        if amount < 0 or input_tokens < 0 or output_tokens < 0:
            message = "budget usage cannot be negative"
            raise ValueError(message)
        if self.exhausted:
            raise BudgetExhaustedError(
                self.exhausted_dimension or dimension,
                "Agent run budget was already exhausted",
            )
        observed = self._utc(now or datetime.now(tz=UTC))
        active_started = self.active_started_at
        # A v1 checkpoint did not persist an active marker.  Preserve its
        # original semantics for the active run, while new pause checkpoints
        # use ``None`` to exclude the HITL interval.
        segment = 0.0
        if not self.suspended and active_started is not None:
            segment = max(0.0, (observed - self._utc(active_started)).total_seconds())
        elif not self.suspended and self.active_elapsed_seconds == 0:
            segment = max(0.0, (observed - self._utc(self.started_at)).total_seconds())
        active_elapsed = self.active_elapsed_seconds + segment
        if active_elapsed >= self.limits.wall_clock_seconds:
            raise BudgetExhaustedError(
                dimension=WALL_CLOCK_DIMENSION, message="Agent run wall-clock budget exhausted"
            )
        updates: dict[str, int] = {
            "model_calls": self.model_calls,
            "retrieval_calls": self.retrieval_calls,
            "tool_calls": self.tool_calls,
            "deep_research_fanout": self.deep_research_fanout,
            "input_tokens": self.input_tokens + input_tokens,
            "output_tokens": self.output_tokens + output_tokens,
        }
        field = dimension.value
        if field in updates and dimension is not BudgetDimension.TOKEN_USAGE:
            updates[field] += amount
        elif dimension is BudgetDimension.TOKEN_USAGE:
            updates["output_tokens"] += amount
        limit = {
            BudgetDimension.MODEL_CALLS: self.limits.model_calls,
            BudgetDimension.RETRIEVAL_CALLS: self.limits.retrieval_calls,
            BudgetDimension.TOOL_CALLS: self.limits.tool_calls,
            BudgetDimension.DEEP_RESEARCH_FANOUT: self.limits.deep_research_fanout,
            BudgetDimension.TOKEN_USAGE: self.limits.token_usage,
        }.get(dimension)
        used = updates.get(field, updates["input_tokens"] + updates["output_tokens"])
        if limit is not None and used > limit:
            raise BudgetExhaustedError(dimension, f"Agent run {dimension.value} budget exhausted")
        if updates["input_tokens"] + updates["output_tokens"] > self.limits.token_usage:
            raise BudgetExhaustedError(
                BudgetDimension.TOKEN_USAGE,
                "Agent run token-usage budget exhausted",
            )
        return self.model_copy(
            update={
                **updates,
                "active_elapsed_seconds": active_elapsed,
                # A resumed graph naturally reaches ``consume`` first.  Treat
                # that call as the resume boundary so callers cannot forget to
                # invoke ``resume`` and accidentally leave the ledger paused.
                "active_started_at": observed,
                "suspended": False,
            }
        )


WALL_CLOCK_DIMENSION = BudgetDimension.WALL_CLOCK


def new_budget_ledger(limits: BudgetLimits | None = None) -> BudgetLedger:
    return BudgetLedger(limits=limits or BudgetLimits())
