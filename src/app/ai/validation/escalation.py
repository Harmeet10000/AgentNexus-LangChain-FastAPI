"""Escalation sinks: routing is a caller decision, never a pipeline one (ADR-4).

The pipeline decides ESCALATE and its reason; the three stock sinks are thin
adapters the caller wires to its own mechanisms (documents' terminal failure
path, agent_saul's HITL review seam, a dead-letter bridge). No sink is
hard-wired into the pipeline — the pipeline module does not import this one.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any, Protocol, runtime_checkable

from .schema import EvaluationContext, ValidationOutcome

type EscalationHandler = Callable[[ValidationOutcome[Any], EvaluationContext], Awaitable[None]]


@runtime_checkable
class EscalationSink(Protocol):
    """The one-method port an ESCALATE outcome is routed through."""

    async def escalate(
        self, outcome: ValidationOutcome[Any], context: EvaluationContext
    ) -> None: ...


class TerminalFailureSink:
    """Routes an ESCALATE outcome to the caller's typed-failure path."""

    def __init__(self, handler: EscalationHandler) -> None:
        self._handler = handler

    async def escalate(self, outcome: ValidationOutcome[Any], context: EvaluationContext) -> None:
        await self._handler(outcome, context)


class HumanReviewSink:
    """Enqueues an ESCALATE outcome for human-in-the-loop review."""

    def __init__(self, enqueue: EscalationHandler) -> None:
        self._enqueue = enqueue

    async def escalate(self, outcome: ValidationOutcome[Any], context: EvaluationContext) -> None:
        await self._enqueue(outcome, context)


class DeadLetterSink:
    """Bridges an ESCALATE outcome to the caller's dead-letter / terminal event."""

    def __init__(self, handler: EscalationHandler) -> None:
        self._handler = handler

    async def escalate(self, outcome: ValidationOutcome[Any], context: EvaluationContext) -> None:
        await self._handler(outcome, context)
