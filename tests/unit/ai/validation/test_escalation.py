"""Escalation routing tests: caller-decided sinks, and the import boundary."""

from __future__ import annotations

import importlib
import inspect
import pkgutil
import types

from conftest import Clause

import app.ai.validation
import app.ai.validation.pipeline as pipeline_module
from app.ai.validation import (
    DeadLetterSink,
    EscalationSink,
    EvaluationContext,
    GateReport,
    HumanReviewSink,
    TerminalFailureSink,
    ValidationDecision,
    ValidationOutcome,
    ValidationReason,
)


def _escalated_outcome() -> ValidationOutcome[Clause]:
    return ValidationOutcome[Clause](
        decision=ValidationDecision.ESCALATE,
        value=None,
        attempts=2,
        schema_gate=GateReport.passed(Clause(text="clause")),
        semantic=None,
        reason=ValidationReason.EVALUATOR_UNAVAILABLE,
    )


async def test_same_escalate_routes_differently_per_sink() -> None:
    """Two sinks receive the identical ESCALATE outcome; no shared routing exists."""
    terminal_received: list[ValidationOutcome[Clause]] = []
    review_received: list[ValidationOutcome[Clause]] = []
    dead_letter_received: list[ValidationOutcome[Clause]] = []
    context = EvaluationContext(seam="seam_a")

    async def _to_terminal(outcome: ValidationOutcome[Clause], context: EvaluationContext) -> None:
        terminal_received.append(outcome)

    async def _to_review(outcome: ValidationOutcome[Clause], context: EvaluationContext) -> None:
        review_received.append(outcome)

    async def _to_dead_letter(
        outcome: ValidationOutcome[Clause], context: EvaluationContext
    ) -> None:
        dead_letter_received.append(outcome)

    terminal = TerminalFailureSink(handler=_to_terminal)
    review = HumanReviewSink(enqueue=_to_review)
    dead_letter = DeadLetterSink(handler=_to_dead_letter)
    outcome = _escalated_outcome()

    assert isinstance(terminal, EscalationSink)
    assert isinstance(review, EscalationSink)
    assert isinstance(dead_letter, EscalationSink)

    await terminal.escalate(outcome, context)
    await review.escalate(outcome, context)
    await dead_letter.escalate(outcome, context)

    # Equivalence: every sink saw the very same outcome object, unchanged.
    assert terminal_received == [outcome]
    assert review_received == [outcome]
    assert dead_letter_received == [outcome]
    assert outcome.decision is ValidationDecision.ESCALATE
    assert outcome.reason is ValidationReason.EVALUATOR_UNAVAILABLE


def test_pipeline_module_imports_no_sink_implementation() -> None:
    """The pipeline decides ESCALATE; it does not import routing."""
    source = inspect.getsource(pipeline_module)
    assert "Sink" not in source
    assert ".escalation" not in source
    imported_modules = [
        value for value in vars(pipeline_module).values() if isinstance(value, types.ModuleType)
    ]
    names = {module.__name__ for module in imported_modules}
    assert not any(name.startswith("app.features") for name in names)
    assert not any(name.endswith("escalation") for name in names)


def test_validation_package_imports_nothing_from_features() -> None:
    """Import boundary: app/ai/validation never touches app.features."""
    modules = [app.ai.validation]
    modules.extend(
        importlib.import_module(module_info.name)
        for module_info in pkgutil.walk_packages(
            app.ai.validation.__path__, prefix="app.ai.validation."
        )
    )

    for module in modules:
        imported = [value for value in vars(module).values() if isinstance(value, types.ModuleType)]
        for imported_module in imported:
            name = imported_module.__name__
            assert not name.startswith("app.features"), f"{module.__name__} imports {name}"
