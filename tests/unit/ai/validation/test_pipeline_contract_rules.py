"""The seven contract rules, one named test each (review.md verification matrix)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from conftest import (
    FATAL_CODE,
    RETRYABLE_CODE,
    SCHEMA_ERROR_CODE,
    SEAM,
    AttemptSpec,
    RecordingBudget,
    RecordingProducer,
    ScriptedEvaluator,
    captured_decision_events,
)
from conftest import Clause as SeamSchema

from app.ai.validation import (
    RetryFeedback,
    SemanticFinding,
    SemanticReport,
    ValidationDecision,
    ValidationPipeline,
    ValidationReason,
)

if TYPE_CHECKING:
    from app.ai.validation import EvaluationContext

RETRYABLE = frozenset({SCHEMA_ERROR_CODE, RETRYABLE_CODE})


def _wired(script: list[AttemptSpec]) -> tuple[RecordingProducer, ScriptedEvaluator]:
    producer = RecordingProducer(script)
    return producer, ScriptedEvaluator(script, producer)


async def test_schema_failure_short_circuits_the_evaluator(
    context: EvaluationContext,
) -> None:
    """Rule 1: a value that fails the schema gate never reaches the evaluator."""
    producer, evaluator = _wired([AttemptSpec(schema_ok=False)])
    pipeline = ValidationPipeline(
        schema=SeamSchema,
        evaluator=evaluator,
        max_attempts=2,
        retryable_findings=frozenset(),  # schema errors are not declared retryable
    )

    outcome = await pipeline.validate(producer, context)

    assert evaluator.values_seen == []  # the evaluator was NOT invoked
    assert outcome.decision is ValidationDecision.ESCALATE
    assert outcome.reason is ValidationReason.NON_RETRYABLE_REJECTION
    assert outcome.schema_gate.ok is False
    assert outcome.schema_gate.errors[0].code == SCHEMA_ERROR_CODE
    assert outcome.semantic is None
    assert outcome.attempts == 1


async def test_schema_failure_retries_with_declared_error_codes(
    context: EvaluationContext,
) -> None:
    """Rule 1b: with the schema-error code declared retryable, the retry rules apply."""
    producer, evaluator = _wired(
        [AttemptSpec(schema_ok=False), AttemptSpec(schema_ok=True, verdict="pass")]
    )
    pipeline = ValidationPipeline(
        schema=SeamSchema,
        evaluator=evaluator,
        max_attempts=2,
        retryable_findings=RETRYABLE,
    )

    outcome = await pipeline.validate(producer, context)

    assert outcome.decision is ValidationDecision.ACCEPT
    assert evaluator.values_seen != []  # the second attempt reached the evaluator
    feedback = producer.feedbacks[1]
    assert isinstance(feedback, RetryFeedback)
    assert [error.code for error in feedback.schema_errors] == [SCHEMA_ERROR_CODE]
    assert feedback.semantic_findings == ()
    assert feedback.attempt == 1


async def test_unconfigured_evaluator_escalates_fail_closed(
    context: EvaluationContext,
) -> None:
    """Rule 2a: no evaluator configured => typed unavailable => ESCALATE, never a pass."""
    producer = RecordingProducer([AttemptSpec(schema_ok=True)])
    pipeline = ValidationPipeline(schema=SeamSchema, evaluator=None, max_attempts=1)

    outcome = await pipeline.validate(producer, context)

    assert outcome.decision is ValidationDecision.ESCALATE
    assert outcome.reason is ValidationReason.EVALUATOR_UNAVAILABLE
    assert outcome.semantic is not None
    assert outcome.semantic.verdict == "unavailable"
    assert outcome.value is None


async def test_raising_evaluator_escalates_fail_closed(
    context: EvaluationContext,
) -> None:
    """Rule 2b: an evaluator that raises => typed unavailable => ESCALATE."""
    producer, evaluator = _wired([AttemptSpec(schema_ok=True, verdict="raise")])
    pipeline = ValidationPipeline(
        schema=SeamSchema,
        evaluator=evaluator,
        max_attempts=3,
        retryable_findings=RETRYABLE,
    )

    outcome = await pipeline.validate(producer, context)

    assert outcome.decision is ValidationDecision.ESCALATE
    assert outcome.reason is ValidationReason.EVALUATOR_UNAVAILABLE
    assert outcome.semantic is not None
    assert outcome.semantic.verdict == "unavailable"
    assert "grader down" in outcome.semantic.findings[0].message


async def test_retry_carries_structured_feedback(context: EvaluationContext) -> None:
    """Rule 3: the second attempt knows why the first failed."""
    script = [
        AttemptSpec(schema_ok=False),  # first attempt: schema failure
        AttemptSpec(schema_ok=True, verdict="reject", codes=(RETRYABLE_CODE,)),
        AttemptSpec(schema_ok=True, verdict="pass"),
    ]
    producer, evaluator = _wired(script)
    pipeline = ValidationPipeline(
        schema=SeamSchema,
        evaluator=evaluator,
        max_attempts=3,
        retryable_findings=RETRYABLE,
    )

    outcome = await pipeline.validate(producer, context)

    assert outcome.decision is ValidationDecision.ACCEPT
    assert outcome.attempts == 3
    assert producer.feedbacks[0] is None  # the first invocation is never a retry
    second = producer.feedbacks[1]
    assert isinstance(second, RetryFeedback)
    assert second.attempt == 1
    assert [error.code for error in second.schema_errors] == [SCHEMA_ERROR_CODE]
    third = producer.feedbacks[2]
    assert isinstance(third, RetryFeedback)
    assert third.attempt == 2
    assert third.schema_errors == ()
    assert [finding.code for finding in third.semantic_findings] == [RETRYABLE_CODE]


async def test_non_retryable_rejection_escalates_immediately(
    context: EvaluationContext,
) -> None:
    """Rule 4: non-retryable findings escalate regardless of attempts remaining."""
    producer, evaluator = _wired(
        [AttemptSpec(schema_ok=True, verdict="reject", codes=(FATAL_CODE,))]
    )
    pipeline = ValidationPipeline(
        schema=SeamSchema,
        evaluator=evaluator,
        max_attempts=4,  # plenty of attempts remaining
        retryable_findings=RETRYABLE,
    )

    outcome = await pipeline.validate(producer, context)

    assert outcome.decision is ValidationDecision.ESCALATE
    assert outcome.reason is ValidationReason.NON_RETRYABLE_REJECTION
    assert len(producer.feedbacks) == 1  # no retry was consumed
    assert outcome.attempts == 1


async def test_each_reinvocation_charges_the_budget(context: EvaluationContext) -> None:
    """Rule 5a: every re-invocation of the producer charges the ledger."""
    script = [
        AttemptSpec(schema_ok=True, verdict="reject", codes=(RETRYABLE_CODE,)),
        AttemptSpec(schema_ok=True, verdict="reject", codes=(RETRYABLE_CODE,)),
        AttemptSpec(schema_ok=True, verdict="pass"),
    ]
    producer, evaluator = _wired(script)
    budget = RecordingBudget(capacity=5)
    pipeline = ValidationPipeline(
        schema=SeamSchema,
        evaluator=evaluator,
        max_attempts=3,
        retryable_findings=RETRYABLE,
        budget=budget,
    )

    outcome = await pipeline.validate(producer, context)

    assert outcome.decision is ValidationDecision.ACCEPT
    assert len(producer.feedbacks) == 3
    assert budget.charges == 2  # exactly the two re-invocations, never the first


async def test_budget_exhaustion_escalates_with_typed_reason(
    context: EvaluationContext,
) -> None:
    """Rule 5b: exhaustion mid-pipeline => ESCALATE budget_exhausted, no fresh allowance."""
    producer, evaluator = _wired(
        [AttemptSpec(schema_ok=True, verdict="reject", codes=(RETRYABLE_CODE,))]
    )
    budget = RecordingBudget(capacity=1)  # allows one retry, then exhausted forever
    pipeline = ValidationPipeline(
        schema=SeamSchema,
        evaluator=evaluator,
        max_attempts=10,  # attempts remain — the budget, not the loop, stops the run
        retryable_findings=RETRYABLE,
        budget=budget,
    )

    outcome = await pipeline.validate(producer, context)

    assert outcome.decision is ValidationDecision.ESCALATE
    assert outcome.reason is ValidationReason.BUDGET_EXHAUSTED
    assert len(producer.feedbacks) == 2  # one initial call, one charged retry
    assert budget.charges == 1
    # No fresh allowance: the exhausted ledger was never reset or replaced.
    assert budget.charge() is False
    assert budget.charges == 1


async def test_retryable_rejection_with_no_attempts_left_escalates(
    context: EvaluationContext,
) -> None:
    """Rule 4b: retryable findings still escalate once attempts are exhausted."""
    producer, evaluator = _wired(
        [AttemptSpec(schema_ok=True, verdict="reject", codes=(RETRYABLE_CODE,))]
    )
    pipeline = ValidationPipeline(
        schema=SeamSchema,
        evaluator=evaluator,
        max_attempts=1,
        retryable_findings=RETRYABLE,
    )

    outcome = await pipeline.validate(producer, context)

    assert outcome.decision is ValidationDecision.ESCALATE
    assert outcome.reason is ValidationReason.ATTEMPTS_EXHAUSTED
    assert len(producer.feedbacks) == 1


async def test_every_outcome_emits_exactly_one_decision_event(
    context: EvaluationContext,
) -> None:
    """Rule 7: one structured validation_decision event per outcome."""
    accept_script = [
        AttemptSpec(schema_ok=True, verdict="reject", codes=(RETRYABLE_CODE,)),
        AttemptSpec(schema_ok=True, verdict="pass"),
    ]
    accept_producer, accept_evaluator = _wired(accept_script)
    escalate_producer = RecordingProducer([AttemptSpec(schema_ok=False)])
    accept_pipeline = ValidationPipeline(
        schema=SeamSchema,
        evaluator=accept_evaluator,
        max_attempts=2,
        retryable_findings=RETRYABLE,
    )
    escalate_pipeline = ValidationPipeline(schema=SeamSchema, evaluator=None, max_attempts=1)

    with captured_decision_events() as events:
        await accept_pipeline.validate(accept_producer, context)
        await escalate_pipeline.validate(escalate_producer, context)

    assert len(events) == 2
    assert events[0] == {
        "seam": SEAM,
        "decision": "accept",
        "reason": "accepted",
        "attempts": 2,
    }
    assert events[1]["decision"] == "escalate"
    assert events[1]["reason"] == "non_retryable_rejection"
    assert events[1]["attempts"] == 1
    assert events[1]["seam"] == SEAM


async def test_accepted_outcome_carries_both_passing_reports(
    context: EvaluationContext,
) -> None:
    producer, evaluator = _wired([AttemptSpec(schema_ok=True, verdict="pass")])
    pipeline = ValidationPipeline(schema=SeamSchema, evaluator=evaluator, max_attempts=1)

    outcome = await pipeline.validate(producer, context)

    assert outcome.decision is ValidationDecision.ACCEPT
    assert outcome.value is not None
    assert outcome.schema_gate.ok is True
    assert outcome.semantic is not None
    assert outcome.semantic.verdict == "pass"


async def test_evaluator_returning_unavailable_verdict_escalates(
    context: EvaluationContext,
) -> None:
    """An evaluator that itself reports unavailability is the same typed state."""
    producer, evaluator = _wired([AttemptSpec(schema_ok=True, verdict="unavailable")])
    pipeline = ValidationPipeline(
        schema=SeamSchema,
        evaluator=evaluator,
        max_attempts=2,
        retryable_findings=RETRYABLE,
    )

    outcome = await pipeline.validate(producer, context)

    assert outcome.decision is ValidationDecision.ESCALATE
    assert outcome.reason is ValidationReason.EVALUATOR_UNAVAILABLE
    assert len(producer.feedbacks) == 1  # unavailability is not a retryable rejection
    assert outcome.semantic is not None
    assert outcome.semantic.verdict == "unavailable"


def test_max_attempts_must_include_the_first_attempt() -> None:
    with pytest.raises(ValueError, match="max_attempts"):
        ValidationPipeline(schema=SeamSchema, evaluator=None, max_attempts=0)


def test_unavailable_report_is_typed_not_a_pass() -> None:
    report = SemanticReport.unavailable("down")
    assert report.findings == (SemanticFinding(code="evaluator_unavailable", message="down"),)
