"""Property tests for the pipeline's two invariants (tasks.md 2.2).

Over arbitrary scripted scenarios:

1. No path returns ACCEPT without both gates passing.
2. No path returns RETRY without attempts remaining and a retryable finding —
   every actual re-invocation of the producer is preceded by a retryable
   rejection (schema or semantic) with attempts and budget remaining.
"""

from __future__ import annotations

import asyncio

from conftest import (
    SCHEMA_ERROR_CODE,
    SEAM,
    AttemptSpec,
    Clause,
    RecordingBudget,
    RecordingProducer,
    ScriptedEvaluator,
)
from hypothesis import given, settings
from hypothesis import strategies as st

from app.ai.validation import (
    EvaluationContext,
    RetryFeedback,
    ValidationDecision,
    ValidationPipeline,
    ValidationReason,
)

RETRYABLE_CODES = ("retry_me", SCHEMA_ERROR_CODE)
NON_RETRYABLE_CODES = ("forged_identity", "grounding_failure")
ALL_CODES = RETRYABLE_CODES + NON_RETRYABLE_CODES

verdict_strategy = st.sampled_from(["pass", "reject", "unavailable", "raise"])
code_strategy = st.sampled_from(ALL_CODES)
codes_strategy = st.lists(code_strategy, max_size=2, unique=True).map(tuple)

attempt_strategy = st.fixed_dictionaries(
    {"schema_ok": st.booleans(), "verdict": verdict_strategy, "codes": codes_strategy}
).map(lambda fields: AttemptSpec(**fields))

script_strategy = st.lists(attempt_strategy, min_size=1, max_size=4)


def _retry_codes(spec: AttemptSpec) -> set[str]:
    """The failure codes an attempt exposes to the retry rules."""
    if not spec.schema_ok:
        return {SCHEMA_ERROR_CODE}
    if spec.verdict == "reject":
        return set(spec.codes)
    return set()


@given(
    script=script_strategy,
    max_attempts=st.integers(min_value=1, max_value=5),
    retryable=st.sets(st.sampled_from(RETRYABLE_CODES), max_size=2),
    budget_capacity=st.one_of(st.none(), st.integers(min_value=0, max_value=4)),
)
@settings(max_examples=100, deadline=None)
def test_gate_and_retry_invariants_hold_on_every_path(
    script: list[AttemptSpec],
    max_attempts: int,
    retryable: set[str],
    budget_capacity: int | None,
) -> None:
    producer = RecordingProducer(script)
    evaluator = ScriptedEvaluator(script, producer)
    budget = RecordingBudget(capacity=budget_capacity) if budget_capacity is not None else None
    pipeline: ValidationPipeline[Clause] = ValidationPipeline(
        schema=Clause,
        evaluator=evaluator,
        max_attempts=max_attempts,
        retryable_findings=frozenset(retryable),
        budget=budget,
    )
    context = EvaluationContext(seam=SEAM)

    outcome = asyncio.run(pipeline.validate(producer, context))

    # Invariant 1: no path returns ACCEPT without both gates passing.
    if outcome.decision is ValidationDecision.ACCEPT:
        assert outcome.schema_gate.ok is True
        assert outcome.semantic is not None
        assert outcome.semantic.verdict == "pass"
        assert outcome.value is not None
        assert outcome.reason is ValidationReason.ACCEPTED
    else:
        assert outcome.decision is ValidationDecision.ESCALATE
        assert outcome.value is None

    # The loop never surfaces RETRY as a returned outcome; retries are internal.
    assert outcome.decision is not ValidationDecision.RETRY

    # Invariant 2: every re-invocation required attempts remaining, a retryable
    # finding, and an accepted budget charge.
    retries = len(producer.feedbacks) - 1
    assert len(producer.feedbacks) <= max_attempts
    for index in range(retries):
        previous = script[min(index, len(script) - 1)]
        previous_codes = _retry_codes(previous)
        assert previous_codes & retryable, (
            f"retry {index + 1} without a retryable finding (previous codes: {previous_codes})"
        )
        assert index + 1 < max_attempts
        if budget is not None:
            assert budget_capacity is not None
            assert index < budget_capacity  # the charge was accepted, never faked
    if budget is not None:
        assert budget.charges == retries

    # Feedback always carries the attempt number it retries from.
    for index, feedback in enumerate(producer.feedbacks[1:], start=1):
        assert isinstance(feedback, RetryFeedback)
        assert feedback.attempt == index
