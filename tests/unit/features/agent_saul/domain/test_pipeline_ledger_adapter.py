"""D7 task 3.1: Saul's ledger conforms to the pipeline's budget port.

The adapter is a translation, not a replacement: recorded charges equal the
actual producer re-invocations, exhaustion mid-pipeline yields ESCALATE with
reason ``budget_exhausted``, and no fresh allowance is ever minted.
"""

from typing import Any

import pytest
from pydantic import BaseModel

from app.ai.validation import (
    BudgetLedger as BudgetLedgerPort,
)
from app.ai.validation import (
    EvaluationContext,
    MechanicalEvaluator,
    SemanticFinding,
    ValidationDecision,
    ValidationPipeline,
    ValidationReason,
)
from app.features.agent_saul.domain.budget import (
    BudgetDimension,
    BudgetLimits,
    PipelineLedgerAdapter,
    new_budget_ledger,
)

pytestmark = pytest.mark.unit

_SEAM = "agent_saul.test"


class _SeamValue(BaseModel):
    text: str


def _rejecting_evaluator() -> MechanicalEvaluator[_SeamValue]:
    return MechanicalEvaluator(
        predicates=[lambda _value, _context: SemanticFinding(code="retry_me", message="reject")]
    )


async def test_adapter_satisfies_the_pipeline_budget_port() -> None:
    adapter = PipelineLedgerAdapter(new_budget_ledger())
    assert isinstance(adapter, BudgetLedgerPort)


async def test_recorded_charges_equal_actual_producer_reinvocations() -> None:
    """Charge arithmetic: one accepted charge per re-invocation, none for the first."""
    producer_calls = 0

    async def produce(_feedback: Any) -> dict[str, Any]:
        nonlocal producer_calls
        producer_calls += 1
        return {"text": "value"}

    adapter = PipelineLedgerAdapter(
        new_budget_ledger(BudgetLimits(model_calls=10)),
        dimension=BudgetDimension.MODEL_CALLS,
    )
    pipeline = ValidationPipeline(
        schema=_SeamValue,
        evaluator=_rejecting_evaluator(),
        max_attempts=3,
        retryable_findings=("retry_me",),
        budget=adapter,
    )

    outcome = await pipeline.validate(produce, EvaluationContext(seam=_SEAM))

    assert outcome.decision is ValidationDecision.ESCALATE
    assert outcome.reason is ValidationReason.ATTEMPTS_EXHAUSTED
    assert outcome.attempts == 3
    # Three produce calls: the first attempt is free, each re-invocation
    # charged exactly once.
    assert producer_calls == 3
    assert adapter.charges == 2
    assert adapter.ledger.model_calls == 2


async def test_exhaustion_mid_pipeline_yields_budget_exhausted_escalation() -> None:
    produce_calls = 0

    async def produce(_feedback: Any) -> dict[str, Any]:
        nonlocal produce_calls
        produce_calls += 1
        return {"text": "value"}

    adapter = PipelineLedgerAdapter(
        # Zero allowance: the first retry charge is refused.
        new_budget_ledger(BudgetLimits(model_calls=0)),
        dimension=BudgetDimension.MODEL_CALLS,
    )
    pipeline = ValidationPipeline(
        schema=_SeamValue,
        evaluator=_rejecting_evaluator(),
        max_attempts=3,
        retryable_findings=("retry_me",),
        budget=adapter,
    )

    outcome = await pipeline.validate(produce, EvaluationContext(seam=_SEAM))

    assert outcome.decision is ValidationDecision.ESCALATE
    assert outcome.reason is ValidationReason.BUDGET_EXHAUSTED
    assert produce_calls == 1
    assert adapter.charges == 0
    assert adapter.ledger.exhausted is True
    assert adapter.ledger.exhausted_dimension is BudgetDimension.MODEL_CALLS


async def test_no_fresh_allowance_after_exhaustion() -> None:
    adapter = PipelineLedgerAdapter(
        new_budget_ledger(BudgetLimits(model_calls=1)),
        dimension=BudgetDimension.MODEL_CALLS,
    )

    assert adapter.charge() is True
    assert adapter.charge() is False
    # Once exhausted, always exhausted: no reset, no fresh pool, and the
    # underlying usage counters never move backwards.
    assert adapter.charge() is False
    assert adapter.charges == 1
    assert adapter.ledger.model_calls == 1
    assert adapter.ledger.exhausted is True
