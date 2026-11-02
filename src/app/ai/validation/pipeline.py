"""The one typed pipeline every LLM seam runs its output through.

Schema gate first, semantic evaluator second, bounded feedback-carrying retry,
budget honesty, fail-closed unavailability, one decision event per outcome.
The seven contract rules (PLAN-result-validation-layer.md §3.2) are hard
requirements; each has a named test in ``tests/unit/ai/validation/``.

Infrastructure independence: this module imports only stdlib, Pydantic and the
observability logger — never ``app.features``. The budget is a port
(:class:`BudgetLedger`) callers adapt their ledgers to; the pipeline never
constructs or extends a ledger (ADR-3).
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from pydantic import BaseModel, ValidationError

from app.observability.logging import logger

from .schema import (
    GateReport,
    RetryFeedback,
    SchemaError,
    SemanticReport,
    ValidationDecision,
    ValidationOutcome,
    ValidationReason,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from .schema import (
        EvaluationContext,
    )

type ProduceFn = Callable[[RetryFeedback | None], Awaitable[object]]


@runtime_checkable
class BudgetLedger(Protocol):
    """Port for spend accounting, mirrored on the existing ledger shape.

    ``charge`` asks the ledger to accept one unit of spend. It returns ``True``
    when the charge was accepted and ``False`` when the budget is exhausted —
    exhaustion is an expected answer, not an exception, so adapters stay trivial.
    The pipeline only charges; it never creates, resets or extends a ledger.
    """

    def charge(self) -> bool: ...


class SemanticEvaluator[T: BaseModel](Protocol):
    """The semantic gate as a port: async, typed, one report per call.

    Implementations may raise; the pipeline converts any raise into the typed
    ``unavailable`` verdict and fails closed (ESCALATE), never a silent pass.
    """

    async def evaluate(self, value: T, context: EvaluationContext) -> SemanticReport: ...


class ValidationPipeline[T: BaseModel]:
    """Runs ``produce`` under the schema and semantic gates with bounded retries.

    ``max_attempts`` includes the first attempt. ``retryable_findings`` holds the
    finding codes (semantic codes and schema-error codes alike) worth a retry —
    a rejection with no such code escalates immediately regardless of attempts
    remaining. ``budget`` is optional; when supplied, every re-invocation of
    ``produce`` charges it.

    ``validate`` returns ACCEPT or ESCALATE. RETRY is an internal decision: it
    surfaces to the caller as the :class:`RetryFeedback` the next ``produce``
    invocation receives, never as a returned outcome.
    """

    def __init__(
        self,
        schema: type[T],
        evaluator: SemanticEvaluator[T] | None,
        *,
        max_attempts: int,
        retryable_findings: Sequence[str] = (),
        budget: BudgetLedger | None = None,
    ) -> None:
        if max_attempts < 1:
            msg = "max_attempts includes the first attempt, so it must be >= 1"
            raise ValueError(msg)
        self._schema = schema
        self._evaluator = evaluator
        self._max_attempts = max_attempts
        self._retryable_findings = frozenset(retryable_findings)
        self._budget = budget

    async def validate(
        self,
        produce: ProduceFn,
        context: EvaluationContext,
    ) -> ValidationOutcome[T]:
        """Produce, gate, and retry until a terminal decision is reached."""
        attempts = 0
        feedback: RetryFeedback | None = None
        gate = GateReport.failed(())
        semantic: SemanticReport | None = None
        value: T | None = None
        decision = ValidationDecision.ESCALATE
        reason = ValidationReason.ATTEMPTS_EXHAUSTED

        while attempts < self._max_attempts:
            raw = await produce(feedback)
            attempts += 1

            gate = self._run_schema_gate(raw)
            semantic = None

            if gate.ok:
                parsed_value = gate.value
                if not isinstance(parsed_value, self._schema):
                    # Defensive: the gate must hand the pipeline a schema instance.
                    gate = GateReport.failed(
                        (
                            SchemaError(
                                code="schema_gate_defect",
                                message="Schema gate did not produce the declared type",
                            ),
                        )
                    )
                else:
                    value = parsed_value
                    semantic = await self._run_evaluator(value, context)
                    if semantic.verdict == "unavailable":
                        decision, reason = (
                            ValidationDecision.ESCALATE,
                            ValidationReason.EVALUATOR_UNAVAILABLE,
                        )
                        break
                    if semantic.verdict == "pass":
                        decision, reason = (
                            ValidationDecision.ACCEPT,
                            ValidationReason.ACCEPTED,
                        )
                        break

            # The gate failed (schema errors or a semantic rejection): apply the
            # retry rules. A retry needs attempts remaining AND a retryable code.
            failed_codes = [e.code for e in gate.errors] + [
                f.code for f in (semantic.findings if semantic is not None else ())
            ]
            if attempts < self._max_attempts and self._retryable_findings & set(failed_codes):
                if self._budget is not None and not self._budget.charge():
                    decision, reason = (
                        ValidationDecision.ESCALATE,
                        ValidationReason.BUDGET_EXHAUSTED,
                    )
                    break
                feedback = RetryFeedback(
                    attempt=attempts,
                    schema_errors=gate.errors,
                    semantic_findings=(semantic.findings if semantic is not None else ()),
                )
                continue

            decision = ValidationDecision.ESCALATE
            reason = (
                ValidationReason.NON_RETRYABLE_REJECTION
                if not self._retryable_findings & set(failed_codes)
                else ValidationReason.ATTEMPTS_EXHAUSTED
            )
            break

        outcome = ValidationOutcome[T](
            decision=decision,
            value=value if decision is ValidationDecision.ACCEPT else None,
            attempts=attempts,
            schema_gate=gate,
            semantic=semantic,
            reason=reason,
        )
        logger.bind(
            seam=context.seam,
            decision=outcome.decision.value,
            reason=outcome.reason.value,
            attempts=outcome.attempts,
        ).info("validation_decision")
        return outcome

    def _run_schema_gate(self, raw: object) -> GateReport:
        """Parse the produced value against the seam's schema.

        Coercion happens only where the schema itself declares it (Pydantic's
        lax-to-strict settings live on the model, not here).
        """
        try:
            return GateReport.passed(self._schema.model_validate(raw))
        except ValidationError as error:
            return GateReport.failed(SchemaError.from_validation_error(error))

    async def _run_evaluator(self, value: T, context: EvaluationContext) -> SemanticReport:
        """Run the semantic evaluator, converting every failure to run into ``unavailable``.

        An unconfigured evaluator and a raising evaluator are the same typed
        state (ADR-1): fail closed with ESCALATE, never a vacuous pass.
        """
        if self._evaluator is None:
            return SemanticReport.unavailable("No semantic evaluator is configured for this seam")
        try:
            return await self._evaluator.evaluate(value, context)
        except Exception as error:  # noqa: BLE001 — the contract requires any raise to fail closed
            return SemanticReport.unavailable(f"Semantic evaluator raised: {error!r}")
