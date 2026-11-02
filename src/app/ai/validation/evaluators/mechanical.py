"""Pure, model-free evaluation: composed predicates, zero cost per call (ADR-5).

Existing seam validators (relationship span checks, quote containment, enum
membership) become predicate callables at adoption time — the vocabulary maps,
the code stays where it is.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

from pydantic import BaseModel

from ..schema import EvaluationContext, SemanticFinding, SemanticReport

if TYPE_CHECKING:
    from collections.abc import Sequence

type MechanicalPredicate[T: BaseModel] = Callable[[T, EvaluationContext], SemanticFinding | None]


class MechanicalEvaluator[T: BaseModel]:
    """Runs pure predicates in order; any finding rejects, otherwise pass.

    No model calls, ever — this evaluator is free on the happy path and gives
    :class:`~app.ai.validation.evaluators.composite.CompositeEvaluator` its
    short-circuit gate.
    """

    def __init__(self, predicates: Sequence[MechanicalPredicate[T]]) -> None:
        self._predicates = tuple(predicates)

    async def evaluate(self, value: T, context: EvaluationContext) -> SemanticReport:
        findings = tuple(
            finding
            for finding in (predicate(value, context) for predicate in self._predicates)
            if finding is not None
        )
        if findings:
            return SemanticReport(verdict="reject", findings=findings)
        return SemanticReport(verdict="pass")
