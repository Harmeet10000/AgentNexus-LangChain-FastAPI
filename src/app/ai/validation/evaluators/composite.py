"""Composed evaluation: mechanical predicates first, the model only on pass (ADR-5).

A mechanical rejection short-circuits the model call — the happy path gains zero
model calls, and a mechanical finding never waits on a grader.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import BaseModel

from ..schema import SemanticReport

if TYPE_CHECKING:
    from ..pipeline import SemanticEvaluator
    from ..schema import EvaluationContext
    from .mechanical import MechanicalEvaluator


class CompositeEvaluator[T: BaseModel]:
    """Runs the mechanical evaluator first; consults the LLM evaluator only on pass.

    A missing mechanical evaluator is a vacuous pass; a missing LLM evaluator
    behind a mechanical pass is the typed ``unavailable`` verdict (fail-closed),
    because the composite was asked for a semantic verdict it cannot produce.
    """

    def __init__(
        self,
        mechanical: MechanicalEvaluator[T] | None,
        llm: SemanticEvaluator[T] | None,
    ) -> None:
        self._mechanical = mechanical
        self._llm = llm

    async def evaluate(self, value: T, context: EvaluationContext) -> SemanticReport:
        if self._mechanical is not None:
            mechanical_report = await self._mechanical.evaluate(value, context)
            if mechanical_report.verdict != "pass":
                return mechanical_report
        if self._llm is None:
            return SemanticReport.unavailable(
                "CompositeEvaluator has no LLM evaluator for the semantic pass"
            )
        return await self._llm.evaluate(value, context)
