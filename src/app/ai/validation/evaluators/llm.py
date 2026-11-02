"""LLM-backed semantic evaluation over the provider-neutral generation seam.

The evaluator owns a bounded output schema (the grading model may only answer
``pass``/``reject`` with findings), respects the caller's budget via the same
:class:`~app.ai.validation.pipeline.BudgetLedger` port, and maps the generation
seam's typed failures to the ``unavailable`` verdict, which the pipeline fails
closed on (ADR-1). An unexpected raise propagates and the pipeline converts it
to the same typed state.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, ConfigDict, Field
from returns.result import Failure

from app.ai.generation import TextGenerationRequest
from app.observability.logging import logger

from ..schema import SemanticFinding, SemanticReport

if TYPE_CHECKING:
    from app.ai.generation import TextGenerator

    from ..pipeline import BudgetLedger
    from ..schema import EvaluationContext


class LLMEvaluationOutput(BaseModel):
    """The bounded schema the grading model must answer in."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    verdict: Literal["pass", "reject"]
    findings: tuple[SemanticFinding, ...] = ()
    score: float | None = Field(default=None, ge=0.0, le=1.0)


class LLMEvaluator[T: BaseModel]:
    """Wraps a :class:`TextGenerator` as a :class:`SemanticEvaluator`.

    ``instruction`` is the seam's grading prompt; the value under evaluation is
    appended as JSON. ``budget``, when supplied, is charged once per model call
    — the evaluator respects the caller's budget instead of minting its own.
    """

    def __init__(
        self,
        generator: TextGenerator,
        *,
        instruction: str,
        budget: BudgetLedger | None = None,
        temperature: float | None = None,
    ) -> None:
        self._generator = generator
        self._instruction = instruction
        self._budget = budget
        self._temperature = temperature

    async def evaluate(self, value: T, context: EvaluationContext) -> SemanticReport:
        if self._budget is not None and not self._budget.charge():
            return SemanticReport.unavailable("Evaluator budget exhausted before the model call")
        request = TextGenerationRequest(
            prompt=(
                f"{self._instruction}\n\n"
                f"Evaluation seam: {context.seam}\n\n"
                f"Output under evaluation:\n{value.model_dump_json()}"
            ),
            response_schema=LLMEvaluationOutput,
            temperature=self._temperature,
        )
        result = await self._generator.generate_structured(request)
        if isinstance(result, Failure):
            return SemanticReport.unavailable(f"LLM evaluation failed: {result.failure().message}")
        output = result.unwrap().output
        if not isinstance(output, LLMEvaluationOutput):
            return SemanticReport.unavailable("LLM evaluator received an unexpected output schema")
        if output.verdict == "pass":
            return SemanticReport(verdict="pass", score=output.score)
        logger.bind(
            seam=context.seam,
            finding_codes=[finding.code for finding in output.findings],
        ).debug("llm_evaluation_rejection")
        return SemanticReport(verdict="reject", findings=output.findings, score=output.score)
