"""Evaluator implementation tests: mechanical short-circuit, composite ordering, LLM mapping."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from conftest import Clause, RecordingBudget
from returns.result import Failure, Success

from app.ai.generation import (
    GeneratedStructured,
    InMemoryTextGenerator,
    TextGenerationUnavailableError,
)
from app.ai.validation import (
    CompositeEvaluator,
    LLMEvaluator,
    MechanicalEvaluator,
    SemanticFinding,
    SemanticReport,
)
from app.ai.validation.evaluators.llm import LLMEvaluationOutput

if TYPE_CHECKING:
    from collections.abc import Callable

    from app.ai.validation import (
        EvaluationContext,
    )


def _finding(code: str) -> SemanticFinding:
    return SemanticFinding(code=code, message=f"failed: {code}")


def _predicate(
    finding: SemanticFinding | None,
    calls: list[bool],
) -> Callable[[Clause, EvaluationContext], SemanticFinding | None]:
    def _check(value: Clause, context: EvaluationContext) -> SemanticFinding | None:
        calls.append(True)
        return finding

    return _check


class RecordingLLMEvaluator:
    """A spy evaluator standing in for the model-backed one."""

    def __init__(self, report: SemanticReport) -> None:
        self.report = report
        self.calls = 0

    async def evaluate(self, value: Clause, context: EvaluationContext) -> SemanticReport:
        self.calls += 1
        return self.report


def _generator_with(
    output: LLMEvaluationOutput,
) -> InMemoryTextGenerator:
    return InMemoryTextGenerator(structured_results=[Success(GeneratedStructured(output=output))])


async def test_mechanical_rejection_never_costs_a_model_call(
    context: EvaluationContext,
) -> None:
    """A mechanical finding short-circuits the model call."""
    mechanical_calls: list[bool] = []
    mechanical = MechanicalEvaluator[Clause](
        [_predicate(_finding("span_mismatch"), mechanical_calls)]
    )
    llm = RecordingLLMEvaluator(SemanticReport(verdict="pass"))
    composite = CompositeEvaluator(mechanical=mechanical, llm=llm)

    report = await composite.evaluate(Clause(text="clause"), context)

    assert mechanical_calls == [True]
    assert llm.calls == 0  # the model evaluator was NOT invoked
    assert report.verdict == "reject"
    assert report.findings == (_finding("span_mismatch"),)


async def test_composite_consults_the_llm_only_on_mechanical_pass(
    context: EvaluationContext,
) -> None:
    """Composite ordering: mechanics first, model second, reject needs no model."""
    mechanical_calls: list[bool] = []
    mechanical = MechanicalEvaluator[Clause]([_predicate(None, mechanical_calls)])
    llm = RecordingLLMEvaluator(SemanticReport(verdict="reject", findings=(_finding("grounding"),)))
    composite = CompositeEvaluator(mechanical=mechanical, llm=llm)

    report = await composite.evaluate(Clause(text="clause"), context)

    assert mechanical_calls == [True]
    assert llm.calls == 1  # consulted exactly once, after the mechanical pass
    assert report.verdict == "reject"
    assert report.findings == (_finding("grounding"),)


async def test_composite_passes_when_both_gates_pass(
    context: EvaluationContext,
) -> None:
    mechanical = MechanicalEvaluator[Clause]([_predicate(None, [])])
    llm = RecordingLLMEvaluator(SemanticReport(verdict="pass", score=0.9))
    composite = CompositeEvaluator(mechanical=mechanical, llm=llm)

    report = await composite.evaluate(Clause(text="clause"), context)

    assert report.verdict == "pass"
    assert report.score == pytest.approx(0.9)


async def test_composite_without_llm_fails_closed_on_mechanical_pass(
    context: EvaluationContext,
) -> None:
    mechanical = MechanicalEvaluator[Clause]([_predicate(None, [])])
    composite = CompositeEvaluator(mechanical=mechanical, llm=None)

    report = await composite.evaluate(Clause(text="clause"), context)

    assert report.verdict == "unavailable"


async def test_llm_evaluator_maps_structured_output_to_reports(
    context: EvaluationContext,
) -> None:
    generator = _generator_with(
        LLMEvaluationOutput(verdict="reject", findings=(_finding("grounding"),), score=0.25)
    )
    evaluator = LLMEvaluator[Clause](generator, instruction="grade the clause")

    report = await evaluator.evaluate(Clause(text="clause"), context)

    assert report.verdict == "reject"
    assert report.findings == (_finding("grounding"),)
    assert report.score == pytest.approx(0.25)
    request = generator.calls[0]
    assert request.response_schema is LLMEvaluationOutput
    assert "clause" in request.prompt


async def test_llm_evaluator_pass_output_carries_score(
    context: EvaluationContext,
) -> None:
    generator = _generator_with(LLMEvaluationOutput(verdict="pass", score=1.0))
    evaluator = LLMEvaluator[Clause](generator, instruction="grade the clause")

    report = await evaluator.evaluate(Clause(text="clause"), context)

    assert report.verdict == "pass"
    assert report.score == pytest.approx(1.0)


async def test_llm_evaluator_failure_is_unavailable_never_a_pass(
    context: EvaluationContext,
) -> None:
    generator = InMemoryTextGenerator()  # no script: the adapter fails unavailable
    evaluator = LLMEvaluator[Clause](generator, instruction="grade the clause")

    report = await evaluator.evaluate(Clause(text="clause"), context)

    assert report.verdict == "unavailable"
    assert report.findings[0].code == "evaluator_unavailable"


async def test_llm_evaluator_charges_its_budget_per_model_call(
    context: EvaluationContext,
) -> None:
    budget = RecordingBudget(capacity=1)
    generator = _generator_with(LLMEvaluationOutput(verdict="pass", score=1.0))
    evaluator = LLMEvaluator[Clause](generator, instruction="grade", budget=budget)

    first = await evaluator.evaluate(Clause(text="clause"), context)
    second = await evaluator.evaluate(Clause(text="clause"), context)

    assert first.verdict == "pass"
    assert budget.charges == 1
    assert second.verdict == "unavailable"  # the second call cannot be paid for


async def test_typed_generation_failure_maps_to_unavailable_report(
    context: EvaluationContext,
) -> None:
    generator = InMemoryTextGenerator(
        structured_results=[Failure(TextGenerationUnavailableError())]
    )
    evaluator = LLMEvaluator[Clause](generator, instruction="grade the clause")

    report = await evaluator.evaluate(Clause(text="clause"), context)

    assert report.verdict == "unavailable"
