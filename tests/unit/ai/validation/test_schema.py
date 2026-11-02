"""Vocabulary tests: the ACCEPT-only value invariant and the closed reason set."""

from __future__ import annotations

import pytest
from conftest import Clause
from pydantic import ValidationError

from app.ai.validation import (
    GateReport,
    SchemaError,
    SemanticFinding,
    SemanticReport,
    ValidationDecision,
    ValidationOutcome,
    ValidationReason,
)


def test_accept_outcome_carries_the_value() -> None:
    value = Clause(text="buyer must pay")
    outcome = ValidationOutcome[Clause](
        decision=ValidationDecision.ACCEPT,
        value=value,
        attempts=1,
        schema_gate=GateReport.passed(value),
        semantic=SemanticReport(verdict="pass"),
        reason=ValidationReason.ACCEPTED,
    )
    assert outcome.value == value
    assert outcome.decision is ValidationDecision.ACCEPT


@pytest.mark.parametrize("decision", [ValidationDecision.RETRY, ValidationDecision.ESCALATE])
def test_non_accept_outcome_cannot_carry_a_value(decision: ValidationDecision) -> None:
    with pytest.raises(ValidationError, match="Only an ACCEPT outcome may carry a value"):
        ValidationOutcome[Clause](
            decision=decision,
            value=Clause(text="buyer must pay"),
            attempts=2,
            schema_gate=GateReport.failed(
                (SchemaError(code="string_too_short", message="too short"),)
            ),
            semantic=SemanticReport(
                verdict="reject",
                findings=(SemanticFinding(code="retry_me", message="nope"),),
            ),
            reason=ValidationReason.NON_RETRYABLE_REJECTION,
        )


def test_accept_outcome_without_a_value_is_invalid() -> None:
    with pytest.raises(ValidationError, match="must carry the validated value"):
        ValidationOutcome[Clause](
            decision=ValidationDecision.ACCEPT,
            value=None,
            attempts=1,
            schema_gate=GateReport.passed(Clause(text="buyer must pay")),
            semantic=SemanticReport(verdict="pass"),
            reason=ValidationReason.ACCEPTED,
        )


def test_outcome_is_frozen() -> None:
    outcome = ValidationOutcome[Clause](
        decision=ValidationDecision.ESCALATE,
        value=None,
        attempts=1,
        schema_gate=GateReport.failed(()),
        semantic=None,
        reason=ValidationReason.EVALUATOR_UNAVAILABLE,
    )
    with pytest.raises(ValidationError):
        outcome.decision = ValidationDecision.ACCEPT  # type: ignore[misc]


def test_reason_vocabulary_is_the_closed_core_set() -> None:
    assert {reason.value for reason in ValidationReason} == {
        "accepted",
        "attempts_exhausted",
        "budget_exhausted",
        "evaluator_unavailable",
        "non_retryable_rejection",
    }


def test_unavailability_is_a_typed_report_not_a_pass() -> None:
    report = SemanticReport.unavailable("grader down")
    assert report.verdict == "unavailable"
    assert report.findings[0].code == "evaluator_unavailable"


def test_schema_errors_carry_typed_codes_and_locators() -> None:
    with pytest.raises(ValidationError) as exc_info:
        Clause.model_validate({"text": "", "kind": "obligation"})
    errors = SchemaError.from_validation_error(exc_info.value)
    assert errors[0].code == "string_too_short"
    assert errors[0].locator == "text"
    assert errors[0].message
