"""The shared decision vocabulary every validated LLM seam speaks.

One vocabulary replaces the per-seam accept/retry/escalate shapes (the ask loop's
``ContextGrade``, the relationship pass's drop reasons, extraction's failure codes).
The vocabulary is infrastructure-independent: ``app/ai`` may not import
``app.features``, so everything here is stdlib plus Pydantic v2.

The invariant that matters most is enforced at the model level, not by convention:
a :class:`ValidationOutcome` carries its value if and only if its decision is
:meth:`ACCEPT <ValidationDecision.ACCEPT>` (see ADR-2 of the change design).
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

type SemanticVerdict = Literal["pass", "reject", "unavailable"]


class ValidationDecision(StrEnum):
    """The one decision set every seam outcome carries."""

    ACCEPT = "accept"  # schema + semantic gates passed
    RETRY = "retry"  # retryable rejection with attempts remaining (internal to the loop)
    ESCALATE = "escalate"  # retries exhausted, non-retryable rejection, or unavailable gate


class ValidationReason(StrEnum):
    """The closed, stable core reason vocabulary.

    Seam-specific detail belongs in finding codes (see :class:`SemanticFinding`),
    not here — an open reason enum is exactly how three vocabularies grew last time.
    """

    ACCEPTED = "accepted"
    ATTEMPTS_EXHAUSTED = "attempts_exhausted"
    BUDGET_EXHAUSTED = "budget_exhausted"
    EVALUATOR_UNAVAILABLE = "evaluator_unavailable"
    NON_RETRYABLE_REJECTION = "non_retryable_rejection"


class SchemaError(BaseModel):
    """One typed schema-gate failure: a Pydantic error type or a custom code."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    code: str = Field(min_length=1)
    message: str
    locator: str | None = None

    @classmethod
    def from_validation_error(cls, error: ValidationError) -> tuple[SchemaError, ...]:
        """Convert Pydantic's error list into the typed vocabulary.

        Locators are dotted field paths; the input payload is deliberately not
        carried (it can be arbitrarily large and the producer already has it).
        """
        return tuple(
            cls(
                code=exc["type"],
                message=exc["msg"],
                locator=".".join(str(part) for part in exc["loc"]) or None,
            )
            for exc in error.errors()
        )


class SemanticFinding(BaseModel):
    """One semantic problem with a seam-scoped code and an optional locator."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    code: str = Field(min_length=1)
    message: str
    locator: str | None = None


class GateReport(BaseModel):
    """The schema gate's report for one attempt."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    ok: bool
    errors: tuple[SchemaError, ...] = ()
    # The parsed value when ok; the raw payload is the producer's business.
    value: Any | None = None

    @classmethod
    def passed(cls, value: Any) -> GateReport:
        return cls(ok=True, errors=(), value=value)

    @classmethod
    def failed(cls, errors: tuple[SchemaError, ...]) -> GateReport:
        return cls(ok=False, errors=errors, value=None)


class SemanticReport(BaseModel):
    """The semantic gate's report for one attempt.

    ``score`` is evaluator-provided and never fabricated: an evaluator that has
    no score simply leaves it ``None``.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    verdict: SemanticVerdict
    findings: tuple[SemanticFinding, ...] = ()
    score: float | None = None

    @classmethod
    def unavailable(cls, message: str) -> SemanticReport:
        """The typed unavailability state (ADR-1): never a silent pass."""
        return cls(
            verdict="unavailable",
            findings=(SemanticFinding(code="evaluator_unavailable", message=message),),
        )


class EvaluationContext(BaseModel):
    """Per-seam evaluation context.

    ``seam`` names the seam for the one ``validation_decision`` event; seam-specific
    data (document ids, boundary text ids, ...) rides as extra attributes so this
    module never grows a field per consumer.
    """

    model_config = ConfigDict(extra="allow", frozen=True)

    seam: str = Field(min_length=1)


class RetryFeedback(BaseModel):
    """Structured feedback handed back to the producer on a RETRY.

    A retry without this is a contract violation of the pipeline, not a caller
    choice — the second attempt must know why the first failed.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    attempt: int = Field(ge=1)
    schema_errors: tuple[SchemaError, ...] = ()
    semantic_findings: tuple[SemanticFinding, ...] = ()


class ValidationOutcome[T: BaseModel](BaseModel):
    """The frozen, generic result of one validation run.

    ``value`` is present if and only if ``decision`` is ACCEPT, enforced by a
    model validator rather than a convention. Outcomes can be persisted next to
    a seam's record without mutation risk.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    decision: ValidationDecision
    value: T | None = None
    attempts: int = Field(ge=1)
    schema_gate: GateReport
    semantic: SemanticReport | None = None
    reason: ValidationReason

    @model_validator(mode="after")
    def _value_iff_accept(self) -> ValidationOutcome[Any]:
        if self.decision is ValidationDecision.ACCEPT and self.value is None:
            msg = "An ACCEPT outcome must carry the validated value"
            raise ValueError(msg)
        if self.decision is not ValidationDecision.ACCEPT and self.value is not None:
            msg = "Only an ACCEPT outcome may carry a value"
            raise ValueError(msg)
        return self
