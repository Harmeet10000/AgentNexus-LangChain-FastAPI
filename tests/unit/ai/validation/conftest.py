"""Shared test doubles for the validation contract-rule suite.

Everything here is scripted and recording: the producer replays payloads and
records the feedback each call received, the evaluator replays verdicts, and
the budget grants a fixed number of charges and then stays exhausted forever —
never minting a fresh allowance.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import TYPE_CHECKING, NamedTuple

import pytest
from loguru import logger as loguru_logger
from pydantic import BaseModel, Field

from app.ai.validation import EvaluationContext, SemanticFinding, SemanticReport

if TYPE_CHECKING:
    from collections.abc import Iterator
    from typing import Any

SEAM = "test_seam"
SCHEMA_ERROR_CODE = "string_too_short"
RETRYABLE_CODE = "retry_me"
FATAL_CODE = "forged_identity"


class Clause(BaseModel):
    """The seam schema used throughout the suite."""

    text: str = Field(min_length=1)
    kind: str = "obligation"


class AttemptSpec(NamedTuple):
    """Scripted behavior for one attempt."""

    schema_ok: bool
    # "pass" | "reject" | "unavailable" | "raise"; only consulted when schema_ok.
    verdict: str = "pass"
    codes: tuple[str, ...] = ()


class RecordingProducer:
    """Replays scripted payloads and records the feedback each call received."""

    def __init__(self, script: list[AttemptSpec]) -> None:
        self.script = script
        self.feedbacks: list[Any] = []

    async def __call__(self, feedback: Any) -> object:
        self.feedbacks.append(feedback)
        spec = self.script[min(len(self.feedbacks) - 1, len(self.script) - 1)]
        if spec.schema_ok:
            return {"text": "a clause", "kind": "obligation"}
        return {"text": "", "kind": "obligation"}  # fails the min_length gate


class ScriptedEvaluator:
    """Replays scripted semantic verdicts, indexed by attempt number.

    The evaluator reads the attempt off the producer's call count: the pipeline
    invokes them back-to-back within one loop iteration.
    """

    def __init__(self, script: list[AttemptSpec], producer: RecordingProducer) -> None:
        self.script = script
        self.producer = producer
        self.values_seen: list[Any] = []

    async def evaluate(self, value: Any, context: EvaluationContext) -> SemanticReport:
        attempt = len(self.producer.feedbacks)
        self.values_seen.append(value)
        spec = self.script[min(attempt - 1, len(self.script) - 1)]
        if spec.verdict == "raise":
            msg = "grader down"
            raise RuntimeError(msg)
        if spec.verdict == "unavailable":
            return SemanticReport(verdict="unavailable")
        if spec.verdict == "reject":
            return SemanticReport(
                verdict="reject",
                findings=tuple(
                    SemanticFinding(code=code, message=f"failed: {code}") for code in spec.codes
                ),
            )
        return SemanticReport(verdict="pass")


class RecordingBudget:
    """Grants ``capacity`` charges; once exhausted it stays exhausted."""

    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self.charges = 0

    def charge(self) -> bool:
        if self.charges >= self.capacity:
            return False
        self.charges += 1
        return True


@pytest.fixture
def context() -> EvaluationContext:
    return EvaluationContext(seam=SEAM)


@contextmanager
def captured_decision_events() -> Iterator[list[dict[str, Any]]]:
    """Capture ``validation_decision`` events emitted while the block runs."""
    events: list[dict[str, Any]] = []

    def _sink(message: Any) -> None:
        record = message.record
        if record["message"] == "validation_decision":
            events.append(dict(record["extra"]))

    sink_id = loguru_logger.add(_sink, level="DEBUG", format="{message}")
    try:
        yield events
    finally:
        loguru_logger.remove(sink_id)
