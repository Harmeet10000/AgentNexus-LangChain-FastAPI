"""The shared post-LLM validation layer.

One typed pipeline every LLM seam runs its output through: schema gate first,
typed semantic evaluation, feedback-carrying bounded retry, budget honesty,
fail-closed evaluator unavailability, caller-routed escalation, and exactly one
decision event per outcome. Infrastructure-independent: ``app/ai`` never imports
``app.features``, so every feature may import this module.
"""

from .escalation import (
    DeadLetterSink,
    EscalationHandler,
    EscalationSink,
    HumanReviewSink,
    TerminalFailureSink,
)
from .evaluators import CompositeEvaluator, LLMEvaluator, MechanicalEvaluator, MechanicalPredicate
from .pipeline import BudgetLedger, SemanticEvaluator, ValidationPipeline
from .schema import (
    EvaluationContext,
    GateReport,
    RetryFeedback,
    SchemaError,
    SemanticFinding,
    SemanticReport,
    SemanticVerdict,
    ValidationDecision,
    ValidationOutcome,
    ValidationReason,
)

__all__ = [
    "BudgetLedger",
    "CompositeEvaluator",
    "DeadLetterSink",
    "EscalationHandler",
    "EscalationSink",
    "EvaluationContext",
    "GateReport",
    "HumanReviewSink",
    "LLMEvaluator",
    "MechanicalEvaluator",
    "MechanicalPredicate",
    "RetryFeedback",
    "SchemaError",
    "SemanticEvaluator",
    "SemanticFinding",
    "SemanticReport",
    "SemanticVerdict",
    "TerminalFailureSink",
    "ValidationDecision",
    "ValidationOutcome",
    "ValidationPipeline",
    "ValidationReason",
]
