# PLAN: Result Validation Layer (Post-LLM)

Normative reference for introducing a unified, typed post-LLM validation pipeline:

```
LLM output
  → Schema validation (pydantic gate)
  → Semantic validation (Evaluator Node)
  → Decision: ACCEPT / RETRY / ESCALATE
```

**Status of this plan:** the *pieces* of this pipeline exist in the codebase, fragmented
per feature. The *unified layer* does not exist. This document maps what exists, what is
missing, and specifies the module to build. Where the plan conflicts with
`.opencode/instructions/`, the instructions win.

---

## 1. What already exists (do not rebuild — integrate or wrap)

| Stage | Existing implementation | Seam |
|---|---|---|
| Schema validation | `with_structured_output(schema)` + pydantic models (`LegalMetadataExtraction`), `GeneratedAnswer.model_validate` (documents/retrieval/graph/state.py:81), `QueryPlan` validation | legal-metadata extraction; ask path |
| Schema validation | `_validate_relationship_extractions` (span/clause/vocabulary checks, drop-counted, `RelationshipDropReason`) + `GroundedRelationship` model (`extraction/langextract/relationships.py`) | relationship pass-2 |
| Schema validation | `Citation` model with evidence-contract fields, `_validate_canonical_segments` | agent_saul |
| Semantic validation | `_grade_context` → `ContextGrade(sufficient, missing_aspects, rewrite_suggestion)` via a `ContextGrader` protocol (documents/application/queries/documents.py:1193, 350) — **this is an Evaluator Node already** | documents ask |
| Semantic validation | `_verify_quotes_against_canonical_source` (quote re-read + entailment), mechanical evidence-handle gates before model entailment (agent_saul/graph/nodes.py:1842, 2024, 2081) | agent_saul verdicts |
| Retry | Grade loop: `grade.sufficient` else `rewrite_suggestion` → retry once → `_FALLBACK_ANSWER` (documents.py:1000) | documents ask |
| Retry | `ModelRetryMiddleware` + `ToolRetryMiddleware` + HITL middleware (`app/ai/guardrails.py`) | langchain agents |
| Retry | Cumulative `ExtractionBudget` ledger (checkpoint-safe, per-share pools) driving Celery autoretry; `dont_autoretry_for` for terminal classes (worker/jobs/search.py) | extraction; search ingest |
| Escalate | HITL pause (`make_human_review_node`, agent_saul/graph/nodes.py:2335); terminal ingestion events + dead-letter; relationship state `failed` → document retry via checkpoint | agent_saul; documents |

## 2. What is missing (the actual gap)

1. **No shared decision vocabulary.** Every seam invents its own
   accept/retry/escalate shape (`ContextGrade.sufficient`, `GraphRelationshipsState`,
   `ExtractionFailed` codes, agent_saul review flags).
2. **No reusable pipeline.** Each new LLM seam re-implements validation from scratch;
   the last three features (search, relationships, legal metadata) each hand-rolled a
   different subset.
3. **Retries are not feedback-carrying** anywhere except the ask loop's
   `rewrite_suggestion`. A rejected output is retried as a dice roll.
4. **No unified escalation routing.** HITL, terminal events, dead-letter, and typed
   failures are three unrelated mechanisms; a seam author must know which to pick.
5. **Evaluator unavailability is not typed.** If a grader/evaluator model is down,
   seams either crash or silently pass — "vacuous success" (the exact defect class
   PR #109/#110 existed to remove). Absence must be a typed state, never an
   accidental pass.

## 3. The module to build: `src/app/ai/validation/`

Infrastructure-independent (`app/ai` may not import `app.features` — lint-imports
enforces this; features import `app.ai` freely). Pydantic v2 throughout.

### 3.1 Core types (`schema.py`)

```python
class ValidationDecision(StrEnum):
    ACCEPT = "accept"      # schema + semantic gates passed
    RETRY = "retry"        # retryable rejection, attempts remain
    ESCALATE = "escalate"  # retries exhausted, or non-retryable rejection

class GateReport(BaseModel, frozen=True):
    ok: bool
    errors: tuple[SchemaError, ...]        # typed pydantic errors + custom codes
    value: object | None                   # the parsed value when ok

class SemanticReport(BaseModel, frozen=True):
    verdict: Literal["pass", "reject", "unavailable"]
    findings: tuple[SemanticFinding, ...]  # code + message + optional locator
    score: float | None = None             # evaluator-provided, never fabricated

class ValidationOutcome(BaseModel, frozen=True, generic over T):
    decision: ValidationDecision
    value: T | None                        # set iff ACCEPT
    attempts: int
    schema_gate: GateReport
    semantic: SemanticReport | None
    reason: str                            # typed reason code, stable vocabulary
```

### 3.2 The pipeline (`pipeline.py`)

```python
class SemanticEvaluator[T](Protocol):
    async def evaluate(self, value: T, context: EvaluationContext) -> SemanticReport: ...

class ValidationPipeline[T]:
    def __init__(
        self,
        schema: type[T],                      # pydantic model = the schema gate
        evaluator: SemanticEvaluator[T] | None,
        *,
        max_attempts: int,                    # includes the first attempt
        retryable_findings: frozenset[str],   # finding codes worth a retry
        budget: BudgetLedger | None = None,   # optional spend integration
    ) -> None: ...

    async def validate(
        self,
        produce: Callable[[RetryFeedback | None], Awaitable[object]],
        context: EvaluationContext,
    ) -> ValidationOutcome[T]: ...
```

Contract rules (each is a hard requirement, each needs a test):

1. **Schema gate first.** Pydantic parse failure is never passed to the evaluator.
   Coercion is allowed only where the schema declares it.
2. **Evaluator second, and its failure to run is typed.** If the evaluator raises or is
   unconfigured, the report is `verdict="unavailable"` and the decision is
   **ESCALATE** (fail-closed), never ACCEPT. Unavailable ≠ pass.
3. **RETRY carries feedback.** `produce` receives a `RetryFeedback` (schema errors +
   semantic findings + attempt number). Callers append it to the prompt. Retrying
   without feedback is a design violation.
4. **Only retryable finding codes retry.** A non-retryable rejection (e.g. forged
   identity, grounding failure) escalates immediately regardless of attempts left.
5. **Budget honesty.** When a `BudgetLedger` is supplied, every re-invocation of
   `produce` charges it; budget exhaustion mid-pipeline → ESCALATE with reason
   `budget_exhausted`. No fresh allowance may be minted (the PR #110 lesson).
6. **Determinism where possible.** Mechanical checks (span containment, entity-in-quote,
   enum membership) belong in the evaluator implementations as pre-LLM checks; the LLM
   evaluator is only consulted for what mechanics cannot decide.
7. **Observability.** Every outcome emits one structured log event
   (`validation_decision`, decision, reason, attempts, seam name) and OTel counters.
   The caller persists `ValidationOutcome` alongside its result where a record exists.

### 3.3 Evaluator implementations (`evaluators/`)

- `MechanicalEvaluator[T]`: composed of pure predicate callables
  `(value: T, context) -> SemanticFinding | None` (span checks, enum checks,
  referential checks). No model calls. Reusable across seams.
- `LLMEvaluator[T]`: wraps a generation protocol (`app/ai/generation/protocol.py`)
  with its own bounded output schema; returns findings with locators. Must respect
  the caller's budget or declare its own sub-share.
- `CompositeEvaluator`: mechanical first; LLM only if mechanics pass.

### 3.4 Escalation sinks (`escalation.py`)

Typed routing decided by the CALLER, not the pipeline (the pipeline only decides
ESCALATE + reason):

```python
class EscalationSink(Protocol):
    async def escalate(self, outcome: ValidationOutcome, context: EvaluationContext) -> None: ...
```

- `TerminalFailureSink` — converts to the caller's typed failure (documents' graph
  write error path; search's terminal ingest error).
- `HumanReviewSink` — enqueues for HITL (agent_saul's existing review seam).
- `DeadLetterSink` — terminal event / dead-letter (documents terminal bridge).

## 4. Integration order (each step ships with parity or regression tests)

1. **Core module + unit tests** (`tests/unit/ai/validation/`): every contract rule in
   3.2 gets a test; property: no path returns ACCEPT without both gates passing;
   no path returns RETRY without attempts remaining and a retryable finding.
2. **Relationship pass (lowest risk, highest fit).** `relationships.py` already has
   the schema gate and drop reasons. Wrap pass-2 output in the pipeline:
   MechanicalEvaluator (existing validators become predicates) + retry only for
   `span_mismatch`/`unknown_clause` class findings within the existing
   relationship budget share; ESCALATE → existing drop counts + failed state.
   Behavior must be observably identical except for added feedback-carrying retries.
3. **Legal metadata extraction.** Schema gate exists (pydantic). Add a
   MechanicalEvaluator (parties/jurisdiction strings must occur in the boundary text)
   + one bounded feedback-carrying retry on rejection; ESCALATE → existing heuristic
   merge is NOT a fallback — escalate to a persisted `metadata_validation_failed`
   state (honest absence, per the fidelity-metrics philosophy).
4. **Documents ask grade loop.** Refactor `_grade_context` behind
   `SemanticEvaluator[RetrievedContext]` and the loop behind the pipeline. The
   observable ask contract must be byte-identical (parity test on the existing
   grade-loop tests). This deletes hand-rolled retry logic rather than adding any.
5. **agent_saul.** Type-level adoption only: its mechanical gates + entailment +
   HITL already implement the pattern; map them onto the shared report/decision types
   so all seams speak one vocabulary. No behavior change.
6. **Docs + settings.** `PLAN` checkboxes updated; any new settings get
   `SEARCH_*`-style typed Fields; lint-imports must stay 5/5.

## 5. Non-goals

- No new mandatory model call on the happy path (mechanical evaluators are free).
- No prompt-content changes beyond appending structured retry feedback.
- No behavior change to merged seams without a parity test proving it.
- No generic "LLM as judge of everything" — evaluators are per-seam, scoped.

## 6. Definition of done

- `app/ai/validation` importable by every feature; lint-imports 5/5.
- Every contract rule in 3.2 has a named test.
- Two seams integrated with tests proving feedback-carrying retries and typed
  escalation (relationship pass + legal metadata minimum; ask loop as parity refactor).
- All existing suites green; ast-grep at baseline; every `ValidationOutcome` decision
  is reachable in tests, including the unavailable-evaluator ESCALATE path.
