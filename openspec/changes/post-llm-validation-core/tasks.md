> Change class: **M**. Tasks ordered by dependency; each has a concrete verification outcome. Pure addition — no existing seam is touched.

## 1. Core types (`schema.py`)

- [x] 1.1 `ValidationDecision` (ACCEPT/RETRY/ESCALATE), `GateReport`, `SemanticReport` (verdict `pass | reject | unavailable`, findings with locators, optional never-fabricated score), frozen generic `ValidationOutcome[T]` with the validator-enforced value-iffy-ACCEPT invariant. Verify: unit tests for the invariant (constructing a value-bearing non-ACCEPT outcome raises). — `schema.py`; invariant proven by `tests/unit/ai/validation/test_schema.py` (value-bearing non-ACCEPT and valueless ACCEPT both raise ValidationError; frozen enforced).

## 2. Pipeline (`pipeline.py`)

- [x] 2.1 `SemanticEvaluator[T]` protocol and `ValidationPipeline[T]` with `max_attempts` (including the first), `retryable_findings`, optional `budget` port. Verify: the seven contract rules each have a named test — schema-first short-circuit; unavailable ⇒ ESCALATE (unconfigured + raising); feedback-carrying RETRY (errors + findings + attempt number); retryable-only; per-reinvocation budget charge + `budget_exhausted`; one decision event per outcome. — `pipeline.py` (+`BudgetLedger` port); all named in `tests/unit/ai/validation/test_pipeline_contract_rules.py` (12 tests, incl. `test_schema_failure_short_circuits_the_evaluator` asserting the evaluator was NOT invoked, and the one-event-per-outcome loguru capture).
- [x] 2.2 Property tests: no path returns ACCEPT without both gates passing; no path returns RETRY without attempts remaining and a retryable finding. Verify: property suite green, including the unavailable-evaluator ESCALATE path required by the plan's definition of done. — `tests/unit/ai/validation/test_pipeline_properties.py`: hypothesis (100 examples) over scripted verdicts incl. `unavailable`/`raise`; every re-invocation proven preceded by retryable finding + attempts + accepted budget charge.

## 3. Evaluators (`evaluators/`)

- [x] 3.1 `MechanicalEvaluator` (pure predicate composition, zero model calls), `LLMEvaluator` (bounded output schema over `app/ai/generation`, budget-respecting), `CompositeEvaluator` (mechanical first, model only on mechanical pass). Verify: mechanical short-circuit test (no model call on predicate failure); composite ordering test. — `evaluators/{mechanical,llm,composite}.py`; `tests/unit/ai/validation/test_evaluators.py`: `test_mechanical_rejection_never_costs_a_model_call` (spy LLM evaluator: 0 calls) + composite ordering and LLM budget-charge tests over `InMemoryTextGenerator`.

## 4. Escalation sinks (`escalation.py`)

- [x] 4.1 `EscalationSink` protocol + `TerminalFailureSink`, `HumanReviewSink`, `DeadLetterSink` as thin caller-wired adapters; the pipeline imports only the protocol. Verify: two-sink equivalence test (same ESCALATE routes differently per caller config); import-boundary test. — `escalation.py` (pipeline module imports no sink — proven by `test_pipeline_module_imports_no_sink_implementation`); equivalence in `test_escalation.py::test_same_escalate_routes_differently_per_sink`.

## 5. Boundaries and gates

- [x] 5.1 lint-imports stays 5/5 with `app/ai/validation` importable by every feature; `app/ai/validation` imports nothing from `app.features`. Verify: lint-imports run + an import test. — `uv run lint-imports`: 5 kept, 0 broken; namespace-based `test_validation_package_imports_nothing_from_features` green.
- [x] 5.2 `uv run ruff check --fix src/` · `uv run ty check src/` · `uv run pytest tests/unit/ai/validation/ -q` · `ast-grep scan src/` — green. — ruff clean on `src/app/ai/validation/` + tests (format clean); `ty check src/app/ai/validation/` all checks passed; `pytest tests/unit/ai/validation/ -q` 35 passed and `pytest tests/unit/ai/ -q` 96 passed (no regressions); `ast-grep scan src/` reports zero findings in `app/ai/validation` (the 34 pre-existing errors elsewhere in src/ are unchanged baseline).

## 6. Archive

- [ ] 6.1 Sync the delta into `openspec/specs/post-llm-validation-pipeline/` (new).
