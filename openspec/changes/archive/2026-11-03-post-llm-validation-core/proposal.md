> Change class (pick one): **S** single-file fix / config / bump / docs · **M** feature in one module · **L** cross-cutting (multi-module, migration, security, public API)

**M** — one new infrastructure module (`src/app/ai/validation/`) plus its test suite. It changes no existing behaviour: every existing seam keeps working exactly as today until the adoption changes land. It is the foundation three adoption changes depend on, which is why it ships alone, first.

## Why

Every LLM seam in the application invents its own accept/retry/escalate shape: the ask loop's `ContextGrade`, the relationship pass's drop reasons, extraction's failure codes, agent_saul's review flags. Retries are dice rolls (no feedback carried), evaluator outages are silent passes ("vacuous success" — the exact defect class PRs #109/#110 existed to remove), and a seam author must know three unrelated escalation mechanisms to fail honestly. `PLAN-result-validation-layer.md` specifies the unified layer; its integration-order step 1 is this change. D7 of `PLAN-agent-tools.md` and both adoption changes are gated on it.

## What Changes

- **New module `src/app/ai/validation/`**, infrastructure-independent (`app/ai` may not import `app.features` — the lint-imports boundary holds), Pydantic v2 throughout:
  - `schema.py` — the shared decision vocabulary: `ValidationDecision` (ACCEPT / RETRY / ESCALATE), `GateReport`, `SemanticReport` (verdict `pass | reject | unavailable`, findings with locators, never-fabricated score), generic `ValidationOutcome[T]` (decision, value iff ACCEPT, attempts, both reports, stable typed reason).
  - `pipeline.py` — `SemanticEvaluator[T]` protocol and `ValidationPipeline[T]`: schema gate first, evaluator second, bounded attempts, retry only for declared retryable finding codes, optional budget-ledger integration.
  - `evaluators/` — `MechanicalEvaluator` (composed pure predicates, zero model calls), `LLMEvaluator` (bounded output schema, budget-respecting), `CompositeEvaluator` (mechanical first; LLM only if mechanics pass).
  - `escalation.py` — `EscalationSink` protocol with `TerminalFailureSink`, `HumanReviewSink`, `DeadLetterSink`. Routing is caller-decided; the pipeline only decides ESCALATE + reason.
- **Seven contract rules as hard requirements** (each with a named test): schema-gate-first; evaluator-unavailable ⇒ ESCALATE (fail-closed, never ACCEPT); RETRY carries structured feedback into `produce`; only retryable finding codes retry; budget honesty (every re-invocation charges, exhaustion ⇒ ESCALATE `budget_exhausted`, no fresh allowance); mechanical checks before model calls; one structured `validation_decision` event per outcome.
- **Nothing else changes.** No existing seam is rewired in this change — adoption is deliberately separated so each seam lands with its own parity proof.

## Scope / Non-Goals

In scope: the module, its types, the pipeline, the evaluator and sink implementations, the contract-rule test suite including the property tests and the unavailable-evaluator path.

Out of scope (explicit): adopting any existing seam (the `validation-seam-adoption` and `saul-validation-adoption` changes); prompt-content changes beyond appending structured retry feedback at adoption time; any new mandatory model call on the happy path; generic "LLM as judge of everything" (evaluators are per-seam, scoped).

## Capabilities

### New Capabilities

- `post-llm-validation-pipeline`: the shared typed pipeline every LLM seam may run its output through — schema gate, typed semantic evaluation, feedback-carrying bounded retry, budget honesty, fail-closed unavailability, caller-routed escalation, one decision event per outcome.

### Modified Capabilities

None.

## Impact

New code: `src/app/ai/validation/` (schema, pipeline, evaluators, escalation), `tests/unit/ai/validation/`. Affected boundaries: `app/ai` import rules (lint-imports must stay 5/5 — the module may import `app.ai.generation` and `app.resilience` only). Normative inputs: `PLAN-result-validation-layer.md` (§3 types and contract rules are adopted verbatim as requirements); PR #109/#110 lessons (vacuous success, fresh-allowance minting) are the named failure modes; `feature-error-contract` (typed failure vocabulary consumed, not modified).
