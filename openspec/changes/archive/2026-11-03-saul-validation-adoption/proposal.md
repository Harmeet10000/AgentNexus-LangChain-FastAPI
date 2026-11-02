> Change class (pick one): **S** single-file fix / config / bump / docs · **M** feature in one module · **L** cross-cutting (multi-module, migration, security, public API)

**M** — confined to Agent Saul: type-level vocabulary adoption in the verdict/entailment path, one sink wiring, and ledger charging through the existing budget. Explicitly no behaviour change. **Gated change:** implementation is blocked on the validation core existing (`post-llm-validation-core`); sequencing is after `mount-retrieval-graph` (D6) so qna's grade loop is refactored once, not twice.

## Why

Saul's verdict path already *implements* the validation pattern — mechanical evidence-handle gates before model entailment, fail-closed on provider errors — but it speaks a private dialect: local pass/fail shapes, ad-hoc escalation into `make_human_review_node`, and ledger counting beside the pipeline's contract. D7 of `PLAN-agent-tools.md` adopts the shared vocabulary so every seam in the application can be read in one language, without moving a single gate. The middleware boundary is part of the decision: tool-calling seams keep `ModelRetryMiddleware` / `ToolRetryMiddleware` for transport retry, and the validation layer exists for the *other* half — semantic gates over values that already parsed.

## What Changes

- **Type-level adoption of the verdict/entailment path (no behaviour change).** The mechanical gates (`_verify_quotes_against_canonical_source`, span-containment, entity-in-quote) map onto `MechanicalEvaluator` predicate shapes and `SemanticReport` findings with locators; the path's accept/escalate exits map onto `ValidationOutcome` with `ValidationDecision`. Every finding code enters the shared vocabulary. No gate logic moves; existing verdict-path tests stay green, unmodified.
- **`HumanReviewSink` wiring.** `make_human_review_node` becomes the concrete `HumanReviewSink`: an ESCALATE outcome from any Saul-side pipeline routes to the existing HITL review node instead of inventing a fourth escalation path. Routing stays caller-decided — Saul's graph wires the sink.
- **Budget charging through Saul's ledger.** Where a Saul seam runs the pipeline with retries, every `produce` re-invocation charges `domain/budget.py`'s ledger; exhaustion mid-pipeline yields ESCALATE with reason `budget_exhausted`; no fresh allowance is minted. The entailment pass's existing budget attribution is consumed, not replaced.
- **The middleware boundary asserted, not changed.** Transport retry stays in the middleware stack for the tool-calling seams; a test asserts the stack unchanged so this change cannot silently absorb it.

## Scope / Non-Goals

In scope: the type-level mapping, sink wiring, ledger charging, the middleware-boundary assertion, and the acceptance tests (ACCEPT and ESCALATE reachable, unchanged behaviour).

Out of scope (explicit): any gate-logic change or quality movement in the verdict path; the qna engine's grade loop (`mount-retrieval-graph` + the seam-adoption change own it); deep-research synthesis validation (a future seam, noted in D5's review); any change to the tool-calling seams' retry configuration.

## Capabilities

### New Capabilities

None.

### Modified Capabilities

- `agent-saul-delegation`: the verdict/entailment path emits shared-vocabulary validation outcomes; ESCALATE routes to the human-review node through the sink contract; validation retries draw from the budget ledger under the pipeline's honesty rules.

## Impact

Affected code: `src/app/features/agent_saul/graph/nodes.py` (verdict-path types), `graph/factory.py` (sink wiring), `domain/budget.py` (ledger port conformance), tests: type-mapping tests, interrupt-through-sink test, ledger-charge test, middleware-stack assertion. Normative inputs: `PLAN-agent-tools.md` § D7 (adopted verbatim); `post-llm-validation-pipeline` (consumed); `PLAN-result-validation-layer.md` §4 step 5; reference `14-output-validation-and-retry.md` (the middleware/semantic boundary).
