> Change class (pick one): **S** single-file fix / config / bump / docs · **M** feature in one module · **L** cross-cutting (multi-module, migration, security, public API)

**L** — it rewires three independent LLM seams (relationship extraction, legal-metadata extraction, the documents ask grade loop) onto the shared validation pipeline, each under a parity gate on live behaviour. The seams live in different capabilities, and the ask loop is a served retrieval path — the combination is what makes this L rather than three Ms.

## Why

Integration-order steps 2–4 of `PLAN-result-validation-layer.md`: the core pipeline exists only as vocabulary until real seams adopt it, and each of these three seams currently hand-rolls a different subset of validation. Adopting in this order (lowest risk, highest fit first) proves the pipeline against the seams that already have the strongest validation, and retires hand-rolled retry logic rather than adding any. The ask-loop refactor is deliberately a parity refactor: the observable ask contract must be byte-identical.

## What Changes

- **Relationship pass (step 2).** Pass-2 output wraps in the pipeline: the existing span/clause/vocabulary validators become `MechanicalEvaluator` predicates; retry applies only to `span_mismatch`/`unknown_clause`-class findings within the existing relationship budget share; ESCALATE maps onto the existing drop counts and the failed relationship state. Behaviour is observably identical except for the added feedback-carrying retries.
- **Legal metadata extraction (step 3).** A `MechanicalEvaluator` asserts extracted parties/jurisdiction strings occur in the boundary text; rejection gets exactly one bounded feedback-carrying retry; ESCALATE persists a `metadata_validation_failed` state — the existing heuristic merge is explicitly **not** a fallback (honest absence, per the fidelity-metrics philosophy).
- **Ask grade loop (step 4).** `_grade_context` moves behind `SemanticEvaluator[RetrievedContext]` and the two-iteration rewrite loop behind the pipeline. The observable ask contract — response shapes, fallback, warnings, cache keys — is byte-identical (parity suite on the existing grade-loop tests). This deletes hand-rolled retry logic rather than adding any.
- **Common to all three:** one `validation_decision` event per outcome (seam-named), outcomes persisted alongside seam records where a record exists, and zero new mandatory model calls on the happy path.

## Scope / Non-Goals

In scope: the three seam integrations above, their parity/regression tests, and the decision-event wiring.

Out of scope (explicit): agent_saul adoption (the `saul-validation-adoption` change — D7, sequenced after `mount-retrieval-graph`); changes to the pipeline itself (only consumed; if a seam proves a rule wrong, the rule changes in the core change first); prompt-content changes beyond appending structured retry feedback; the retrieval engine cutover (`mount-retrieval-graph`).

## Capabilities

### New Capabilities

None.

### Modified Capabilities

- `document-ingestion-pipeline`: relationship pass-2 validation runs through the shared pipeline with feedback-carrying retries inside the existing budget share; legal-metadata extraction gains a mechanical boundary-text gate, one bounded retry, and a persisted honest-failure state instead of a heuristic fallback.
- `agentic-retrieval-loop`: the grade/rewrite loop runs through the shared pipeline with identical observable behaviour (parity-gated); the loop's decisions become shared-vocabulary outcomes.

## Impact

Affected code: `src/app/extraction/langextract/relationships.py` (validators → predicates, pipeline wrap), legal-metadata extraction path (`_prepare_legal_metadata` chain) + the persisted metadata-validation state, `src/app/features/documents/application/queries/documents.py` (grade loop refactor), tests: parity suites for all three seams, drop-count/drop-reason regression, metadata-state tests. Normative inputs: `PLAN-result-validation-layer.md` §4 steps 2–4 (adopted verbatim); `post-llm-validation-pipeline` (consumed); `hierarchical-document-chunking` (boundary text the metadata gate checks against).
