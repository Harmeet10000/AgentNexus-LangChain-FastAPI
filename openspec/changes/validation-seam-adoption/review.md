> Change class: **L** (full checklist + verification matrix). Role: reviewer, not author. Read proposal.md, specs/, design.md before completing anything.

## Completeness

- [x] Each spec requirement has at least one scenario (document-ingestion-pipeline: 2 requirements / 4 scenarios; agentic-retrieval-loop: 2 requirements / 3 scenarios)
- [x] Edge and error cases named: budget exhaustion mid-retries, non-retryable drops, fabricated metadata, second rejection, exhausted loop, fallback equivalence
- [x] Delta operations correct: both are existing capabilities receiving ADDED requirements; the ask-loop requirement is written as an invariance (byte-identical) rather than a new behaviour

## Correctness

- [x] Requirements observably testable — drop counts, persisted states, byte-identical responses, event emission
- [x] No implementation details in specs (predicate signatures, module paths live in design/tasks only)
- [x] The heuristic-merge prohibition is stated as an observable outcome (persisted failed state), not as a code prohibition

## Verification matrix

| Requirement | Proof |
|---|---|
| Relationship pipeline parity | drop-reason regression suite: non-retryable outcomes identical pre/post; retryable outcomes retried with feedback inside the share |
| Relationship budget honesty | exhaustion test → existing drop/failed state + `budget_exhausted` event |
| Metadata mechanical gate | fabricated-string fixture → one feedback-carrying retry → persisted failed state |
| Honest absence | consumer-visible distinction between not-extracted and validation-failed states |
| Ask loop parity | parity suite replay of every grade-loop case, byte-identical |
| Shared-vocabulary decisions | decision-event assertions; fallback-as-ESCALATE equivalence test |

## Decisions to defend in review

1. **Recorded-behaviour parity over equivalence arguments** (design Decision 1) — challenge any proposal to "re-derive" expected outputs at parity time.
2. **The heuristic merge is prohibited, not deprecated** (proposal What Changes) — confirm no consumer depends on unvalidated metadata today; if one does, its migration belongs in this change.
3. **Adoption order is risk-ordered and strict** (Migration Plan) — ask-loop parity last; confirm the reviewer accepts the sequencing dependency between the three landings.

## Cross-change interactions

- `post-llm-validation-pipeline`: consumed wholesale; any rule change goes back to the core change first.
- `mount-retrieval-graph`: independent landings; if the cutover precedes this change, the ask parity suite retargets at the compiled engine (design Decision 5).
- `hierarchical-document-chunking`: the metadata gate checks against boundary text — its fixtures are the gate's test corpus.
- `saul-validation-adoption`: inherits the ask-loop refactor (D7's sequencing note); no direct dependency on this change's code.
