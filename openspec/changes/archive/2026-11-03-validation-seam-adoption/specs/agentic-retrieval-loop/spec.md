## Purpose

Move the ask loop's grade/rewrite machinery onto the shared validation vocabulary with
a byte-identical observable contract, proven by parity before any serving behaviour
can change.

## ADDED Requirements

### Requirement: The grade loop is a pipeline validation with identical observable behaviour

The ask path's context grading and rewrite loop SHALL run through the shared
validation pipeline (grading as the semantic evaluator, the rewrite loop as bounded
feedback-carrying retry). The ask endpoints' observable contract — response shapes,
fallback answer behaviour, warning envelope, cache-key derivation, iteration bound —
SHALL be byte-identical to the pre-pipeline behaviour, proven by a parity suite over
the existing grade-loop tests.

#### Scenario: Parity holds across the existing grade-loop cases

- **WHEN** the parity suite replays every existing grade-loop test case through the
  pipeline-backed loop
- **THEN** every response is byte-identical to the recorded inline behaviour
  (sufficient/insufficient decisions, rewrite usage, fallback, coverage, warnings)

#### Scenario: The rewrite arrives as structured feedback

- **WHEN** the grader returns insufficient with a rewrite suggestion and attempts
  remain
- **THEN** the retry runs with that suggestion carried as pipeline feedback, and the
  two-iteration bound is unchanged

### Requirement: Loop decisions carry the shared vocabulary

Each ask-loop validation SHALL emit one seam-named decision event, and the loop's
terminal outcomes SHALL be expressible as shared-vocabulary decisions (ACCEPT for
sufficient+generated, ESCALATE to the fallback answer for exhausted attempts) without
new escalation mechanisms.

#### Scenario: The fallback is an ESCALATE outcome, not a bespoke path

- **WHEN** both iterations complete with insufficient context
- **THEN** the loop's outcome is an ESCALATE decision whose sink produces the existing
  fallback answer behaviour, byte-identical to today's
