## Purpose

Bring the ingestion-side LLM seams — the relationship pass and legal-metadata
extraction — onto the shared validation pipeline, with observably identical behaviour
except where feedback-carrying retries and honest failure states are added.

## ADDED Requirements

### Requirement: Relationship pass-2 validation runs through the shared pipeline

The relationship extraction's validation SHALL run as pipeline validation with the
existing span/clause/vocabulary checks as mechanical predicates. Only the declared
retryable finding classes SHALL trigger a retry, inside the existing relationship
budget share; every other rejection SHALL follow the existing drop-count and failed-
state behaviour unchanged.

#### Scenario: Identical drops except feedback-carrying retries

- **WHEN** pass-2 produces outputs spanning good, retryable-flawed, and
  non-retryable-flawed relationships
- **THEN** good and non-retryable outcomes match the pre-pipeline behaviour exactly
  (same drops, same reasons, same state)
- **AND** retryable-flawed outputs are retried with structured feedback before
  dropping, within the existing budget share

#### Scenario: Budget exhaustion is not a new failure mode

- **WHEN** the relationship budget share is exhausted during retries
- **THEN** the outcome is ESCALATE mapped onto the existing drop/failed-state
  behaviour, with the decision event naming `budget_exhausted`

### Requirement: Legal-metadata extraction is mechanically gated against the boundary text

Extracted parties and jurisdiction SHALL be checked against the document boundary
text by a mechanical evaluator. A rejection SHALL receive exactly one feedback-
carrying retry; a still-rejected extraction SHALL persist an explicit
metadata-validation-failed state and SHALL NOT fall back to a heuristic merge of
unvalidated metadata.

#### Scenario: Fabricated metadata is caught and retried once

- **WHEN** extraction returns a party string that does not occur in the boundary text
- **THEN** exactly one retry runs with the finding as structured feedback
- **AND** a second rejection persists the metadata-validation-failed state rather
  than merging unvalidated metadata

#### Scenario: Honest absence replaces the heuristic fallback

- **WHEN** metadata validation fails terminally for a document
- **THEN** the document record carries the explicit failed state, downstream
  consumers can distinguish it from successful extraction, and no unvalidated
  metadata is merged
