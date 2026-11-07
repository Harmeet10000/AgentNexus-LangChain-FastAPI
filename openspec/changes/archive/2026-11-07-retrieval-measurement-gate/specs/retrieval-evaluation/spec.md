## ADDED Requirements

### Requirement: The golden corpus carries the documented coverage target

The golden retrieval corpus SHALL grow to at least 150 labeled queries toward the documented
150–300 target, and its composition SHALL cover the documented matrix: every document family,
each recorded jurisdiction, the difficulty bands the manifest declares, and the query styles
(obligation lookup, cross-reference, permission ("does X allow Y"), temporal), including the
multi-contract and scanned-OCR profiles the corpus records.

#### Scenario: The corpus size target is met

- **WHEN** the golden corpus manifest is read
- **THEN** it SHALL declare at least 150 labeled retrieval queries, each carrying the provenance
  fields the validator already enforces

#### Scenario: The matrix is covered, not just the count

- **WHEN** the corpus composition is checked
- **THEN** every document family, the multi-contract documents (the 5-boundary packet among
  them), and the scanned-OCR profile SHALL be represented by at least one query, and a query
  added only to inflate the count without extending coverage SHALL be rejected in review

#### Scenario: Growth preserves comparability

- **WHEN** queries are added to the corpus
- **THEN** the corpus version SHALL be incremented so reports before and after the growth are not
  silently compared, and previously recorded baselines SHALL remain addressable by their version

### Requirement: Retrieval metrics gate continuous integration

Continuous integration SHALL run the deterministic retrieval harness over the golden corpus and
SHALL fail the build when recall-at-k or nDCG fall below the recorded threshold, instead of
validating provenance only. The thresholds SHALL be recorded alongside the report they were
derived from, and a threshold change SHALL cite the report that justifies it.

#### Scenario: A quality regression fails the build

- **WHEN** a change degrades measured recall or nDCG on the golden corpus below the recorded
  threshold
- **THEN** the CI gate SHALL fail naming the metric, the observed value, and the threshold

#### Scenario: Metrics run offline and deterministically

- **WHEN** the CI gate executes
- **THEN** it SHALL run without a live database or model provider against the recorded retrieval
  artefacts, and repeated runs on the same tree SHALL produce identical values

#### Scenario: A branch's contribution is attributable

- **WHEN** the harness produces a report for a fused retrieval path with more than one branch
- **THEN** the report SHALL carry per-branch attribution rows so a gated branch's measured
  contribution is visible rather than folded into the aggregate
