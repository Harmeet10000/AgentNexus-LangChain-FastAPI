## ADDED Requirements

### Requirement: Saul retrieval uses one bounded evidence path

All Saul search, RAG, inline ask, and corrective retrieval MUST use the same document-scoped retrieval contract. The
retrieval graph MUST remain bounded, and any topology change or removal MUST be justified by a recorded experiment.

#### Scenario: Corrective retrieval remains bounded
- **WHEN** grounding reports insufficient context
- **THEN** at most one budgeted corrective evidence search may run for the finding, subject to the run cap

#### Scenario: Retrieval graph removal is evidence-based
- **WHEN** the compiled retrieval graph is considered for deletion
- **THEN** the decision MUST include A/B quality, latency, and cost evidence stratified by question type

