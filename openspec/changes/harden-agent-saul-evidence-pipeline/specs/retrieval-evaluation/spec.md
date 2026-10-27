## ADDED Requirements

### Requirement: The evaluation set measures grounding and security as well as retrieval

The versioned evaluation set MUST include direct, multi-document, multi-hop, unanswerable, missing-authority,
conflicting-clause, adjacent-clause, definition/exception, prompt-injection, and two-tenant cases. Reports MUST
separate retrieval relevance from atomic-claim entailment and record judge model/prompt and SME adjudication where used.

#### Scenario: Unanswerable question rewards abstention
- **WHEN** the corpus cannot support an answer
- **THEN** the score MUST measure correct abstention separately from retrieval recall

#### Scenario: Scope violation fails the suite
- **WHEN** a query returns another tenant's chunk or an invented/forged evidence ID is cited
- **THEN** the evaluation MUST mark the run failed regardless of answer quality

### Requirement: Quality and operations metrics are recorded per experiment

Evaluation reports MUST include evidence recall, atomic entailment, unsupported-claim rate, correct abstention,
citation validity, scope violations, tool calls, p95 latency, tokens, and cost, with a comparison baseline. Changes
to retrieval or prompting MUST cite a reproducible report.

#### Scenario: Retrieval improvement is reproducible
- **WHEN** a change claims better retrieval or grounding
- **THEN** it MUST cite a report from the same versioned gold set and baseline

### Requirement: Ingestion and multi-contract evaluations are separate from retrieval relevance

The evaluation suite MUST include ingestion-fidelity results from the versioned source-to-chunk gold set and a
multi-contract comparison of per-document versus per-contract metadata. These results MUST be reported separately from
retrieval relevance and MUST gate deletion/port decisions for legacy ingestion stages.

#### Scenario: Ingestion fidelity is measured
- **WHEN** the queued pipeline runs against the ingestion gold set
- **THEN** the report MUST include source coverage/order, warning reasons, OCR/table outcomes, and contract identity

#### Scenario: Legacy-stage port decision cites evidence
- **WHEN** a Phase-3 canonicalization, contextualization, merge, or fan-out stage is considered for porting
- **THEN** the decision MUST cite the relevant ingestion/retrieval experiment result and its quality/cost trade-off
