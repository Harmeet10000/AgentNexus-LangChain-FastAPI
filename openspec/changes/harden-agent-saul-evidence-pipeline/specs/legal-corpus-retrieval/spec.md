## ADDED Requirements

### Requirement: Legal retrieval is explicitly scoped and deterministic

Legal search and point retrieval MUST accept a server-derived tenant and allowed document scope, optional jurisdiction
and clause filters, and MUST use one shared ranked/fusion behavior. Results MUST include page and provenance where
available.

#### Scenario: Cross-document scope is explicit
- **WHEN** a legal query is issued for several documents
- **THEN** only the requested allowed document IDs MUST participate, with no implicit tenant-wide expansion

#### Scenario: Search and ask agree on ranking
- **WHEN** search, RAG, and inline ask issue the same query and filters
- **THEN** they MUST return the same ranked evidence identifiers

### Requirement: Retrieval failure statuses are not empty evidence

The legal evidence contract MUST return a typed envelope with one of `ok`, `no_evidence`, `index_not_ready`,
`ingestion_failed`, `unauthorized`, `transient_failure`, `timeout`, or `budget_exhausted`, plus coverage, retryability,
and provenance. An empty result MUST NOT represent an unavailable or unauthorized corpus.

#### Scenario: Corpus outage is distinguishable
- **WHEN** Postgres, Redis, or Graphiti retrieval times out or fails
- **THEN** the response MUST report the corresponding transient/timeout status and MUST NOT claim no evidence

#### Scenario: No matching authority is a valid negative
- **WHEN** all allowed ready documents are searched successfully and no chunk matches
- **THEN** the response MUST return `no_evidence` with complete coverage

### Requirement: Graphiti is a coverage signal, not legal truth

Graphiti relations MAY narrow or supplement retrieval and report coverage, but canonical persisted chunks remain the
source of truth for quoted evidence. Graphiti failure MUST degrade the coverage signal without authorizing a broader
corpus or suppressing valid canonical chunks.

#### Scenario: Graphiti is unavailable
- **WHEN** Graphiti is down during a legal evidence search
- **THEN** canonical chunk retrieval MAY succeed, graph coverage MUST be marked unavailable, and no Graphiti-only claim
  MUST be treated as verified

### Requirement: Retrieval and ingestion degradation are observable

The system MUST emit count and rate counters and expose dashboard/alert signals for Graphiti unavailability, rerank
fallback, grader fallback, `completed_with_warnings`, stuck documents, dead-lettered events, `index_not_ready`, and
`no_evidence`. Alerts MUST be configurable for a sustained rise and MUST preserve tenant-safe correlation without
logging legal text.

#### Scenario: Sustained degradation alerts
- **WHEN** any configured degradation counter rises above its sustained threshold
- **THEN** the observability system MUST emit an alert containing the counter, window, and safe correlation context

#### Scenario: Fallback is counted
- **WHEN** reranking or grading falls back to a deterministic/degraded path
- **THEN** the corresponding fallback counter MUST increment and the evidence/report coverage MUST remain truthful

#### Scenario: Legacy ask distinction is explicit
- **WHEN** `/legal/ask` continues to require `graphiti_verified` while Agent Saul accepts grounded canonical chunks
  that are not graph-indexed
- **THEN** the difference MUST be documented and surfaced as coverage semantics, and neither path MUST treat Graphiti
  availability as legal truth
