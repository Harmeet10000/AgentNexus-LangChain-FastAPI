## ADDED Requirements

### Requirement: Ingestion exposes truthful typed terminal states

Queued document ingestion MUST expose typed statuses that distinguish received, processing, completed,
completed-with-warnings, and failed, with a stable reason for failure. A dead-lettered dispatch MUST transition the
document out of `received` and MUST emit a terminal event through the outbox for UI consumers.

#### Scenario: Worker completion notifies the UI
- **WHEN** a queued document reaches completed or completed-with-warnings
- **THEN** an outbox terminal-state event MUST be emitted and the UI MUST be able to observe readiness

#### Scenario: Dead letter is terminal
- **WHEN** all dispatch attempts are exhausted
- **THEN** the document MUST become failed with a reason and MUST NOT remain received forever

### Requirement: Canonical chunks and provenance are stable evidence input

The queued `/documents` pipeline MUST be the canonical producer of retrievable chunks. Each chunk returned as evidence
MUST carry document version and pipeline provenance, and partial extraction MUST be reported as a warning rather than
silently presented as complete coverage.

#### Scenario: Partially processed corpus is disclosed
- **WHEN** a document completes with extraction warnings
- **THEN** evidence and status responses MUST expose completed-with-warnings and the affected coverage

### Requirement: Multi-contract documents preserve contract-local metadata

When an accepted file contains multiple contracts, the pipeline MUST establish contract boundaries before classification
and legal-metadata extraction, run those operations per contract, and attach a stable `contract_index` to every affected
chunk. The index MUST be carried to relationship episodes, evidence references, findings, and reports. If the Phase 0
fixture confirms one contract per supported file, the pipeline MUST retain the fixture and record that result rather than
silently assuming it forever.

#### Scenario: Two contracts receive separate metadata
- **WHEN** a PDF contains two contracts with different parties or governing law
- **THEN** each contract's chunks and metadata MUST carry distinct contract identity and no metadata value may mix them

#### Scenario: Boundary detection is insufficient
- **WHEN** contract boundaries cannot be established with sufficient confidence
- **THEN** ingestion MUST complete with a typed warning or fail and MUST NOT publish mixed legal metadata

### Requirement: Fidelity gates surface source loss before retrieval

The pipeline MUST compute a source text-coverage ratio and compare produced chunks with source text for coverage and
order. It MUST record typed `completed_with_warnings` reasons for materially lost tables/sections, implausible chunk
counts, low OCR confidence or coverage when OCR is enabled and the deployment exposes those signals, or other
configured fidelity failures. Fidelity checks MUST be independent of retrieval relevance.

#### Scenario: Table-heavy or scanned source loses content
- **WHEN** a table-heavy or scanned contract has low source-to-chunk coverage or OCR quality
- **THEN** the document MUST be marked `completed_with_warnings` with the applicable reason before Saul can analyze it

#### Scenario: Clause ordering is corrupted
- **WHEN** chunk order does not preserve source order beyond the configured tolerance
- **THEN** ingestion MUST expose a fidelity warning and MUST NOT present the document as fully complete

### Requirement: Ingestion treats contract text as untrusted input

Extraction and metadata models MUST treat document text as untrusted content. Prompt-injection fixtures containing
instructions or forged party names MUST NOT alter authoritative metadata or cause relationship writes outside the
document's server-derived tenant, document, and contract scope.

#### Scenario: Injected clause instructions are ignored
- **WHEN** a contract clause instructs the extraction model to change parties, roles, or system behavior
- **THEN** authoritative metadata MUST remain source-derived and the instruction MUST NOT widen graph or evidence scope

#### Scenario: Forged party name cannot poison Graphiti
- **WHEN** injected text attempts to create a party or relationship outside the current document/contract
- **THEN** the write MUST be rejected or scoped to the current document and the CI fixture MUST fail on any poisoning

### Requirement: Ingestion has a source-to-chunk gold set

The repository MUST maintain a versioned ingestion gold set covering scanned PDFs, tables, clause numbering, annexures,
multi-contract files, and a Phase-3-style clause-kind document. Labels MUST describe source-to-chunk coverage and order,
not retrieval relevance, and the set MUST run against the queued pipeline.

#### Scenario: Gold set checks source fidelity
- **WHEN** the queued pipeline runs against the ingestion gold set
- **THEN** the report MUST compare expected coverage/order and typed warning outcomes for every fixture
