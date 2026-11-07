## Purpose

Define the standing governance record for disconnected, orphaned, and quarantined surfaces in the
source tree: every kept surface is written down with a reason and an exit criterion, deletion is
always a recorded decision, and re-wiring a quarantined surface is gated by the boundary contract
that would have caught the original disconnection.

## ADDED Requirements

### Requirement: The quarantine ledger exists and records every kept surface

A durable quarantine ledger SHALL exist at `docs/quarantine-ledger.md`, and every surface in the
source tree that is orphaned (zero production importers), unmounted, or deliberately kept
disconnected SHALL have a row in it. Each row SHALL name the surface, its path, why it is kept, a
port-or-discard criterion, and a revisit trigger.

#### Scenario: A known orphan is recorded retroactively

- **WHEN** the ledger is first created
- **THEN** it SHALL contain rows for, at minimum: the docling `ingest_v2.py` orphan, the
  `langextract_batch_processor.py` and `docling_preprocessor.py` orphans, the folded
  `features/ingestion` compatibility shim (folded into `documents/api/http/` by commit `25ba6f5`),
  the legacy graph path state keys, and the unmounted retrieval graph awaiting the
  `mount-retrieval-graph` cutover

#### Scenario: A row names its exit

- **WHEN** any ledger row is read
- **THEN** it SHALL state a port-or-discard criterion and a revisit trigger, not merely a reason
  to keep the surface

#### Scenario: A quarantine without a trigger is not a quarantine

- **WHEN** a surface is proposed for quarantine during review
- **THEN** the proposal SHALL be rejected unless its ledger row names a revisit trigger, because
  a quarantine without a trigger is a landfill with better prose

### Requirement: Nothing gets deleted without a ledger row

A surface recorded in the quarantine ledger SHALL NOT be deleted from the source tree except as a
recorded discard decision against its row, and a surface that was never quarantined SHALL NOT be
silently deleted while disconnected — it SHALL be quarantined first, then discarded.

#### Scenario: A discard is a ledger update, not just a commit

- **WHEN** a quarantined surface is deleted
- **THEN** the ledger row SHALL be resolved in the same change (marked discarded, with the
  criterion that fired), not merely removed from disk

#### Scenario: A discovered orphan is quarantined, not purged

- **WHEN** a session discovers a disconnected surface that has no ledger row
- **THEN** the surface SHALL gain a row before any deletion of it is proposed

### Requirement: Re-wiring a quarantined surface requires the boundary contract first

Any re-wiring or revival of a quarantined graph-driving surface SHALL first land the state key-set
boundary contract defined by the `ingestion-state-boundary` capability, because
`TypedDict(total=False)` plus LangGraph's silent key-dropping leave no type-system protection on
that edge.

#### Scenario: The legacy path is re-wired

- **WHEN** the quarantined ingestion compatibility service is wired to any graph again
- **THEN** the key-set boundary test SHALL exist and pass against the new wiring before the
  wiring ships
