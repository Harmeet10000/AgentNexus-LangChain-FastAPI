## Purpose

Defines policy-gated document deletion and minimal non-content access auditing so legal corpus lifecycle operations do
not leave originals, chunks, graph episodes, caches, or idempotency records behind without an accountable trail.

## ADDED Requirements

### Requirement: Document deletion removes every derived artifact within policy

Once the retention and recovery policy is approved, an authorized document deletion MUST remove the original, canonical
chunks, Graphiti episodes, search/answer caches, and idempotency records for that document and its versions. Deletion
MUST be tenant-scoped, idempotent, and MUST not be exposed as a model-callable operation.

#### Scenario: Authorized delete removes derived data
- **WHEN** an authorized document-management operation deletes a document after the retention gate is approved
- **THEN** the original and every listed derived artifact MUST be removed or marked irrecoverable, with no cross-tenant deletion

#### Scenario: Delete is blocked before policy approval
- **WHEN** the retention/recovery policy or V10 deletion inventory is unresolved
- **THEN** the delete operation MUST remain disabled and MUST return a typed policy-gate failure

### Requirement: Document reads have a minimal non-content audit trail

Every document/evidence read MUST emit an audit record containing authenticated user, tenant, document and version,
operation, timestamp, outcome, and correlation/thread identifier. The audit MUST NOT include raw document text or
secrets and MUST survive deletion according to the approved retention policy.

#### Scenario: Authorized and denied reads are auditable
- **WHEN** a user reads evidence or is denied an evidence read
- **THEN** an audit record MUST be written for the outcome with no quoted legal content

#### Scenario: Audit survives artifact deletion
- **WHEN** a document's content and derived artifacts are deleted
- **THEN** the policy-permitted non-content access audit MUST remain queryable for accountability

### Requirement: Deletion writes a tombstone retained for the approved retention window

A deletion MUST write a tombstone record containing document id, tenant id, deletion event
type, timestamp, authenticated actor, and the document's version range at deletion. The
tombstone and the non-content access audit MUST be retained for a named retention constant
(REQUIRES-LEGAL-CONFIRMATION) and MUST contain no document content, query text, or derived
findings. Retention constants MUST be configuration, not code literals.

#### Scenario: Tombstone records the deletion without content
- **WHEN** a policy-approved delete executes for a document
- **THEN** a tombstone MUST exist naming the document id, tenant, event, timestamp, actor, and version range, with no document content

#### Scenario: Retained records expire on schedule
- **WHEN** the audit/tombstone retention window elapses after a hard purge
- **THEN** the tombstone and access-audit records for that document MUST be purged according to the configured constant

### Requirement: Soft-delete window allows admin recovery before hard purge

Deletion MUST begin as a soft delete: tombstoned content is excluded from every retrieval
and evidence path yet physically retained for a named soft-delete window
(REQUIRES-LEGAL-CONFIRMATION). Administrator recovery within the window MUST restore
original, chunks, and episodes but never caches or idempotency records. Once the window
elapses, hard purge MUST remove all content artifacts and recovery MUST no longer be
possible.

#### Scenario: Admin recovers within the window
- **WHEN** an administrator recovers a soft-deleted document before the soft-delete window elapses
- **THEN** the original, chunks, and episodes MUST return to active retrieval status with citations intact and the tombstone MUST record the recovery

#### Scenario: Hard purge is final
- **WHEN** the soft-delete window elapses without recovery
- **THEN** the original, chunks, embeddings, episodes, caches, and idempotency records MUST be removed and recovery MUST fail with a typed unavailable outcome

#### Scenario: Tombstoned documents are invisible to retrieval
- **WHEN** a soft-deleted document is tombstoned and recovery has not occurred
- **THEN** no search, evidence read, or grounded finding MUST surface the document's content for any principal

### Requirement: Deleting a document invalidates graph edges version-scoped

Deleting a document MUST invalidate — not silently orphan — graph edges and entailment
relationships whose citations reference it. Invalidation MUST be version-scoped, recording
document id, version(s), reason, actor, and timestamp, consistent with the chunk versioning
contract. Recovered documents MUST restore their invalidated edges; hard purged documents
MUST leave cross-document invalidation records as non-content metadata for the retention
window.

#### Scenario: Edges citing a deleted document are invalidated
- **WHEN** a document is soft deleted
- **THEN** edges citing any version of it MUST be marked invalidated with the reason and MUST be excluded from grounded findings

#### Scenario: Recovery restores invalidated edges
- **WHEN** a soft-deleted document is recovered within the window
- **THEN** its invalidated edges MUST return to active status with original citations intact

### Requirement: Delete remains disabled until the policy gate is approved

The delete operation MUST remain disabled until the user approves this change's retention
and recovery policy and the legal/security provider gate (L1) is recorded. Until then any
delete attempt MUST return a typed policy-gate failure, MUST perform no removal, and MUST
be model-callable by no tool.

#### Scenario: Delete attempts fail typed before approval
- **WHEN** any principal attempts document deletion while the policy gate is unresolved
- **THEN** the operation MUST return a typed policy-gate failure without removing any artifact

#### Scenario: Delete is not a model-callable operation
- **WHEN** the agent or a sub-agent enumerates its callable tools
- **THEN** no delete operation MUST be exposed to the model at any gate state

