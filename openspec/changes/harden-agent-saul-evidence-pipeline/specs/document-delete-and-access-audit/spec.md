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

