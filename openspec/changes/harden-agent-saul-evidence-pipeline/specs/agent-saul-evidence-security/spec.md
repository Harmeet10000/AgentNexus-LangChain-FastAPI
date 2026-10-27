## Purpose

Defines the server-authoritative security boundary around Agent Saul and legal evidence so identity, tenant scope,
document scope, roles, uploads, and evidence handles cannot be widened by a client or model.

## ADDED Requirements

### Requirement: Authorization is server-derived and immutable for a run

At session start and on every resume, the system MUST derive an authorization context from authenticated claims and
current tenant membership. The context MUST include the authenticated user, tenant, legal role, allowed document IDs,
allowed operations, and thread identity; it MUST NOT be supplied by the model or trusted from client fields.

#### Scenario: Client-supplied authority is ignored
- **WHEN** a start or resume payload contains permissions, reviewer role, user ID, or tenant scope
- **THEN** the server MUST ignore those authority values and derive them from current authenticated claims

#### Scenario: Role revocation fails closed on resume
- **WHEN** a paused thread resumes after its reviewer membership has been revoked
- **THEN** the resume MUST be rejected and no evidence or approval operation MUST run

### Requirement: Every legal read is tenant- and document-scoped

Every legal corpus read MUST require a server-derived tenant ID and an explicit allowed document-ID set. A requested
document outside that set, a document in another tenant, or a non-ready document MUST return a typed denial or readiness
status without revealing its contents or existence beyond the minimum authorization response.

#### Scenario: Cross-tenant identical chunks remain isolated
- **WHEN** two tenants contain identical statute-like chunks and a user in tenant A searches
- **THEN** no result, citation, handle, count, or cache entry from tenant B MUST be returned

#### Scenario: Unauthorized document ID is requested
- **WHEN** a tool receives a document ID not present in the server-derived allowlist
- **THEN** it MUST return `unauthorized` and MUST perform no corpus read

### Requirement: Tenant legal roles constrain operations

Within a tenant, `analyst` MUST be able to run analysis, `reviewer` MUST additionally approve, reject, or override,
and `admin` MUST manage documents but MUST NOT approve a report solely by being an admin. Self-review MUST remain
permitted only when recorded as `self_reviewed` in the audit and report.

#### Scenario: Analyst cannot approve
- **WHEN** an analyst submits an approval or override during human review
- **THEN** the operation MUST be rejected and the report MUST remain unapproved

#### Scenario: Reviewer self-approval is auditable
- **WHEN** a reviewer approves a report they initiated
- **THEN** approval MUST succeed and the report and audit event MUST carry `self_reviewed=true`

### Requirement: Evidence handles are opaque, scoped, and revalidated

Evidence handles MUST be server-issued opaque identifiers bound to a thread, tenant, document version, and run. State
MUST store handles or IDs rather than granting authority through chunk text. Every read, including after a HITL pause,
MUST revalidate the binding and current authorization.

#### Scenario: Forged handle is rejected
- **WHEN** a caller alters or invents an evidence handle
- **THEN** the read MUST return `unauthorized` or `no_evidence` without disclosing chunk content

#### Scenario: Handle from another thread is rejected
- **WHEN** a valid handle issued to thread A is presented by thread B
- **THEN** the read MUST fail closed

### Requirement: Upload and extraction budgets are data-informed and fail closed

The upload surface MUST enforce a documented, data-informed maximum contract size and extension allowlist before fully
buffering or parsing. The system MUST NOT rely on an arbitrary hard-coded 10 MiB ceiling without a recorded corpus
decision. Each accepted document MUST also have bounded extraction cost, latency, model-call/token, and Celery task
budgets. The supplied MIME type MUST NOT be the sole type check.

#### Scenario: Oversized upload is bounded
- **WHEN** a client sends a file whose declared or observed size exceeds the approved configured maximum
- **THEN** the request MUST return a typed validation error and MUST NOT read the complete file into memory

#### Scenario: Large accepted document is budgeted
- **WHEN** a file is within the approved upload size but extraction would exceed its cost, latency, model, or task-time budget
- **THEN** ingestion MUST stop or complete with a typed warning/error and MUST NOT run unbounded work

#### Scenario: Disallowed extension is rejected
- **WHEN** a file extension is not allowlisted
- **THEN** the upload MUST be rejected before parsing regardless of its MIME header
