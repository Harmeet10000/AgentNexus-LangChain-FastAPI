# D1 draft — document deletion, retention, and recovery policy (task 7.1)

Status: **DRAFT — awaiting user approval of the retention window and this policy.**
Delete remains **disabled** until this gate closes (tasks 7.1/7.2 rule; design.md §10).
Input: `notes/phase0-verification-records.md` §V10 (no delete operation exists today;
`application/commands/delete_document.py` is a reserved placeholder).

Every decision below is a **DEFAULT** the user can approve or amend. Nothing here is
implemented; this document plus `specs/document-delete-and-access-audit/spec.md` are the
only artifacts this task produces.

---

## Named constants (APPROVED by owner directive 2026-11-14 (formal external review still recommended before real client data per 7.3/9.6))

| Constant | Default | Status |
| --- | --- | --- |
| `DELETION_SOFT_WINDOW_DAYS` | 30 | APPROVED by owner directive 2026-11-14 (formal external review still recommended before real client data per 7.3/9.6) — soft-delete recovery window before hard purge |
| `AUDIT_TOMBSTONE_RETENTION_DAYS` | 90 | APPROVED by owner directive 2026-11-14 (formal external review still recommended before real client data per 7.3/9.6) — how long tombstones + non-content audit records are kept after hard purge |

Rationale for the defaults: 30 days matches common "recently deleted" recovery windows and
gives a legal-hold margin without committing to indefinite content retention; 90 days is a
defensible accountability window for a legal corpus (audit trail long enough to reconstruct
who touched an evidentiary document for a quarter, short enough to be a bounded privacy
commitment). **Neither number is a legal determination** — L1 (task 7.3) and counsel must
confirm before the gate closes. Both are to be named constants in settings, not literals,
so amending them is a config change plus a policy-document edit, not a code search.

---

## Decision 1 — What a user-initiated, policy-approved delete removes (DEFAULT: everything content-bearing)

Deletion is tenant-scoped and idempotent, executed in an auditable sequence. Per §V10 the
operation MUST remove or mark irrecoverable:

1. **Original artifact (R2/S3)**: the object behind `documents.source_uri`/`object_uri`.
2. **Derived rows (Timescale/Postgres)**: `chunks` rows for the document and all its
   versions — canonical chunk text, provenance columns, embeddings, `graphiti_episode_id`,
   `graphiti_verified`.
3. **Graphiti (Neo4j)**: clause-episode nodes and edges within the document's group
   partition. Edges whose citations reference the deleted document are **invalidated,
   not silently orphaned** (Decision 4).
4. **Redis caches**: search/answer caches and `IdempotencyGuard` records keyed by
   tool/step/document scope. Version-scoped keys (tenant + document version + pipeline
   version, per `document-retrieval-schema`) make scope complete; a delete also tolerates
   cache expiry as a backstop since no cached value survives its key.
5. **Idempotency records**: `IdempotencyGuard` entries for the document's operations.
6. **Checkpoint state**: LangGraph checkpoint/thread state that references the document id
   is invalidated for future resumes; threads whose analysis depended on the document MUST
   refuse resume with a typed evidence-unavailable failure rather than silently degrade.

**Not deleted** (Decision 2): the non-content access audit record and the tombstone.
Outbox lifecycle events already emitted for the document are append-only history and are
NOT rewritten; the delete emits a further `document.deleted` lifecycle event.

## Decision 2 — What is RETAINED (DEFAULT: minimum non-content audit + tombstone)

A legal corpus needs accountability after content removal, but retention of content must
be bounded. Exactly two records survive deletion:

- **Minimum non-content read-audit record** — one record per read attempt:
  `authenticated_user_id`, `tenant_id`, `document_id`, `document_version`, operation
  (`read`/`search_hit`/`denied`), timestamp, outcome, correlation/thread identifier.
  **NO document content, NO query text, NO chunk text, NO derived findings.** This matches
  the audit shape in `specs/document-delete-and-access-audit/spec.md` and stays queryable
  for `AUDIT_TOMBSTONE_RETENTION_DAYS` after hard purge.
- **Tombstone record** — one record per deleted document: `document_id`, `tenant_id`,
  deletion event type (`soft_delete`/`hard_purge`/`recovered`), timestamp, actor
  (authenticated principal who performed it), and the document's version range at
  deletion. No content. Purged when `AUDIT_TOMBSTONE_RETENTION_DAYS` elapses from hard
  purge.

## Decision 3 — Recovery semantics (DEFAULT: 30-day soft-delete window, then hard purge)

- **Soft delete** (user- or admin-initiated): content artifacts are tombstoned — marked
  deleted and excluded from every retrieval/evidence path (tenant filters, allowlist,
  BM25/trigram/fused selects, Graphiti reads) — but physically retained for
  `DELETION_SOFT_WINDOW_DAYS`. A tombstone is written immediately.
- **Recovery** (admin-only, within the window): restores the original object, chunk rows,
  and episode nodes to active status, flips the tombstone to `recovered`, and re-emits a
  lifecycle event. Document ids are never reused across tenants. What recovery restores:
  original, chunks, episodes, retrieval eligibility. What recovery **cannot** restore:
  Redis caches and idempotency records (not deleted until hard purge, and anyway
  regenerable), any analysis conclusions that referenced the document while it was
  tombstoned (they re-run at next use), and prior soft/hard purge history beyond the
  audit record itself.
- **Hard purge** (automatic once the window elapses, or admin-initiated early): executes
  the full Decision-1 removal. After hard purge, recovery is impossible except for the
  retained Decision-2 records.

## Decision 4 — Graph edges and entailment (DEFAULT: version-scoped invalidation)

Deleting a document MUST NOT leave graph edges silently citing content that no longer
exists. The strategy is **version-scoped invalidation**, consistent with the chunk
versioning contract (every chunk and evidence envelope carries document version +
pipeline version; caches/idempotency keys are already version-scoped):

- On soft delete, edges (and their entailment relationships) whose citations reference the
  deleted document's id **at any version** are marked `invalidated` with reason
  `document_deleted` — the invalidation record names document id + version(s) + actor +
  timestamp. They become invisible to evidence assembly and MUST be excluded from
  grounded findings; a finding that already cites an invalidated edge surfaces the
  invalidation rather than the citation.
- On recovery within the window, invalidated edges are re-activated with their original
  citations intact.
- On hard purge, edges scoped to the document partition are removed entirely; the
  invalidation record for edges outside the partition (cross-document citations) persists
  as non-content metadata for the audit retention window.

## Decision 5 — Gate status (DEFAULT: delete stays disabled)

- The delete operation is NOT enabled for pilot data until the user approves the two
  retention constants and this policy, and legal/security sign-off (L1, task 7.3) is
  recorded for the provider-retention question.
- Until then, a delete attempt MUST return a typed policy-gate failure
  (`deletion_policy_not_approved`), never a silent no-op or 500.

## Open items for the user

1. Approve/amend `DELETION_SOFT_WINDOW_DAYS` (default 30) and
   `AUDIT_TOMBSTONE_RETENTION_DAYS` (default 90).
2. Confirm whether tombstones must survive beyond 90 days for litigation-hold purposes
   (a hold flag could freeze purges per tenant — not in this default).
3. Confirm the actor model for hard purge (default: admin-only; automatic expiry runs
   as a system principal recorded in the audit trail).
