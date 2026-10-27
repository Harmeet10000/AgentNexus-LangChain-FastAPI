> Change class: **L**. The proposal establishes motivation and externally visible contracts; this document fixes the architecture and rollout seams before implementation.

## Context

See `proposal.md` for motivation. The current system has an established Agent Saul WebSocket/HITL graph, queued
`/documents` ingestion, persisted chunks, and a shared retrieval stack. The design must preserve the graph's transcript
and pause/resume behavior while closing authority and provenance gaps. The application is an async modular monolith:
routers are thin, services orchestrate, repositories own persistence, worker tasks own queue execution, and expected
domain failures travel as typed results/state rather than transport exceptions.

The most important boundary is that the model may choose a query or evidence handle but never chooses identity,
tenant, document allowlists, reviewer role, or thread scope. Graphiti is optional relationship infrastructure; a
canonical chunk in the tenant-scoped document store is the only source that can support a quoted legal claim. The
ingestion pipeline must also preserve contract identity and source fidelity before downstream evidence work trusts it.

## Goals / Non-Goals

**Goals:**

- Make every read auditable as `(authenticated user, tenant, allowed documents, document version, pipeline version,
  thread)` and fail closed when any component is unresolved.
- Give all retrieval surfaces one bounded, deterministic evidence contract with typed availability and coverage states.
- Keep checkpoint payloads small and resumable by storing IDs/opaque handles, not legal text.
- Make findings atomic, grounded, reviewable, and honest about incomplete ingestion or graph coverage.
- Make migrations reversible and rollout observable, with permanent isolation and grounding tests as release gates.

**Non-Goals:**

- No change to Graphiti group IDs, no shared external statute corpus, and no MCP exposure.
- No model-callable upload, resume, approval, or re-ingestion; no Saul-side LLM segmentation.
- No legal certification, exact offset highlighting, or claim of production readiness without legal/security/customer
  validation.

## Decisions

### 1. One immutable authorization context at the tool boundary

Construct an immutable server-side context at session creation and reconstruct it on resume from current auth claims,
tenant membership, document readiness, and thread identity. Pass it through `ToolRuntime` into tools and sub-agents.
The context is not serialized into model messages and is not accepted as tool arguments. Every repository predicate
uses `tenant_id` plus the allowlist; `user_id` remains an audit field. If `ToolRuntime` cannot reach compiled
sub-agents, close over a validated context when each tool set is built; do not fall back to model-visible parameters.

**Alternative rejected:** trusting `WSStartMessage.permissions` or `WSResumeMessage.reviewer_role` is simpler but is
an authority-confusion vulnerability. A global request context is also rejected because it is unsafe across concurrent
WebSocket runs and worker tasks.

### 2. Canonical chunks, handles in state, text at call time

The queued ingestion pipeline owns extraction, structure, chunks, and terminal status. Saul consumes a deterministic
chunk-to-segment adapter (`chunk_id` is the clause identity) and never performs model segmentation or cross-document
concatenation. Retrieval returns a result envelope and issues opaque, run-scoped handles. LangGraph state stores
handles, IDs, typed findings, and small metadata; `read_evidence_by_id` fetches text after authorization and version
checks. A single `_row_to_chunk`/hydration path keeps page and provenance behavior consistent across search, RAG, and ask.

**Alternative rejected:** storing full text in checkpoints makes audit and retention unsafe and causes large-document
state growth. Letting Graphiti issue evidence handles would make relation availability equivalent to legal truth.

### 3. Shared retrieval service with explicit degradation

Implement one `retrieve(ctx, query, filters, k, budget)` service composed of fused Postgres retrieval, optional source
narrowing, Graphiti narrowing/coverage, reranking with deterministic fused-order fallback, and token budgeting. Every
database, Redis, and Graphiti call has an explicit timeout. The service returns typed statuses rather than collapsing
outages, unauthorized reads, incomplete indexes, and no matches into an empty list. Existing specialized tools either
delegate to this service with restricted filters or are removed after parity evidence.

**Alternative rejected:** retaining separate search/RAG/inline-ask algorithms creates ranking drift and makes a green
path conceal a broken path. A Graphiti-first design is rejected because queued chunks are canonical and Graphiti can
be unavailable or incomplete.

### 4. Versioned tenant migration and compatibility adapter

Add tenants and membership with a 1:1 backfill from current users, keep `user_id` for audit, and migrate predicates and
cache/idempotency keys in one controlled schema change. Add chunk `pipeline_version` and `graph_indexed_at` (legacy for
backfilled rows), while preserving existing Graphiti group IDs. Replace state `doc_id` with `allowed_doc_ids` and
`primary_doc_id`; a versioned hydrator upgrades old checkpoints before any node reads them. Old DTO authority fields are
removed/ignored at the edge so unknown fields cannot grant scope.

**Alternative rejected:** using `user_id` as the long-term tenant key would prevent team membership and makes future
sharing semantics ambiguous. A dual tenant/user predicate is retained during backfill only; the end state is tenant
scope plus explicit document allowlist.

### 5. Grounding before approval, persistence, or finalization

Risk and compliance emit typed findings into a claim ledger. Grounding first runs deterministic handle/run/scope/version
and quote-substring checks, then entailment on survivors (a distinct model where practical). Unsupported or contradicted
claims are rejected/downgraded. Human review sees those states and coverage; approval is role-checked again on resume;
persist memory runs only after approval. `not_assessed`/`not_assessable` is a first-class result, never a pass.

**Alternative rejected:** trusting `result.verified` or a Graphiti relation is not a proof of quote validity or claim
entailment. A single final summary is rejected because it discards atomic evidence and prevents targeted review.

### 6. Queue status and events are the ingestion source of truth

The worker records typed stage/terminal statuses and reasons, marks exhausted dead letters as failed, and emits a
terminal outbox event for UI notification. The request only records/enqueues. Ingestion state is read by the
orchestrator and evidence tools; it is never inferred from an empty search result. Stuck-age detection and retry
metrics are emitted from worker/outbox boundaries.

### 7. Explicit budgets, retention, and observability

Each run owns a budget ledger (calls, retrievals, fan-out, tokens, wall clock) and a bounded corrective search. Logs
carry correlation IDs, tenant/document hashes or IDs permitted by policy, pipeline/state versions, result status,
retryability, and latency; they never carry raw legal text or secrets. Evidence handle issuance/read, authorization
denials, role changes, grounding outcomes, graph availability, and coverage are metrics/audit events. Paused threads
are retained for 24 hours and garbage-collected after 7 days; obsolete tool outputs are compacted/cleared.

### 8. Contract-aware ingestion and fidelity gates

Run the Phase 0 two-contract and OCR checks before choosing the implementation branch. If classification confirms that a
file can contain multiple contracts, detect boundaries before classification/legal-metadata extraction, assign a stable
`contract_index` to chunks, and carry it through Graphiti episodes, evidence references, findings, and reports. If the
fixture proves one contract per file for the supported corpus, retain the fixture and record that result rather than
inventing a segmentation layer. In either branch, compare chunk text with source text for coverage and order and record
typed warnings for lost tables/sections, implausible chunk counts, low OCR confidence, or other fidelity failures.

The upload ceiling is a documented, data-informed configuration decision rather than an unconditional 10 MiB rule. A
separate per-document ledger caps extraction cost, latency, model calls/tokens, and Celery task time; exceeding a cap
returns a typed failure or `completed_with_warnings` according to the stage contract. Contract text is untrusted input:
injection fixtures must prove that instructions or forged party names cannot alter metadata or widen Graphiti writes.

**Alternative rejected:** a fixed arbitrary byte ceiling can reject legitimate contracts, while no ceiling allows one
large document to monopolize workers or LLM spend. Extracting metadata once for a whole multi-contract PDF silently
mixes parties and governing law; relying only on retrieval metrics misses source loss before retrieval.

### 9. Native tool and transport boundary

Saul binds native `@tool` functions only; MCP is a future integration and is not part of this change. Read-only
evidence and knowledge operations execute under the server-derived context without an approval interrupt. Mutating
operations, including memory persistence, are unavailable until an authorized human approval and remain role-checked
on resume. Uploads remain HTTP-only, and the agent receives document identifiers; resume, approval, and re-ingestion
are not model-callable tools.

### 10. Deletion, access audit, and provider-data gates

Document deletion is a policy-gated lifecycle operation. Once the retention policy is approved, it removes the original,
canonical chunks, Graphiti episodes, caches, and idempotency records in an auditable sequence, while retaining a minimal
non-content audit record of who read which document and when. V10 determines what the current system actually removes;
the delete operation must not be enabled for pilot data until engineering and legal agree on retention and recovery.

Real client data is blocked until legal and security validate Gemini's terms, retention, training settings, and provider
data handling for extraction, embeddings, and generation. Synthetic or approved fixtures are the only allowed data while
that gate is unresolved.

## Risks / Trade-offs

- [Cross-tenant disclosure during migration] → Run read-only backfill checks, dual-predicate isolation tests, and keep
  the Saul read kill switch off until two-tenant CI is green.
- [Legacy checkpoints cannot be upgraded] → Use explicit state-version adapters, quarantine only the affected thread,
  retain the old reader during rollout, and return an actionable typed resume error.
- [Grounding latency and cost] → Apply deterministic checks first, bound entailment and corrective retrieval with the
  run ledger, measure p95/tokens, and allow an internal demo mode that visibly remains `not_assessed`.
- [Graphiti outage reduces relationship recall] → Treat it as coverage degradation, preserve canonical chunk search,
  enforce timeouts, and disclose availability in the report rather than broadening scope.
- [Status/event disagreement] → Commit document status and outbox event in one transaction where possible; make event
  consumers idempotent and use age-based stuck detection as a safety net.
- [Tool registry drift] → Assert the role/tool binding table at construction and in CI; unknown tools fail loudly.
- [Metadata or party mixing across contracts] → Run V8 before implementation, detect boundaries before metadata,
  carry `contract_index`, and fail the ingestion gold set when a finding lacks contract identity.
- [Fidelity loss or OCR degradation] → Compare source/chunk coverage and order, expose typed warning reasons, and show
  `completed_with_warnings` before Saul analysis starts.
- [Unbounded large-file spend] → Use a data-informed upload ceiling and per-document cost/latency/task budgets with
  typed timeout/budget failures.
- [Deletion conflicts with retention policy or provider handling] → Keep D1/L1 as explicit pilot gates and enable no
  real-client-data or destructive delete path until legal/security decisions are recorded.

## Migration Plan

1. **Phase 0 (read-only verification):** perform V1–V7 plus V8 classification/metadata scope, V9 OCR configuration and
   quality signals, and V10 delete inventory. No production behavior changes; record whether multi-contract handling,
   OCR warnings, and deletion/audit semantics are confirmed or inferred.
2. **Phase 1 (security hotfix):** deploy `not_assessed`, server-derived authority, scoped precedent SQL, the
   data-informed upload limit and per-document budgets, and permanent two-tenant fixtures. Abort rollout if any
   unauthorized read, false pass, unbounded task, or oversized parse occurs.
3. **Phase 2 (schema/state):** add tenant/membership tables and 1:1 backfill; in the same migration change document
   deduplication from `(user_id, sha256)` to `(tenant_id, sha256)`; add provenance and document statuses; deploy the
   versioned state upgrader and cache/idempotency key changes. Dual-read old scope, one-write new scope; verify migrated
   counts, dedup behavior, and checkpoint resume before enabling new sessions.
4. **Phase 3 (ingestion hardening):** before evidence work, implement the V8-selected contract path, contract indexes,
   fidelity gate, OCR/table/section warnings, ingestion prompt-injection fixtures, per-document budgets, and I4's
   source-to-chunk gold set. The Saul gateway must show `completed_with_warnings` reasons before analysis starts.
5. **Phase 4 (retrieval/evidence):** enable shared `retrieve`, typed evidence tools, handles, page/provenance and
   contract-index output, timeout/degradation paths, role-specific bindings, O1 counters/alerts, and retention.
   Shadow-compare old/new ranked IDs and run seeded outage and forged-handle suites.
6. **Phase 5 (grounded Saul):** enable chunk adapter, typed findings/coverage, blocking grounding and reviewer checks;
   keep `persist_memory` flag-off until approved-output tests and both ingestion/answer gold-set baselines meet gates.
   Run A/B retrieval and injection/cross-scope red-team experiments.
7. **Phase 6 (cleanup):** after A4, R1, V3, parity, consumer audits, and H1's port-or-discard decisions, remove
   legacy `/ingestion`/phase-3 graph, hybrid precedent tool, stale helpers, and old references. Return 410 first if a
   consumer is discovered; do not delete the Phase-3 graph until its canonicalization/contextualization/merge/fan-out
   decisions have evidence links.
8. **Rollback/kill:** disable new Saul reads/grounding or persistence by feature flag, route to read-only queued status,
   and keep old checkpoint reader and legacy retrieval adapter during the migration window. Roll back code and feature
   flags before destructive cleanup; retain tenant/provenance columns and audit data. Kill the pilot on any cross-scope
   read, forged handle success, role-revocation bypass, false compliant pass, missing terminal event, or p95/cost
   breach beyond the approved budget, ingestion warning-rate alert, or provider governance failure. Real client data
   remains blocked until L1; D1 remains a pilot prerequisite pending retention/audit input.

## Open Questions

- Whether legal/customer review permits real client contracts in the gold set can be resolved by the gold-set owner
  before evaluation data is committed; synthetic fixtures remain the default.
- The exact Redis/Celery event-correlation field can be selected during implementation if it preserves idempotent
  terminal notification and does not change any spec or migration boundary.
- SME staffing and adjudication workflow can be chosen before enabling the judged metrics; no approval gate is removed
  if SME availability is delayed.
- The real maximum contract size must be selected from V8 corpus measurements and approved as configuration before S4;
  implementation must not silently retain the old 10 MiB assumption.
- The legal retention period, delete recovery policy, and minimum non-content read-audit fields for D1 require legal
  input after V10. Until decided, deletion is not enabled for pilot data.
- The Docling OCR confidence/coverage fields available in this deployment determine the exact I2 warning thresholds;
  V9 must resolve the signal names before those thresholds are implemented.
- Gemini provider terms, retention, and training settings require legal/security sign-off (L1) before real contracts;
  this gate cannot be resolved by engineering alone.
