> Change class (pick one): **S** single-file fix / config / bump / docs · **M** feature in one module · **L** cross-cutting (multi-module, migration, security, public API)

**L** — this change crosses the Agent Saul graph, queued document ingestion, authorization, persistence, retrieval, tools, state migration, and evaluation. It changes a security boundary and requires a tenant/data migration.

## Why

Agent Saul can currently treat client-supplied authority and incomplete retrieval state as trustworthy, while the precedent path can cross user scope and final analysis can emit unsupported assurances. The existing WebSocket/HITL graph and uploaded-document corpus are the right foundation, but they need a server-derived authorization boundary, deterministic evidence lifecycle, truthful coverage/status semantics, and blocking grounding before the system can be piloted safely.

## What Changes

- Ship security hotfixes first: make risk/compliance explicitly `not_assessed` until supported, remove client authority over permissions and reviewer role, constrain precedent search by tenant/user and allowed documents, enforce a documented data-informed upload limit and extension allowlist before buffering, add per-document cost/latency/task budgets, and add permanent two-tenant isolation tests.
- Keep the existing Agent Saul WebSocket/HITL graph and queued `/documents` ingestion as the canonical path; replace any Saul-side LLM re-segmentation with an adapter over persisted chunks and never concatenate documents.
- Introduce tenant membership and legal roles (`analyst`, `reviewer`, `admin`) with server-derived, immutable `AuthorizationContext` passed through `ToolRuntime`; scope every read to the tenant and an explicit `allowed_doc_ids` set, while retaining user identity for audit.
- Keep the tool boundary native (`@tool` functions only): read-only tools run under authorized runtime context, while mutating operations are approval-gated; uploads remain HTTP-only and the agent receives document IDs rather than upload/resume/approval controls.
- Replace single-document session scope with `allowed_doc_ids` plus `primary_doc_id`, require ready document statuses, version checkpoints, and migrate old saved state safely.
- Harden ingestion before evidence work: verify multi-contract classification boundaries, carry `contract_index` through chunks/Graphiti/evidence, gate fidelity with source coverage/order/table/section/OCR signals, add untrusted-document prompt-injection fixtures, and maintain an ingestion-specific gold set.
- Add chunk/pipeline provenance, truthful ingestion terminal states and outbox/UI notifications, cache/idempotency scope keys, bounded checkpoint retention, and dead-letter handling.
- Unify search, RAG, and inline ask behind one timeout- and budget-aware `retrieve` path. Add scoped `search_evidence`, handle-based `read_evidence_by_id`, and `get_ingestion_state` tools with typed result statuses; treat Graphiti as a relationship backend and coverage signal, never as evidence truth.
- Add opaque run-scoped evidence handles, typed findings/evidence references, a claim ledger and coverage report. Make deterministic span/version/scope checks and subsequent entailment verification a blocking grounding gate; disclose graph coverage and incomplete documents.
- Restrict tools per node, structure all tool failures, gate human approval and resume by current server role, record self-review/overrides, and persist memory only after authorized approval.
- Build ingestion and answer gold sets, red-team and two-tenant CI suites, deterministic/live retrieval baselines, degradation counters/alerts, and rollout/kill criteria. Harvest Phase-3 stages into explicit port-or-discard decisions before deleting obsolete retrieval/ingestion paths.
- Define document deletion and access-audit behavior (originals, chunks, Graphiti episodes, caches, idempotency records, and read logs) subject to legal retention input, and complete provider data-handling review before real client data.

## Scope / Non-Goals

In scope are the security boundary, tenant/document scope, queued ingestion lifecycle, canonical chunks, provenance, retrieval/evidence APIs, Saul state and grounding contracts, role-gated HITL, observability, migrations, tests, and staged rollout described above. The graph topology remains the existing WebSocket/HITL graph unless the bounded retrieval experiment proves a justified change.

Out of scope: legal sufficiency or compliance certification; a shared statute corpus; model-callable upload, resume, or approval; MCP exposure; changing Graphiti group IDs; Saul-side LLM segmentation; exact span offsets; re-ingestion/version replacement (deferred until stable IDs, cache invalidation, and Graphiti replacement semantics exist); `bm25_force_merge` changes before experiment evidence; and production-readiness claims without legal, security, and customer validation.

## Capabilities

### New Capabilities

- `agent-saul-evidence-security`: server-derived tenant/role authorization, explicit document scope, fail-closed tool access, upload guardrails, evidence-handle binding, and cross-tenant isolation.
- `agent-saul-grounded-findings`: typed findings, contract-aware evidence references, claim ledger, coverage, mandatory deterministic/entailment grounding, truthful abstention, and approval-gated finalization.
- `document-delete-and-access-audit`: policy-gated deletion of document artifacts and auditable document reads.

### Modified Capabilities

- `agent-saul-delegation`: preserve the existing transcript-visible WebSocket/HITL handoff contract while binding role-specific tools and authorization context.
- `agent-tool-registry`: enforce the role/tool binding table and prevent model-supplied identity or scope from granting access.
- `document-ingestion-pipeline`: make queued `/documents` ingestion canonical, enforce data-informed upload/resource limits, handle multi-contract files, gate fidelity, resist ingestion prompt injection, expose typed statuses, terminal events, and failure/dead-letter behavior.
- `document-retrieval-schema`: add tenant/document-version/pipeline provenance and the ready-state contract consumed by evidence retrieval.
- `legal-corpus-retrieval`: constrain corpus retrieval by tenant and allowed documents, unify ranking/fusion, return page/provenance, and report outage/coverage truthfully.
- `agentic-retrieval-loop`: replace ad hoc retrieval with one bounded `retrieve` path and record the experiment/decision on any graph removal or retention.
- `langgraph-checkpointing`: migrate Saul state to identifiers and opaque handles, version old checkpoints, and apply retention/pruning without raw document payloads.
- `retrieval-evaluation`: extend the measurable baseline with entailment, abstention, citation, scope-violation, latency/cost, injection, and role-revocation evaluations.

## Impact

Affected areas include `features/agent_saul`, `features/documents`, `features/auth`, `retrieval`, `resilience/idempotency`, `events/outbox`, `worker/jobs`, Alembic migrations, API/WS DTOs, graph factories/nodes, all legal retrieval tools, checkpoint configuration, deletion/audit infrastructure, provider governance, observability, and CI/evaluation fixtures. The public upload/session/tool contracts change (breaking client-supplied permission/reviewer fields and single-`doc_id` scope); a tenant-aware dedup migration, 1:1 tenant backfill, and state-version upgrade are required. No new external truth source is introduced: Graphiti remains optional/degradable relationship infrastructure.

## Risks

- A scope bug can disclose another tenant's legal corpus; mitigate with server-only context, fail-closed predicates, forged-handle tests, two-tenant fixtures, and a kill switch that disables Saul reads.
- A false positive can become a legal assurance; mitigate with `not_assessed`, mandatory grounding, unsupported/contradicted surfacing, human approval, and no persistence before approval.
- Migration or checkpoint incompatibility can strand active threads; mitigate with explicit versioned upgraders, dual-read/one-write rollout, pause/resume tests, and rollback before destructive cleanup.
- Graphiti outages or incomplete ingestion can be mistaken for no evidence; typed statuses, coverage signals, timeouts, and truthful UI/report disclaimers keep degradation observable.
- Large or multi-contract files can exceed cost/time budgets or mix parties and metadata; data-informed limits, per-document budgets, contract identity, fidelity gates, and ingestion gold-set tests keep the pipeline bounded and truthful.
- Deletion may conflict with legal retention obligations and provider handling may be unacceptable for real client data; legal/security decisions are explicit pilot gates, not assumptions hidden in implementation.

## Delta Gates

- Before any pilot: complete V8–V10, S1–S5, A1–A4, I1, I2, I4, D1, and L1, in addition to the previously listed retrieval and grounding gates.
- Before team-wide daily use: complete I3 and O1, the ingestion and answer gold-set baselines, and the existing CI/red-team suite.
- The critical path is Phase 0 (including V8–V10) → S1–S5 → A1–A4 → ingestion hardening (I1, I2, I4) → R1/R3/R4 → G1–G5 → experiments → H1 → Phase 5/6 deletion and cleanup.
