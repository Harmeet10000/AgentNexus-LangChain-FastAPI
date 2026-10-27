## ADR: Server-derived authorization context at the tool boundary

- **Date:** 2026-10-25
- **Change:** harden-agent-saul-evidence-pipeline

**Status:** Proposed

## Context

Agent Saul runs concurrently over WebSockets and resumes after HITL pauses. Client-provided permissions, reviewer
roles, document IDs, or mutable process-global context can therefore turn a model input into an authority grant.

## Decision

Build an immutable authorization context from authenticated claims and current tenant membership at start and resume,
and pass it explicitly through runtime/tool construction. Tools ignore model-supplied identity and scope.

## Rationale / Alternatives

Explicit runtime context is concurrency-safe and makes every repository predicate auditable. Trusting DTO fields is an
authority-confusion vulnerability; a global request context is unsafe across concurrent WebSocket and worker runs.

## Consequences

Every tool and sub-agent boundary must accept runtime context, and role revocation takes effect on resume. A fallback
tool-closure adapter is permitted if the runtime cannot enter compiled sub-agents, but model-visible authority is never
an alternative. The tenant/membership schema and audit events become long-lived contracts.

## ADR: Canonical chunks are evidence truth; Graphiti is coverage infrastructure

- **Date:** 2026-10-25
- **Change:** harden-agent-saul-evidence-pipeline

**Status:** Proposed

## Context

Queued ingestion persists chunks while Graphiti can be delayed, incomplete, or unavailable. Treating relationship
indexing as truth makes valid uploaded evidence disappear or turns an outage into a false negative.

## Decision

All quoted evidence comes from tenant-scoped canonical chunks with document/pipeline provenance. Graphiti may narrow,
supplement, and report relationship coverage, but cannot by itself verify a legal claim.

## Rationale / Alternatives

The canonical chunk store is durable and already supports the queued ingestion contract. A Graphiti-first or dual-truth
model would make outage semantics ambiguous and allow relation availability to masquerade as legal correctness.

## Consequences

Evidence can be cited while graph coverage is incomplete, provided deterministic and entailment grounding succeeds;
reports must disclose graph availability. Graphiti timeouts and failures are explicit degradation paths, and graph
replacement/version semantics remain deferred.

## ADR: Opaque run-scoped evidence handles instead of text in checkpoints

- **Date:** 2026-10-25
- **Change:** harden-agent-saul-evidence-pipeline

**Status:** Proposed

## Context

Large legal documents and HITL pauses make raw text in LangGraph checkpoints expensive, difficult to retain safely, and
vulnerable to stale or cross-thread reuse.

## Decision

Retrieval issues opaque handles bound to run, thread, tenant, document, and version. State stores handles and typed
references; a read operation revalidates authorization and fetches canonical text at call time.

## Rationale / Alternatives

Handles preserve small, resumable state while retaining server authority and version checks. Storing text trades away
retention safety and cannot prove that a quote remains in the current document version; Graphiti-issued handles would
couple evidence truth to an optional backend.

## Consequences

Handle storage, expiry, forged-handle tests, and read-time revalidation become required infrastructure. Exact offsets
are intentionally deferred; quoted spans are checked as substrings of the canonical chunk.

## ADR: One bounded retrieval service for every Saul evidence path

- **Date:** 2026-10-25
- **Change:** harden-agent-saul-evidence-pipeline

**Status:** Proposed

## Context

Search, RAG, inline ask, and retrieval graphs can otherwise drift in ranking, timeout, scope, and outage semantics.
Corrective loops also need a common budget and a reliable stop condition.

## Decision

Provide one scoped `retrieve` contract with shared ranking/fusion, explicit provider timeouts, typed degradation,
coverage, provenance, and a per-run budget. All callers use it; legacy variants are removed only after parity evidence.

## Rationale / Alternatives

One service makes identical queries deterministic and makes a timeout or authorization failure observable everywhere.
Keeping parallel algorithms is operationally tempting but creates silent quality and security divergence; an unbounded
retrieval graph is rejected for cost and termination reasons.

## Consequences

Retrieval callers must accept typed envelopes and cannot interpret empty results as failure. The service becomes a stable
cross-feature interface, and any graph topology change requires a measured A/B decision.

## ADR: Contract identity and fidelity precede legal evidence

- **Date:** 2026-10-25
- **Change:** harden-agent-saul-evidence-pipeline

**Status:** Proposed

## Context

A PDF may contain multiple contracts, and OCR, tables, annexures, or clause ordering can be lost before retrieval. A
single metadata pass can mix parties or governing law, while retrieval relevance cannot detect source loss.

## Decision

Run V8/V9 discovery first. When multiple contracts are confirmed, detect boundaries before metadata extraction and carry
`contract_index` through chunks, graph episodes, evidence, findings, and reports. Independently gate publication with
source-to-chunk coverage/order and typed fidelity warnings, including available OCR signals.

## Rationale / Alternatives

Contract-local identity is safer than assuming one contract per file or adding LLM re-segmentation blindly. A fidelity
gate is needed because a highly relevant answer from incomplete source text is still unsafe. The exact OCR thresholds and
size ceiling remain configuration decisions driven by measured corpus data.

## Consequences

Ingestion gets a separate source-to-chunk gold set and can produce `completed_with_warnings` before Saul analysis. Every
downstream citation must carry contract identity when applicable; incomplete boundaries block or downgrade findings.

## ADR: Deletion and provider handling are explicit pilot gates

- **Date:** 2026-10-25
- **Change:** harden-agent-saul-evidence-pipeline

**Status:** Proposed

## Context

Deleting a legal document has multiple copies (original, chunks, graph episodes, caches, idempotency records) and may
conflict with retention/recovery obligations. Contract text also reaches Gemini for extraction, embeddings, and
generation, so provider retention/training terms matter before client data is used.

## Decision

Keep deletion disabled until V10 inventory and legal retention/recovery input are resolved, then remove every derived
artifact while retaining policy-permitted non-content read audit. Keep real client data blocked until legal and security
approve provider data handling, retention, and training settings.

## Rationale / Alternatives

Silent best-effort deletion creates an unverifiable legal hold gap; enabling client data before provider review creates a
governance breach that code cannot repair. A staged gate preserves synthetic/internal testing without guessing policy.

## Consequences

D1 and L1 are pilot prerequisites owned jointly by engineering, legal, and security. The implementation must support
typed policy-gate failures, idempotent deletion, and content-free access audit records; unresolved policy questions remain
explicit rather than being hidden as defaults.

## ADR: Native read-only tools and approval-gated mutations

- **Date:** 2026-10-25
- **Change:** harden-agent-saul-evidence-pipeline

**Status:** Proposed

## Context

The agent must be able to search and read its authorized corpus during analysis, but model output must not become an
upload, resume, approval, or persistence authority. MCP is intentionally deferred.

## Decision

Bind native `@tool` functions only. Read-only tools execute under server-derived runtime context; mutating operations
are not exposed or executed until an authorized human approval completes, with role checks repeated on resume. Uploads
remain HTTP-only and pass document identifiers into the agent.

## Rationale / Alternatives

Separating read and mutation gates keeps analysis fluid while making the approval boundary explicit and auditable.
Model-callable mutation or MCP exposure would add an authority surface outside this change's validated contract.

## Consequences

The registry, graph factory, and tests must enforce the native binding table and reject any model-supplied authority
arguments. Future MCP support must define an equivalent server-context and approval contract before exposure.
