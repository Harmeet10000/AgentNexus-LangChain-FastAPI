> Change class (pick one): **S** single-file fix / config / bump / docs · **M** feature in one module · **L** cross-cutting (multi-module, migration, security, public API)

**L** — this change cuts a live HTTP surface (`/search/ask`, `/legal/ask`) over to a new serving implementation, restructures how the retrieval graph is compiled and provisioned, and rewires Agent Saul's `qna` node to a new engine. It crosses `features/documents` (queries, retrieval graph, HTTP router), `features/agent_saul` (graph factory, nodes, budget), lifespan provisioning, and observability.

## Why

The application runs the same plan→retrieve→rerank→grade→generate machine twice: the inline body of `DocumentQueryService.ask` serves production, while the compiled retrieval graph (`features/documents/retrieval/graph/`) is quarantined duplication with zero external callers — and the two have already drifted (edge-aware grounding, version-aware cache keys, the evidence-policy filter, and the coverage/warning envelope exist only inline). Decision D6 (`PLAN-agent-tools.md`, "D6 — the compiled retrieval graph as the qna engine (decided 2026-10-31)") resolves this: mount the compiled graph as the single served retrieval implementation and expose it to Agent Saul's `qna` node as its engine — never as a tool. The D-then-E sequencing from the retired plan's step-6 matrix is abandoned in favour of cutover gated by a parity suite.

## What Changes

- **Parity port first (the cutover blocker):** port every inline-only capability into the graph nodes — edge-aware grounding (`fetch_relationship_edges` + `expand_along_edges` + edge-citation provenance from hydrated rows), version-aware answer cache keys (document version + pipeline version in the scope key), the `require_graphiti_verified` evidence-policy filter, the evidence-policy-correct empty-scope fallback, warning merges, the truthful coverage envelope, preserved access-audit recording, and the bounded grade-loop rewrite (2 iterations) with the injected grader seam.
- **A parity suite that dies on drift:** the same queries run through the compiled graph and recorded inline-path fakes must produce identical response shapes, citation provenance, fallback behaviour, and cache-key derivation; the suite stays as a permanent regression gate.
- **HTTP cutover:** `/search/ask` and `/legal/ask` are served by the compiled graph, compiled once per process with request-scoped collaborators supplied per invocation (per the `compiled-graph-lifecycle` capability). The inline `ask` body is deleted once parity is green — the one deletion the retired plan permits.
- **Agent Saul exposure:** `make_qna_node` becomes the caller of the compiled graph — node-internal subgraph invocation (explicit state mapping in and out, per the `deep_research` precedent), never a tool in any role's toolset. The graph's initial state is constructed from server-owned runtime context (tenant, `allowed_doc_ids`, version scope) — model-supplied identity is forbidden. qna's answers reach the transcript through the existing WebSocket event vocabulary with evidence-handle citations.
- **Budget:** the qna engine's graph invocation draws from Agent Saul's budget ledger (`domain/budget.py`) with per-stage spend attribution; the grade loop stays bounded at 2 iterations.
- **Observability:** per-stage spans across plan→retrieve→fusion→rerank→grade→generate (first slice of G6) and a decision log event per grade outcome (the future escape-rate metric).
- **Docs debt in passing:** the false prose in `legacy_ingest_dependencies.py` (the "commented out by decision (D17)" claim contradicted by the live lifespan provisioning) is corrected, and the retired plan's step-6 residue is marked DONE.

## Scope / Non-Goals

In scope: the parity port and suite, the compiled-graph provisioning change, the HTTP cutover and inline deletion, the qna engine wiring, budget attribution, and the first observability slice described above.

Out of scope (explicit): G4 graphiti-as-fourth-fused-branch (registration point reserved at the fusion call site); G5 small-to-big retrieval; G7 feedback capture; re-ingestion exposure (still gated by the retired plan's deferred preconditions 3–5); the Phase E quarantine ledger (separate item); answer-quality/latency pre-evaluation (abandoned with the D-then-E sequencing); and any change to role toolsets (the engine is deliberately not a tool).

## Capabilities

### New Capabilities

- `qna-retrieval-engine`: Agent Saul's `qna` node invoking the compiled retrieval graph as its grounded-answer engine — node-internal subgraph invocation (never a tool), server-owned context construction, explicit state mapping, budget-ledger draw with per-stage attribution, and transcript delivery through the existing event vocabulary.

### Modified Capabilities

- `legal-corpus-retrieval`: the ask endpoints are served by the compiled retrieval graph with per-endpoint evidence policies preserved; answer cache keys become version-aware; responses carry the truthful coverage and warning envelope.
- `agentic-retrieval-loop`: edge-aware grounding becomes part of the served loop with citation provenance from hydrated rows; the compiled loop is parity-gated against the inline contract; per-stage execution becomes observable (spans + grade decision log).

## Impact

Affected code: `src/app/features/documents/application/queries/documents.py` (inline `ask`/`_ask` deletion, `build_retrieval_graph`/`ask_via_retrieval_graph` retarget, edge-grounding helpers move into graph nodes), `src/app/features/documents/retrieval/graph/` (nodes, state, graph factory gain the ported capabilities and a context schema for per-invocation collaborators), `src/app/features/documents/api/http/router.py` (`/search/ask`, `/legal/ask`), documents HTTP dependencies (graph provisioning), lifespan (compile-once provisioning), `src/app/features/agent_saul/graph/factory.py` and `graph/nodes.py` (`make_qna_node`, `make_deep_research_node` as the structural precedent), `src/app/features/agent_saul/domain/budget.py` (per-stage spend attribution), and `src/app/features/documents/api/http/legacy_ingest_dependencies.py` (docstring truth fix). Public API contracts of the ask endpoints are preserved by the parity gate — no breaking request/response change is intended. Normative inputs: `PLAN-agent-tools.md` §D6 (exposure design) and `openspec/changes/archive/2026-10-31-plan-bulletproof-ingestion.md` (RETIRED — its step-6 decision matrix and recorded parity gaps are the risk register for this change).

## Risks

- **Behaviour change on a live endpoint** → the parity suite is the cutover gate: identical response shapes, citation provenance, fallback behaviour, and cache-key derivation are asserted before any router is retargeted, and the suite survives as a drift killer.
- **Tenant bypass through the new Saul exposure** → V6 context construction: the retrieval state is built only from server-owned values; a propagation test asserts a model-supplied tenant/scope never reaches the graph, and the empty-scope case returns the fallback envelope rather than an unscoped search.
- **State-schema drift between Saul and the retrieval graph** → explicit mapping in and out, no shared state keys, only answer + evidence-handle metadata entering the checkpoint (the `deep_research` precedent).
- **Orchestration overhead on the hot path** → the mounted graph runs without a checkpointer (decided in design); per-ask state writes are avoided, and the parity suite includes a latency sanity check.
- **Compile-once refactor breaks callers** → the graph factory gains a context schema for per-invocation collaborators; the provisioning failure path degrades to a typed unavailability rather than aborting startup.
