> Change class: **L**. The proposal covers *why* and *what*; this covers *how*. Reference the proposal — do not restate it.

## Context

Decision D6 in `PLAN-agent-tools.md` ("the compiled retrieval graph as the qna engine (decided 2026-10-31)") supersedes the retired `openspec/changes/archive/2026-10-31-plan-bulletproof-ingestion.md` step-6 sequencing: mount the compiled graph (approach C) as the single served retrieval implementation and Agent Saul's `qna` engine. The D→E path (extract a shared core, then a measured cutover) is abandoned; a parity suite is the safety net instead of a pre-evaluation. The retired plan's step-6 decision matrix and its recorded parity gaps are the risk register for this change.

Current state (verified on disk, 2026-10-31):

- The served path is the inline `ask`/`_ask` in `src/app/features/documents/application/queries/documents.py` (~826/~859), wrapped by `_record_access` for audit.
- The compiled graph lives in `src/app/features/documents/retrieval/graph/` (`graph.py`, `nodes.py`, `state.py`) and has zero external callers; `ask_via_retrieval_graph` (~698) compiles it per request and is itself unmounted.
- Known parity gaps (the retired plan's register, re-verified): edge-aware grounding exists only inline (`_edge_grounding_for_reranked` ~1277, `_resolved_edge_blocks` ~1357, `_edge_citations` ~1384); the graph's `_answer_cache_key` (`graph/nodes.py` ~693) keys on rewritten query + doc filter + contract index only, while the inline `_build_answer_cache_key` (~1633) adds tenant, user, resolved document versions, pipeline version, and evidence policy; the graph's filter params omit `require_graphiti_verified`; the empty-scope fallback in `ask_via_retrieval_graph` hardcodes `mechanically_verified`; the graph computes no coverage counts and no warning merges; the graph's grader node lacks the injected grader seam (`_grade_context`'s dual path, temperature 0.1); and `ask_via_retrieval_graph` never performs the access-audit recording.
- `qna` today (`agent_saul/graph/nodes.py` ~626) is a schema-locked single-shot structured call (`QnAOutput`) with a clarification interrupt; `make_deep_research_node` (~2780) is the structural precedent for a node wrapping a bounded subgraph with explicit state mapping.
- `compiled-graph-lifecycle` binds all compiled graphs: compile at most once per process, collaborators supplied per invocation, provisioning failure degrades rather than aborting startup. The current per-request `build_retrieval_graph` call and the node factories' closure capture do not satisfy this for a mounted, first-class surface.

## Goals / Non-Goals

**Goals:**

- One implementation of plan→retrieve→rerank→grade→generate, served to both HTTP consumers and Saul's qna node.
- The parity suite as the cutover gate and permanent drift killer.
- Lifecycle-correct provisioning of the mounted graph.
- Per-stage spend and span attribution laid down now so G6 (tracing/budgets) lands on data that already exists.

**Non-Goals:**

- No retrieval-quality work: no G4 graphiti-as-branch, no G5 small-to-big, no new fusion weights or reranking changes.
- No checkpointer on the mounted graph (see Decisions).
- No new WebSocket event vocabulary; no change to role toolsets.
- No re-ingestion exposure and no Phase E quarantine ledger.

## Decisions

### 1. Mount the graph; do not extract a shared core (and do not re-litigate)

The step-6 matrix's recommendation was D→E→C/B. The owner decided (2026-10-31) to skip D and E: approach C directly, gated by parity tests. This design records that decision as settled; extraction is rejected because the graph is the intended survivor and the parity suite pins its behaviour at least as tightly as a shared core would. The inline body's deletion is the one deletion the retired plan permits, and it happens only after the parity suite is green.

### 2. Port the inline-only capabilities into the graph nodes, not around them

Each parity gap is ported into the node that owns its stage, so the graph — not the service method — becomes the behaviour:

| Parity gap (inline location) | Lands in |
|---|---|
| Edge-aware grounding: `fetch_relationship_edges` + `expand_along_edges`, span resolution, edge citations (`documents.py` ~1277–1427) | New `edge_grounding` node between `post_process` and `context_grader` (fail-closed, additive); edge citations joined in the generate/finalise stage |
| Version-aware cache keys (`_build_answer_cache_key` ~1633) | `query_analyzer` (read) and `generate` (write) use the scope-version-aware derivation from `state.document_versions` + pipeline version |
| `require_graphiti_verified` filter | `hybrid_retrieval` node's filter params, from a new state field set by the caller's evidence policy |
| Evidence-policy-correct empty-scope fallback | The graph's scope/fallback logic reads the evidence policy field instead of hardcoding `mechanically_verified` |
| Coverage + warning merges (`_warnings_from_rows`, `_merge_warning_lists`) | Rows-derived warnings carried through retrieval state; coverage computed at finalisation from hydrated rows and scope-unavailable documents |
| Injected grader seam (`_grade_context` dual path, temperature 0.1) | `context_grader` node receives the optional grader through invocation context; seam-absent degrade preserved verbatim |
| Access-audit recording (`ask` wrapper) | Stays at the service boundary: the mounted service method records reads around the graph invocation — the graph must never record audit itself |

Adding the `edge_grounding` node changes the graph topology, so the node/edge shape assertion required by `agentic-retrieval-loop` ("The graph's node and edge shape is asserted") is updated in the same change.

### 3. Compile once per process; collaborators per invocation

`build_retrieval_graph` is called once at lifespan startup and the compiled graph is published on `app.state`. Node factories stop closing over request-scoped collaborators; the `StateGraph` gains a context schema (`StateGraph(RetrievalState, context_schema=...)`) carrying the repository, cache client, language models, relationship-graph client, reranker, and optional grader seam, supplied per `ainvoke(..., context=...)`. This satisfies `compiled-graph-lifecycle` (no captured session/repository; provisioning failure degrades to a typed unavailability, which the ask endpoints and qna both surface). Alternative rejected: keeping per-request compilation — simpler, but a mounted first-class surface compiled per request is exactly the violation the lifecycle capability exists to prevent, and compiling once is where the per-node tracing affordance pays off.

### 4. qna wiring: node-internal invocation with explicit state mapping (reference 10, pattern 1)

`make_qna_node` receives the graph provider (process-scoped compiled graph) and the ledger-bearing state; it maps `LegalAgentState` → `RetrievalState` (query from the transcript, tenant/`allowed_doc_ids`/version scope from server-owned runtime context), invokes, and maps `GeneratedAnswer` + citations → answer message, evidence-handle registrations, and coverage/errors. No shared state keys: the mapping names every field. The clarification-interrupt behaviour of today's qna remains for non-grounded intents; grounded intents take the engine path. A construction test asserts the initial retrieval state's contents (the V6 propagation test); a mapping test asserts nothing crosses by key sharing.

### 5. Budget: pre-check, per-stage charge, ledger stays in Saul

The graph does not depend on `agent_saul`'s ledger. Instead the graph surfaces per-stage stage-usage entries in its result state; the qna node pre-checks the ledger before invocation, then charges `MODEL_CALLS`/`RETRIEVAL_CALLS` per the reported stage usage. The grade loop's cap of 2 iterations is unchanged in the graph. Alternative rejected: passing the ledger into the graph via context — it would invert the layering (documents importing agent_saul's domain) and make HTTP-served asks carry an agent concept.

### 6. No checkpointer on the mounted graph

The retired plan's decision inputs name the risk: a checkpointer turns each ask into per-node state writes. A request-scoped ask has no resumability need; qna's invocations are bounded sub-runs inside Saul's own checkpointed thread. The mounted graph compiles without a checkpointer; attaching one is deferred until a use case (HITL mid-retrieval, resumable asks) exists.

### 7. Observability slice: spans + grade decision log

Each node emits a span attributed to its stage (plan, retrieve, fusion, rerank, grade, generate — plus the new edge-grounding stage); the grader emits a structured decision-log event per verdict (outcome, iteration, escaped-to-fallback flag). This is the first slice of G6; the escape-rate metric itself is later work and is explicitly out of scope.

## Risks / Trade-offs

- [Live-endpoint behaviour change on cutover] → The parity suite gates the router retarget; identical response shapes, citation provenance, fallback behaviour, and cache-key derivation are asserted against recorded inline fakes before any router changes, and the suite survives cutover.
- [Compile-once refactor touches every node factory] → Mechanical conversion to context-schema reads; the existing node unit tests keep passing against an explicitly-supplied context, and the lifecycle scenarios (compile-once, per-invocation collaborators) get dedicated tests.
- [Tenant bypass through the new Saul exposure] → V6 construction from server-owned context only; the propagation test asserts model-supplied identity never reaches retrieval state; empty scope drives the fallback envelope.
- [State-mapping drift as schemas evolve] → The explicit mapping is the only crossing point and a test asserts no shared keys; only answer + handle metadata enters the checkpoint (deep-research precedent).
- [Orchestration overhead on the hot path] → No checkpointer; LangGraph dispatch overhead is marginal against LLM/reranker latency; the parity suite includes a latency sanity comparison.
- [Parity suite ossifies a buggy contract] → The suite pins the *inline* contract as recorded; known-inline defects stay listable, and any deliberate divergence requires a conscious contract change in the recorded fakes, not silent drift.

## Migration Plan

1. Port parity capabilities into graph nodes; node/edge shape assertion updated; no serving change yet.
2. Land the parity suite (graph vs recorded inline fakes); gate green.
3. Provision the compiled graph in lifespan (compile-once, context schema); `ask_via_retrieval_graph` switches to the provisioned graph.
4. Wire `make_qna_node` to the engine (behind the parity gate being green).
5. Retarget `/search/ask` + `/legal/ask` to the graph-served service path.
6. Delete the inline `ask`/`_ask` bodies; correct `legacy_ingest_dependencies.py` docstring prose.
7. Update `openspec/changes/archive/2026-10-31-plan-bulletproof-ingestion.md`'s retirement header (step 6 → DONE) and note it on issue #107 if a CLI is available.

Rollback: the router retarget is a single wiring change — reverting step 5 restores the (still present until step 6) inline path. After step 6, rollback is a revert commit, not a runtime flag; the parity suite is what makes step 6 safe.

## Open Questions

None blocking. Two deliberate non-decisions recorded: the checkpointer stays off (Decision 6) until a use case exists, and the escape-rate metric is derived later from the decision-log events specified here.
