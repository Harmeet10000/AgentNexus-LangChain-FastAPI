# PLAN — Agent Saul Tool Set (todo 242)

**Date:** 2026-10-27 · **Owner:** Harmeet · **Status:** D0 implemented; D1–D4 sequenced
**Companion:** `openspec/changes/archive/2026-10-31-plan-bulletproof-ingestion.md` (pipeline) · this file (agent capability)
**Inputs:** user interview (2026-10-27) + verified inventory + `openspec/specs/agent-tool-registry/`
+ harden change Phase 6 + `.github/skills/langchain-langgraph/references/03` and `12`.

---

## 0. Decisions from the interview (these bind everything below)

1. **Mission, in priority order:** (1) analyze one contract deeply, (2) cross-document
   portfolio traversal, (3) conversational analyst.
2. **Write authority:** approval-gated — the two implemented-but-unwired memory writers
   get wired **behind HITL approval**, never autonomous.
3. **Exposure policy:** tiny projection — ≤7 tools visible per role per turn; the larger
   capability catalog lives server-side.
4. **Deliverable scope:** plan + tool-set changes + defect fixes (D0 done in this pass).

---

## 1. Verified state (2026-10-27)

| Layer | What exists | Verdict |
|---|---|---|
| Risk agent tools | `transfer_to_orchestrator`, `query_knowledge_graph`, `get_obligation_chain`, `search_evidence`, `read_evidence_by_id` (5 visible) | ✅ Already a tiny projection |
| Compliance agent tools | `transfer_to_orchestrator`, `transfer_to_risk`, `search_evidence`, `read_evidence_by_id`, `get_ingestion_state` (5 visible) | ✅ Already a tiny projection |
| Deep research | scoped `EvidenceService` adapter (handle-based; legacy `search_legal_precedents` alias retained for compliance only) | ✅ Matches harden change — upgraded to a bounded tool **subgraph** in D5 |
| Authorization seam | `ToolRuntimeContext` → `trusted_tool_context()`; server-owned `user_id/tenant_id/allowed_doc_ids`; hidden `InjectedToolArg` runtime | ✅ The right seam — extend it, don't replace it |
| Registry | single authoritative registry (`ai/tools/registry.py`), explicit `register_default_tools()`, spec-compliant; old `AgentToolBundle` name collision retired | ✅ Resolved since the last inventory |
| Global registry contents | `web_search`, 3×`transfer_to_*`, `shell_tool`, `read_file`, `write_file`, `list_directory`, `file_search` — **bound to no agent** | ⚠️ Latent: registered, never bound |
| Middleware | `LLMToolSelectorMiddleware`, summarization, retries, HITL, guardrails available | ✅ Sufficient |
| Memory writers | `write_clause_episodes_to_graphiti`, `write_final_report_to_memory` implemented, **unwired** | ⏭ D1 |

**Defects found by the earlier inventory — status after re-verification:**

| # | Defect | Status |
|---|--------|--------|
| 1 | Shell/filesystem tools claim capability grants but never inspect context | **Confirmed live → fixed in D0** |
| 2 | `shell_tool`: arbitrary command, caller-controlled `cwd`, no policy | **Confirmed live → fixed in D0** |
| 3 | File tools accept unrestricted paths, no workspace jail, symlink-blind | **Confirmed live → fixed in D0** |
| 4 | Two registries (base.registry vs registry.py) | **Resolved** — bundle docstring records the retirement; spec mandates one registry |
| 5 | `WebSearchInput` inherits `BaseTool` instead of `BaseModel` | **Confirmed live → fixed in D0** |
| 6 | Tags encode category only — no authority/side-effect/capability metadata | **Confirmed live → fixed in D0 (minimal), completed in D4** |
| 7 | Registry says what's declared, not what each role receives | By design now — factory's per-role bindings are the assignment of record (documented in §2) |
| 8 | Stale comment pointing at `app.retrieval.graphiti.registry` | Resolved by the bundle-docstring rewrite |
| 9 | Tool selector treated as authorization | Principle P1 below; no code change needed (selector only sees an already-authorized subset) |

---

## 2. The tool set — mission by mission

### Principle P1 — visibility is not authority
The selector middleware solves *relevance*; the *executor* re-checks authority. A tool the
model can see but that the runtime would refuse is a failed design. Enforcement lives in
the tool body via `trusted_tool_context` — never in what is bound or selected.

### Principle P2 — asymmetric impact comes from traversal, not more search
The current six tools are all *point* lookups. The capability that changes what Saul can
*do* (not just how fast) is **aggregation across documents** — obligations, conflicts,
party webs. Every one of those is a graph-edge query. Portfolio tools are therefore
**gated on todo 136** (relationship extraction, `openspec/changes/archive/2026-10-31-plan-bulletproof-ingestion.md` Phase D).
No portfolio tool ships before edges exist — a traversal tool over an edgeless graph
returns empty successes and teaches the model to answer from nothing.

### Principle P3 — one writer per fact, and every write approved
Memory writes go through one HITL-gated path with idempotency keys. The agent proposes;
the human approves; the writer is idempotent. (Mirrors the harden change: "persist memory
only after authorized approval".)

### Principle P4 — bounded returns
Every tool returns a typed envelope with bounded size: snippets + **handles**, never raw
documents; `get_ingestion_state`-style typed statuses. Handles are the currency —
`read_evidence_by_id` is the only thing that materializes content. This is what keeps the
context window flat as the corpus grows.

### Principle P5 — egress is a capability
Web search sends model-chosen text to a third party. For a legal corpus that is a
confidentiality surface, not just a tool. It requires an explicit grant like the shell.

### Target table (per role, visible per turn)

| Role | Today (5) | Add (D-order) | Never |
|---|---|---|---|
| risk | handoff, knowledge-graph query, obligation chain, search evidence, read evidence | `get_ingestion_state` (D3 parity); portfolio traversal tools **after edges** (D2) | shell/filesystem/web — not a specialist concern |
| compliance | handoff ×2, search evidence, read evidence, ingestion state | `retrieve_statute_section` stays; portfolio comparison **after edges** (D2) | same |
| deep_research | scoped evidence adapter | D5: one tool wrapping a bounded plan→search→read→synthesize subgraph | raw SQL / unscoped precedent search |
| orchestrator/qna | structured-output nodes (no tools by design) | — | tool-calling in schema-locked nodes |

Context math: 5–7 tools ≈ 1.5–2.5k tokens of schema per role — the projection stays flat
as the catalog grows. New capability = new catalog entry + a policy row, **not** a bigger
model surface.

---

## 3. Architecture: catalog → assignment → projection

Three layers, each already half-present — this plan only makes them explicit:

```
ai/tools/registry.py  (catalog: registration + authority tags)      ← spec's single registry
        │  register_default_tools() — explicit, idempotent
        ▼
AgentToolBundle (assignment: frozen per-role allocation, build-once) ← already exists
        │  factory.py binds ≤7 per role at graph compile time
        ▼
ToolRuntime.context (projection + authority at execution)            ← trusted_tool_context
           capabilities: granted subset; default deny
```

Rule: **no agent ever receives a tool whose authority the runtime cannot enforce.**
The `ToolRuntimeContext` TypedDict gains `capabilities: tuple[str, ...]`; every
system/filesystem/web tool refuses without its grant. Adding a dangerous tool to the
catalog is now safe by construction — binding it does nothing until a policy grants it.

---

## 4. D0 — defect remediation (implemented 2026-10-27)

| Fix | File | Behavior after |
|---|---|---|
| Capability grants enforced | `ai/tools/shell.py` (all 5 tools), via new `_require_capability` + `ToolRuntimeContext.capabilities` | Default deny; no runtime / no grant → typed `ToolResult.fail`, no execution |
| Shell policy | `shell_tool` | Requires grant `system.shell.execute` **and** an explicit `cwd` inside the configured jail |
| Workspace jail | `shell.py` + new `TOOL_WORKSPACE_ROOTS` setting | All path-taking tools resolve-and-verify against jail roots; unset roots ⇒ refuse with an operator-directed message |
| Schema hygiene | `ai/tools/web_search.py` | `WebSearchInput(BaseModel)`; `WebSearchTool` unchanged |
| Egress control | `web_search.py` | Deferred to D4 — the class-based `BaseTool` needs the runtime seam migrated first; the `side-effect:egress` authority tag is already applied |
| Authority tags | `ai/tools/registry.py` | `capability:*` / `side-effect:*` tags added — selection behavior unchanged |

**Acceptance:** unit tests prove deny-by-default for all five tools (no runtime, empty
grants, grant-without-jail, jail-escape attempt, symlink escape); gates green:
`uv run ruff check --fix src/` · `uv run ty check src/` · `uv run pytest` · `ast-grep scan src/`.

---

## 5. Sequenced work after D0

**D1 — wire memory writers behind HITL (next session-sized change).**
`write_clause_episodes_to_graphiti` / `write_final_report_to_memory` become tools that
*stage* a proposal; the existing `plan_approval_node`/`human_review_node` HITL machinery
approves; the idempotency guard keys the write (`doc_id + content hash`). Acceptance:
no write path exists that skips approval; replayed approval writes once; transcripts show
the proposal. Aligns with harden change Phase 6 ("persist memory only after authorized
approval") — execute inside that change, don't fork it.

**D2 — portfolio tools (blocked on todo 136 / plan Phase D).**
After edges flow: `find_conflicting_clauses`, `cross_document_obligations`,
`party_obligation_map` — all thin Graphiti traversals returning handles, fused with the
evidence seam. This is the asymmetric-impact payload: mission 2 is impossible for any
chat-only tool. Ship each behind the tiny-projection cap (swap, don't add).

**D3 — `get_ingestion_state` for risk (parity, small).**
Risk conclusions must disclose incomplete documents. Compliance already has it.

**D4 — complete the catalog (larger refactor).**
Authority metadata as structured registry fields (not just tags); policy file mapping
role → granted capabilities; `LLMToolSelectorMiddleware` scoped *within* an already
authorized subset; JS/TS parity per reference 03 when agents exist there.

**D5 — deep_research as a bounded tool subgraph (decided 2026-10-28; after D3, independent of D4).**
Today deep_research is a single scoped `EvidenceService` adapter call. The 2026-10-25
review decision rebuilds it as a *plan-conditional adapter over `search_evidence` /
`read_evidence_by_id`* — which is a subgraph, exposed as exactly **one** tool. Reference 03
(`.github/skills/langchain-langgraph/references/03`) sets the rule for compiled-subgraph
tools: define a narrow external input/output boundary, never the internal state.

Shape, using reference 03 idioms:

- Compile a `DeepResearchState` subgraph: plan → `search_evidence` →
  `read_evidence_by_id` (handles only, P4) → synthesize, with a hard step/recursion
  budget.
- Expose via one `@tool` wrapper with an explicit Pydantic `args_schema`
  (`ConfigDict(extra="forbid")`): `objective` (+ optional `focus`). `allowed_doc_ids`
  and identity are **not** model arguments — they come from
  `ToolRuntime[ToolRuntimeContext]`, injected and hidden from the model schema.
- The wrapper *constructs* the subgraph's initial state from the server-owned runtime
  context. Context propagation into the subgraph is by explicit construction, never
  ambient — this closes check **V6** (ToolRuntime inside sub-agents) by design, and
  V6's propagation test becomes D5's first acceptance case.
- Read-only: no memory writers, and **no HITL inside the subgraph** — interrupts do
  not cross a tool boundary; approval stays on Saul's orchestration graph.
- Return a bounded envelope: synthesized summary + evidence handles + coverage +
  `status: completed | budget_exhausted | no_evidence | ingestion_incomplete`.
  Budget exhaustion is a typed result with partial handles — never a confident partial
  answer.
- Register in the single registry with authority tags (read-only); the deep_research
  role's projection stays at one visible tool.

Acceptance: deny-by-default without a valid runtime context; a handle issued in run N
fails membership in run N+1; the budget test returns `budget_exhausted` (not a
truncated "done"); every subgraph `read_evidence_by_id` resolves against the same
`ToolRuntimeContext` the wrapper received; gates green.

---

## 6. What NOT to build

- **No shell/filesystem tools for Saul.** Legal analysis needs zero host access; the
  grants exist so the *global* registry is safe if another consumer ever binds them.
- **No model-visible catalog browsing.** The model sees its projection, period.
- **No autonomous memory writes** even with idempotency — decided (interview #2).
- **No portfolio tools before edges** (P2) and **no retrieval-graph tools before
  `/search/ask` routes through the graph** (one-implementation rule).

---

## For the chosen ones

- **The grant model's real payoff is future-proofing, not current security.** Nothing
  binds `shell_tool` today, so D0 changes no live behavior — it changes what a *mistake*
  costs later. Binding a dangerous tool stops being an incident and becomes a no-op
  until a policy grants it. Defense at the executor survives every registry, middleware,
  and prompt refactor above it.
- **`total=False` TypedDict cuts both ways.** The same permissiveness that let a disjoint
  state dict type-check clean (the old 503 bug) also makes `capabilities` invisible to
  `ty`. That's why D0's tests assert the *deny* path for every tool — the type system
  will not hold this line; only the tests will.
- **Jails fail on symlinks, so resolve twice.** `Path.resolve()` before the containment
  check, and the check re-runs on the *resolved* path — a symlink from inside the jail to
  `/etc` must not pass. The test for this is the one that matters.
- **Deny messages are operator documentation.** A refusal that says "grant
  `filesystem.read` is missing from tool runtime context; configure TOOL_WORKSPACE_ROOTS"
  turns the next debugging session into a grep. Failures that explain their own fix get
  fixed; failures that just say "denied" get worked around by widening grants.
- **The context-window algebra is the product constraint.** Every visible tool costs
  schema tokens on *every turn*; every hidden catalog entry costs nothing. "Tiny
  projection, big catalog" is not a security nicety — it's what lets the tool surface
  grow 10× while the prompt stays constant. That asymmetry is the whole game.

---

## D6 — the compiled retrieval graph as the qna engine (decided 2026-10-31)

**Decision (supersedes the D-then-E unification sequencing):** mount the compiled
retrieval graph (`features/documents/retrieval/graph/`, builder
`application/queries/documents.py::build_retrieval_graph`) as the served retrieval
implementation — both for the HTTP ask endpoints and as Agent Saul's grounded-answer
engine — and expose it to Saul **as a node, not a tool**.

### Which node requires it: `qna` — and only qna

The orchestrator already routes grounded-answer intents to the schema-locked `qna`
node (no tools by design). The retrieval graph is exactly what qna was missing: a
bounded plan → retrieve-fused → rerank → grade → generate machine with citations.
qna's current single-shot structured generation becomes the graph's generate stage.

**Exposure pattern: node-internal subgraph invocation (reference 10, pattern 1) —
deliberately NOT a tool.**

- A tool is a model-chosen capability with narrow I/O; a full RAG pipeline is a
  sub-agent with rich internal state (grade loop, rewrite, budget) and a citable
  answer. Reference 11's supervisor→worker handoff is the pattern; reference 03's
  compiled-subgraph-tool rule (narrow boundary, runtime-injected identity) governs
  the *boundary*, while reference 10's invoke-in-node governs the *wiring* — agent
  state and retrieval state have different schemas, so the node maps state in and
  out explicitly.
- **Zero toolset bloat:** risk, compliance, and orchestrator toolsets are unchanged
  (P-visibility math holds). Routing to grounded answers is a conditional edge /
  existing handoff, not a new tool.
- Precedent consistency: deep_research (D5) is likewise a node wrapping a bounded
  subgraph. Two heavy capabilities, two dedicated nodes, flat toolsets.

### Hard requirements (each a spec task with an acceptance test)

1. **V6 context construction:** the node builds the graph's initial state from the
   server-owned `ToolRuntimeContext` (tenant, allowed_doc_ids, version scope) —
   never model-supplied, never ambient. Tenant-bypass is the failure mode this
   closes.
2. **Port inline-only capabilities into the graph before cutover:** edge-aware
   grounding (#110: `fetch_relationship_edges` + `expand_along_edges` + edge
   citations), version-aware cache keys, fallback answer, warning merges. The graph
   must reach behavioral parity with the inline path — the parity suite from the
   existing grade-loop tests is the acceptance gate.
3. **Budget:** the node draws from Saul's ledger; the grade loop stays bounded
   (2 iterations, matching inline); per-stage spend attributed (feeds G6).
4. **HTTP cutover:** `/search/ask` + `/legal/ask` route through the same compiled
   graph; the inline `ask` is deleted once parity is proven (the one deletion the
   bulletproof plan permits). One implementation, two consumers (HTTP + Saul).
5. **Transcript/events:** qna emits the answer through the existing WS event
   vocabulary; citations carry evidence handles per the citation contract.

See `openspec/changes/archive/2026-10-31-plan-bulletproof-ingestion.md` (retired
2026-10-31, archived) for the step-6 decision matrix
and the completion audit that motivated mounting over extraction.
