> Change class (pick one): **S** single-file fix / config / bump / docs · **M** feature in one module · **L** cross-cutting (multi-module, migration, security, public API)

**Class: L** — multi-module (three `ToolRegistry` classes, three `ToolResult` classes, two prompt seams, agent
factories, lifespan-adjacent dependency surface), a retarget of raw SQL onto a different schema and a different
full-text engine, and a correctness boundary where a legal product currently reports infrastructure failure as
legal fact.

## Why

The agent tool layer is unreachable three times over, and while it is unreachable it has accumulated a defect that
would be a compliance incident the moment it runs: two tools report *"this section of law does not exist"* when the
underlying corpus is simply unavailable. This change makes the layer **honest and singular** before anything makes
it live — one registry of record, one tool-result shape, one idempotency contract, and unavailability that can never
be mistaken for absence.

Three findings no scout reported are the thesis, and each is independently sufficient to prove the layer has never
executed:

1. **The tools are never bound to the agents.** All three `create_agent` calls in
   `shared/langgraph_layer/agent_saul/factory.py:114,120,126` pass `tools=[]` with `# TODO` comments, while
   `build_saul_graph` (`agent_saul/graph.py:91`) faithfully threads a tool bundle down to them. The bundle is
   constructed, passed, and discarded.
2. **The surviving registry is empty at runtime.** `shared/langchain_layer/agents/tools/shell.py` — the only module
   using `@register_tool` (`:41,96,114,132,174`) — has **zero importers**, so the module-level singleton at
   `tools/base.py:99` never receives a registration. Adopting it naively converts today's silent `None` into a
   `KeyError` for *every* tool name, not zero. The survivor must be populated before it is adopted.
3. **The package exports the loser.** `shared/langchain_layer/agents/tools/__init__.py:7-12` re-exports
   `ToolRegistry` from `.registry`, not from `.base` — the opposite of what was previously reported. The D6.1
   survivor (`tools/base.py:58`) is today **not re-exported by its own package at all**, so it is the harder of the
   two classes to reach. Changing that line is a behaviour change for every importer of the package symbol, not a
   no-op.

Add the confirmed live defects: `agents/factory.py:146` calls `.get_tool(...)` on a class that defines only `.get`
(an `AttributeError` on first use), and `features/agent_saul/dependencies.py:40-41,45` read `app.state` attributes
that nothing assigns — an unhandled `AttributeError` 500 on already-mounted router surface.

## What Changes

- **One registry of record.** The `langchain_layer` tool registry (D6/D6.1) becomes the single one, populated
  explicitly rather than by import side effect, with tag-based selection preserved. The duplicate registry module is
  deleted; the third class of the same name in the Graphiti layer is **renamed, not deleted** — its file is live and
  has four importers (D6.1).
- **BREAKING (internal import surface):** `...agents.tools.ToolRegistry` changes which class it names, and resolving
  an unknown tool name changes from returning nothing to failing loudly.
- **One tool-result shape, with availability as a first-class signal.** Two duplicate definitions are removed; the
  survivor gains an explicit "backend unavailable" state distinct from "not found".
- **The invisible-failure register is closed.** A database error in statute retrieval, precedent search, and the
  clause-vector stub can no longer be rendered as absence, and a sufficiency verdict can no longer be computed from
  a partial source set as though it were complete. The docstring sentence that licensed this
  (`search_legal_precedents.py:179-180`) is deleted.
- **Idempotency keys become explicit about what identifies a call** — structural identifiers only for
  side-effecting tools, canonicalised query content plus structural scope for read/search tools. Nodes replay from
  their first line after an `interrupt`, so a write keyed on content double-writes on a re-worded retry.
- **One prompt seam** with a deterministic section order (standing instructions and output contract first, evidence
  in the middle with highest-salience items at its head and tail, task restatement last), and serialized data
  payloads that survive rendering byte-exact — today the mandated compact-serialization format collides with prompt
  brace substitution and raises at format time.
- **Structured output and citation enforcement.** The three agent calls that pass no response format get one; the
  output models that carry legal assertions reject an assertion with no citation at **validation** time.
- **Bounded retries owned by middleware**, never swallowing a graph pause.
- **Agent-to-agent handoff gets one envelope and an explicit router**, plus the state-schema hydration step the
  state module already documents but which does not exist, plus a single schema-version constant replacing two
  independent literals.
- **Statute and precedent retrieval is retargeted off tables that do not exist.** `statutes`, `clauses`,
  `parent_documents`, `entities`, `relationships`, `events`, and `memory_versions` have **no migration and no rows**
  — the live database is stamped at `0004` with only billing tables present. Under D15 the target is the unified
  `chunks`. There is nothing to migrate and zero rows anywhere; the full-text engine changes from `tsvector`/`ts_rank`
  to the BM25 path that already works in `features/search/`, which is harvested rather than rewritten.
- **The deliberately unwired graph stays unwired.** Per D17 the commented lifespan wiring is *not* a regression and
  is *not* restored here. What changes is that the gap answers with a clean service-unavailable response instead of
  an internal error, and that nothing in this change makes re-enabling harder.

## Scope / Non-Goals

**In scope:** `shared/langchain_layer/agents/tools/**`, `shared/langchain_layer/agents/factory.py`,
`shared/langchain_layer/prompts.py`, `shared/langchain_layer/middlewares/`, `shared/rag/graphiti/registry.py`
(rename only), `shared/langgraph_layer/agent_saul/{factory,graph,state}.py`,
`features/agent_saul/dependencies.py`, and the two statute-facing raw-SQL tools.

**Out of scope, deliberately:**

- **`open_deep_search/`** (D7). It defines a second, *async* function with the same name as one being unified
  (`open_deep_search/utils.py:260`, consumed at `graph.py:46,281,344,391`). Nothing crosses today; the duplication is
  recorded as a future hazard and the names are **not** unified.
- **Restoring the lifespan wiring** (D17) and mounting anything new.
- **A bespoke structured message bus / agent communication protocol** — the platform's native handoff answers it.
- **The Accept / Retry / Escalate escalation state machine** — only its cheap half (schema-validated output) is here.
- **Memory construction**, which change 4 owns, including the self-documented memory stub whose two called methods
  do not exist.
- **Schema DDL.** This change ships no migration; it consumes change 2's consolidated target and change 0's head
  merge. Full Non-Goals with reasons are in `design.md`.

## Capabilities

Checked `openspec/specs/` first (21 capabilities). None covers a tool registry, tool-result normalization,
idempotency, prompt assembly, agent handoff, or corpus retrieval. Two were evaluated for reuse and rejected on fit:
**`llm-injection`** is about injecting a chat-model client into services (constructor/parameter injection), not about
assembling prompt sections or preserving payloads through rendering; **`typed-exception-handling`** governs *which
exception type is caught and annotated*, whereas the requirements here govern *what the tool reports to the model* —
those two specs are **cited in `design.md`, not edited** (both are also pre-existing validation failures).

### New Capabilities
- `agent-tool-registry`: one registry of record for agent tools, with tag-based selection and loud failure on an
  unknown tool name.
- `agent-tool-contract`: one tool-result envelope in which backend unavailability is never reported as absence, and
  one idempotency-key contract split by tool kind.
- `legal-corpus-retrieval`: statute and precedent retrieval resolves against the unified corpus, and a query that
  cannot reach the corpus says so.
- `agent-prompt-assembly`: one prompt-assembly seam with deterministic section ordering and byte-exact survival of
  serialized data payloads.
- `agent-structured-output`: agent outputs are schema-validated, an assertion without a citation is rejected, and
  usage metadata is retained wherever it is captured.
- `agent-runtime-resilience`: bounded model and tool retries that never swallow a pause, and an unavailable agent
  graph that yields a service-unavailable response rather than an internal error.
- `agent-state-handoff`: one handoff envelope with an explicit router, and a state-schema version that is resolved
  or refused but never ignored.

### Modified Capabilities

None.

## Impact

- **Internal import surface (breaking):** the package-level tool-registry symbol changes identity; unknown tool names
  now raise instead of returning nothing.
- **Cache:** every persisted idempotency entry changes shape and every key changes value. The persisted envelope
  forbids unknown fields, so a new-schema entry read by old code raises. Mitigated by a key-prefix bump and one cold
  cache — not by a dual read.
- **Runtime:** no new dependency. No migration. No new mounted route. Startup behaviour is unchanged (D17).
- **Tests:** these modules are at 0% coverage, so eleven steps carry mandatory new tests; the suite's summary line is
  the signal, since coverage gating makes it exit non-zero regardless.
- **Cross-change:** the importer rewrite here is a **blocking predecessor** of change 0's deletion of the shadow stub
  tree — the Graphiti registry module eager-imports the four tool factories at module scope, so deleting the stubs
  first raises `ImportError` before FastAPI is constructed.

## Risks

- **[Change 0 deletes the shadow stub tree before this change's importer rewrite lands → the app does not import.]**
  → The rewrite is this change's first task and change 0's deletion task cites it; paired restore is a revert of the
  deletion.
- **[Adopting an empty registry turns a silent miss into a hard failure for every tool.]** → Populate and prove
  non-empty first, with an executable proof, before any consumer moves.
- **[Making the layer honest surfaces failures that were previously invisible.]** → That is the intent, and D17 keeps
  the graph unwired, so the exposure is bounded to the dependency surface, which now answers 503.
- **[Change 2 slips and the corpus retarget cannot be written.]** → The retarget floats; the honesty work ships
  without it and the tools report unavailability truthfully in the meantime.
- **[A retry wrapper swallows the human-in-the-loop pause and the graph silently stops pausing.]** → A dedicated
  test asserts the pause propagates through the retry seam; it is the one test that cannot be dropped for time.
