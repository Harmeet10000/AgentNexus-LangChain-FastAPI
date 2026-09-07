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

1. **The tools are never bound to the agents.** In `shared/langgraph_layer/agent_saul/factory.py`, all three
   `create_agent` calls (opening at `:114,120,126`) pass `tools=[]` at `:116,122,128` with `# TODO` comments, while
   `build_saul_graph` (`agent_saul/graph.py:91`) faithfully threads a tool bundle down to them. The bundle is
   constructed, passed, and discarded.
2. **The surviving registry is empty at runtime.** `shared/langchain_layer/agents/tools/shell.py` — the only module
   using `@register_tool` (`:41,96,114,132,174`) — has **zero importers**, so the module-level singleton at
   `tools/base.py:99` never receives a registration. Adopting it naively converts today's **unconditional
   `AttributeError`** into a `KeyError` for *every* tool name, not zero — the failure mode changes but stays total.
   (Not a "silent `None`": there is no silent miss anywhere, per the correction below.) The survivor must be populated
   before it is adopted.
3. **The package exports the loser.** `shared/langchain_layer/agents/tools/__init__.py:7-12` re-exports
   `ToolRegistry` from `.registry`, not from `.base` — the opposite of what was previously reported. The D6.1
   survivor (`tools/base.py:58`) is today **not re-exported by its own package at all**, so it is the harder of the
   two classes to reach. Changing that line is a behaviour change for every importer of the package symbol, not a
   no-op.

Add the confirmed live defects. `agents/factory.py:146` reads `get_tool_registry().get(t)`, and
`get_tool_registry()` returns the class at `tools/registry.py:9`, which defines `get_tools` / `get_tool` /
`get_search_tool` / `get_crawl_tool` and **no `get`** — so it is an **unconditional `AttributeError` today for every
string-named tool**. (Earlier drafts of this proposal had the two classes reversed, saying the call was `.get_tool(...)`
against a class defining only `.get`; the conclusion was right and the direction was wrong. The correction matters
because it removes a hazard that was being cited as justification: there is **no silent miss** anywhere. The
`return None` at `registry.py:24` has **zero reachable callers** — the class's only live uses are `get_tools()` at
`:55,:60` and `.get(t)` at `factory.py:146` — so no code path returns nothing for an unknown tool name.) Separately,
`features/agent_saul/dependencies.py:40-41,45` read `app.state` attributes that nothing assigns — an unhandled
`AttributeError` 500 on already-mounted router surface.

## What Changes

- **One registry of record.** The `langchain_layer` tool registry (D6/D6.1) becomes the single one, populated
  explicitly rather than by import side effect, with tag-based selection preserved. The duplicate registry module is
  deleted; the third class of the same name in the Graphiti layer is **renamed, not deleted** — its file is live and
  has four importers (D6.1).
- **BREAKING (internal import surface):** `...agents.tools.ToolRegistry` changes **which class it names**, and the
  named class has a **different method set** (`register`/`get`/`all`/`by_tags`/`by_names` instead of
  `get_tools`/`get_tool`/`get_search_tool`/`get_crawl_tool`). That symbol-identity change is the whole of the breaking
  surface. It is **not** a change in miss semantics: no reachable caller resolves a tool name through the loser's
  `get_tool`, so nothing today "returns nothing" for an unknown name to begin with — an earlier draft claimed that and
  it was false.
- **One tool-result shape, with availability as a first-class signal — four definitions collapse to one.** Three
  duplicates are removed, and the survivor gains an explicit "backend unavailable" state distinct from "not found". The
  fourth definition is `ToolOutput` (`langchain_layer/agents/tools/base.py:30`), which no scout reported and which is
  the worst of the four: it is named differently, so a gate matching `^class ToolResult` never sees it; it carries a
  `to_agent_string()` returning `f"ERROR: {self.error}"`; and all **13** of its use sites — in `tools/shell.py`, the
  five tools this change registers first — call that method, so those tools return an error **sentence** to the model
  rather than a result. Because the **deployed** `typed-exception-handling` spec names `ToolOutput.fail()` in five
  scenarios, removing it requires a `MODIFIED` delta on that spec rather than a citation.
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
`shared/langchain_layer/prompts.py`, `shared/langchain_layer/agents/middlewares/` (the bare
`shared/langchain_layer/middlewares/` path named in an earlier draft does not exist and would have had an implementer
create a second middleware package), `shared/rag/graphiti/registry.py`
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
  merge. The statute identity attributes and their point-lookup index are **provided** by change 2 under its ADR
  *"`documents` / `chunks` is the sole retrieval schema"* (Accepted); `legal-corpus-retrieval` states the requirement at
  the attribute level, names no column, and the retarget task is gated on change 2 by name.
- **Harvesting `shared/rag/rag_agent_advanced.py` into change 1.** The user decided the file is **relocated to
  `src/app/examples/`** (per `CLAUDE.md`), not harvested-then-deleted. Two losses accepted on the record: its
  `f"Search error: {e!s}"` anti-pattern (`:172,244,293,345,481` (re-measured; earlier drafts were off by three)) **survives, quarantined** — no task edits its bodies —
  and its iterative-RAG prior art (`search_with_self_reflection` `:353`, `expand_query_variations` `:52`) **stays
  unused**, so change 1 designs item 195 from scratch with no design note from it. No harvest task is written.

Full Non-Goals with reasons are in `design.md`.

## Capabilities

Checked `openspec/specs/` first (**20** capabilities — an earlier draft said 21). None covers a tool registry,
tool-result normalization, idempotency, prompt assembly, agent handoff, or corpus retrieval, so the seven below are new.

Two existing capabilities were evaluated for reuse. **`llm-injection`** is rejected on fit: its four requirements are
`SearchService` constructor injection of `llm: BaseChatModel`, document-function `llm` parameters, `_build_chat_model()`
called once in the dependency layer, and API back-compat — chat-model *dependency* injection end to end, with nothing
about prompt assembly, sections, or rendering.

**`typed-exception-handling` IS reused, with a `MODIFIED` delta.** An earlier draft rejected it on the ground that it
"governs *which* exception type is caught and annotated, whereas the requirements here govern *what the tool reports to
the model*". That distinction is falsified by the spec's own text: five of the six scenarios under
`### Requirement: Agent tools SHALL catch OS-level and library-specific exceptions`
(`openspec/specs/typed-exception-handling/spec.md:219,223,227,235,239`) prescribe the **return value**, and they name
**`ToolOutput`** — the fourth envelope this change deletes. Leaving it uncited would leave five scenarios of a deployed
spec pointing at a class that no longer exists. The delta is scoped to that single requirement; change 0's `asyncpg`
requirement and change 1's four added requirements are untouched. `pattern-matching-standard` remains **cited, not
edited** (see `design.md` D-12).

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

- `typed-exception-handling` (**MODIFIED**, one requirement): *"Agent tools SHALL catch OS-level and library-specific
  exceptions"*. Every caught-exception-type clause is unchanged and every scenario title is reproduced verbatim; the
  five `ToolOutput.fail()` return clauses become the surviving envelope's failure constructor, a catch site must return
  the envelope rather than a rendered string, and a corpus-unreachable failure must use the unavailability constructor.
  Requirement ownership is disjoint from the two sibling changes that also touch this spec (change 0 `MODIFIED` the
  `asyncpg` requirement; change 1 `ADDED` four).

## Impact

- **Internal import surface (breaking):** the package-level tool-registry symbol changes **identity** — it names a
  different class with a different method set. That is the whole breaking surface. Unknown tool names raise, which is
  the intended end state, but that is **not** a change from "returning nothing": nothing reachable returns nothing
  today, because `factory.py:146` cannot reach the loser's `get_tool` at all.
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
- **[Adopting an empty registry makes every tool resolution raise `KeyError`, so adoption looks like a fix while still
  failing for every name.]** → Populate and prove non-empty first, with an executable proof, before any consumer moves.
  (This risk is *not* "a silent miss becomes a hard failure" — there is no silent miss today; `factory.py:146` is an
  unconditional `AttributeError`. Adoption is the moment that line becomes reachable at all, which is exactly why
  population must precede it.)
- **[Making the layer honest surfaces failures that were previously invisible.]** → That is the intent, and D17 keeps
  the graph unwired, so the exposure is bounded to the dependency surface, which now answers 503.
- **[Change 2 slips and the corpus retarget cannot be written.]** → The retarget floats; the honesty work ships
  without it and the tools report unavailability truthfully in the meantime.
- **[A retry wrapper swallows the human-in-the-loop pause and the graph silently stops pausing.]** → A dedicated
  test asserts the pause propagates through the retry seam; it is the one test that cannot be dropped for time. It runs
  against a **throwaway two-node graph compiled with an in-memory saver, built inside the test** — not the application
  graph, which stays unwired under D17. `design.md` D-10 authorises that vehicle explicitly, because `interrupt` cannot
  pause without a checkpointer and the real one is never assigned, so read literally the proof discipline would have
  forbidden the one test it calls non-negotiable.
