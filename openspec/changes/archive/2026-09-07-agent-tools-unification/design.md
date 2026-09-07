> Change class: **L** — cross-cutting (multi-module, breaking internal import surface, a retarget onto a different schema and full-text engine, a correctness boundary in a legal product).
> The proposal covers *why* and *what*; this covers *how*. Reference the proposal - do not restate it.

## Context

The layer this change touches has never executed. Three independent proofs, each sufficient on its own:

| Proof | Evidence |
|---|---|
| Tools are never bound to agents | `shared/langgraph_layer/agent_saul/factory.py` — all three `create_agent` calls (opening at `:114,120,126`) pass `tools=[]` at `:116,122,128` with `# TODO: add ... when available  # noqa: FIX002`, while `build_saul_graph` (`agent_saul/graph.py:91`) threads a tool bundle down through `_build_graph_nodes(:95)` to them |
| The survivor registry is empty | `shared/langchain_layer/agents/tools/base.py:99` is a module-level `registry = ToolRegistry()`; the only module using `@register_tool` is `tools/shell.py` (`:41,96,114,132,174`) and it has **zero importers** |
| The builder has no callers | `shared/rag/graphiti/registry.py:98` `build_tool_registry` — zero callers; the lifespan block that would call it (`lifespan.py:234-247`) is commented out, deliberately (D17) |

Three more facts shape every decision below.

**The export inversion.** `shared/langchain_layer/agents/tools/__init__.py:7-12` re-exports `ToolRegistry` from
`.registry`, **not** from `.base`. The scout report claimed the opposite. So `...agents.tools.ToolRegistry` names the
**loser** class today, `base.py` is not re-exported by its own package at all, and the D6.1 survivor is the harder of
the two to reach. Correcting that line is a behaviour change for every importer of the package symbol.

**The fourth envelope.** `dispositions.md` Up#10 was corrected on 2026-08-18: there are **four** competing
tool-result envelope definitions, not three, and the fourth is named differently. `ToolOutput`
(`tools/base.py:30`) carries the same `success` / `data` / `error` / `metadata` shape with the same `ok()` / `fail()`
classmethods, plus a `to_agent_string()` (`:46`) whose failure branch returns `f"ERROR: {self.error}"` — the
string-as-error anti-pattern this change exists to remove, sitting in the package it is unifying. It has **13 use
sites**, all in `tools/shell.py` (`:4,18,68,71,106,108,111,126,129,145,155,158,216`), and every one of them calls
`.to_agent_string()`, so the five tools D-1 promotes into the registry of record today return **strings** to the
model, not envelopes. `ToolOutput` sits **28 lines** from the survivor `ToolResult` (`idempotency.py:34`) in the same
package. It is in scope: see **D-3** and **D-12**.

**The import cycle that breaks boot.** `shared/rag/graphiti/registry.py:40-45` eagerly imports the four `make_*_tool`
factories at module scope. Two of those modules import the shadow stub tree
(`get_obligation_chain.py:29`, `precedent_tools.py:21,22`). Change 0 deletes `shared/agents/**`. If it does so before
this change rewrites those three import lines, `import app.main` raises `ImportError` before FastAPI is constructed.
**This change's first task is therefore a blocking predecessor of a task in change 0.** D6.1 also settles that
`shared/rag/graphiti/registry.py` is **not deletable**: `:34-122` is live and its `ToolRegistry` at `:56` is consumed
by `agent_saul/graph.py:16,91` and `agents/factory.py:182,205`.

Database facts, from `docs/relay/findings-database.md` (live probe, not inference): the instance is stamped at
alembic `0004` while the entire document/vector/search branch was **never applied**. `statutes`, `clauses`,
`parent_documents`, `entities`, `relationships`, `events`, `memory_versions`, `documents`, `chunks`,
`search_documents`, `search_chunks` — **none exist and none ever have**. There are **zero rows anywhere** in this
subject area, so the corpus retarget migrates nothing. `pg_textsearch` 1.3.0 is available (not yet installed), so the
BM25 path in `features/search/` is the viable target engine. `precedent_tools.py:237` is a stub returning `[]`.

Coverage context: every module in scope is at **0%**. There is no regression net except the one this change writes,
which is why eleven of eighteen tasks carry mandatory tests and the two structurally riskiest carry executable proofs
that do not depend on the suite at all.

## Goals / Non-Goals

**Goals:**

- One registry of record, **populated before it is adopted** — the ordering is the whole safety argument.
- One tool-result envelope in which unavailability is a first-class field, not a string a caller must parse.
- Unavailability can never be rendered as absence, and a sufficiency verdict can never be computed from a partial
  source set as though it were whole.
- Idempotency identity that is explicit about structure versus content, because nodes replay from their first line.
- One prompt seam, with the ordering rule implemented once and data payloads that survive rendering byte-exact.
- Schema-validated agent output where an uncited assertion fails validation.
- Bounded retries at one seam that never swallow a pause.
- Measurable type-level gates: `ty` diagnostics **46 → ≤31** after the importer rewrite, **≤28** after registry
  adoption (see Migration Plan).

**Non-Goals** — each of these is a recorded gap, not an oversight:

- **`open_deep_search/` (D7).** No work scheduled. Recorded hazard: `open_deep_search/utils.py:260` defines a second,
  **async** `get_all_tools` taking a `RunnableConfig`, consumed at `open_deep_search/graph.py:46,281,344,391`. Nothing
  crosses today and the names are deliberately **not** unified; if that stack is ever merged, the collision is here in
  writing.
- **Item 67 — structured message bus / agent communication protocol / persistent shared state: DROPPED.** LangGraph's
  native handoff already answers it — an `AIMessage` carrying a `transfer_to_*` tool call with a router edge reading it
  (`brief:ref:1473-1479`). A bespoke bus on top is duplicate machinery. The gap is that cross-agent messaging has no
  transport of its own and inherits whatever the message channel provides.
- **Up#9's Accept / Retry / Escalate state machine: DEFERRED** (`dispositions.md` SPLIT). Only the cheap half —
  declared response schemas and retained usage accounting — is in scope. The gap: a failed validation is refused, but
  nothing decides *whether to retry, repair, or escalate to a human*; that subsystem is unbuilt.
- **Item 151 — `compact-middleware`, `langchain-collapse`, `langchain-cisco-aidefense`: DEFERRED.** Three new
  dependencies with no stated adoption criterion. Their overlap with the retry/middleware seam here is real and is
  recorded so that whoever revisits 151 knows the seam already exists (`middlewares/guardrails.py:49,159` uses
  `@wrap_model_call` today).
- **Item 194 — `headroom-ai` for compression: DEFERRED.** Adjacent to the prompt/serialization work here (both concern
  token budget) but a new dependency with no criterion. Not owned by this change.
- **Restoring the lifespan wiring (D17).** See decision **D-10**.
- **Memory construction.** Change 4 owns it. Recorded gap with a named failure mode: `MemoryManager`
  (`langchain_layer/agents/factory.py:69-74`) is a self-documented stub whose two called methods —
  `inject_long_term_context` (`:246`) and `save_session` (`:256`) — **do not exist**, guarded by
  `enable_long_term_memory` which defaults `True` (`:113`). Today `factory.py:146`'s `AttributeError` fires first and
  masks it; fixing `:146` here **unmasks a second `AttributeError`**. It is not a duplicate, so it is not in this
  unification workstream — but this change must not be read as having fixed `factory.py`.
- **Schema DDL and migrations.** This change ships none. It consumes change 0's head merge and change 2's consolidated
  target.
- **Mounting any route**, and **`features/search/` behaviour changes** beyond harvesting its BM25 + fusion path.
- **Editing `pattern-matching-standard`.** See decision **D-12**. (`typed-exception-handling` **is** edited, by a
  `MODIFIED` delta — D-12 was reversed.)
- **Harvesting `shared/rag/rag_agent_advanced.py`'s iterative-RAG algorithm into change 1.** The user decided this
  file is **relocated to `src/app/examples/`**, not harvested-then-deleted (Q1, closed below). Two losses are accepted
  on the record as a direct consequence, and neither is scheduled anywhere:
  - **The `f"Search error: {e!s}"` anti-pattern survives.** It remains at `:172,244,293,345,481` (re-measured; earlier drafts were off by three) — the exact
    string-as-error shape D-3 exists to remove. It is **quarantined**, not fixed: under `src/app/examples/` it no
    longer reads as production code, it retains its zero importers, and its tools `ImportError` on first call anyway
    (`from ingestion.embedder import create_embedder` at `:119,198,267,373` names a package that does not exist in this
    repo). No task in this change edits its bodies.
  - **The iterative-RAG prior art stays unused.** `search_with_self_reflection` (`:353`, grading at `:420`, query
    refinement at `:460`) and `expand_query_variations` (`:52`) are the repo's only prior art for iterative RAG, and
    `dispositions.md` routes agentic query rewriting to change 1's item 195. Because the file is moved rather than
    harvested, change 1 receives **no** design note from it and will design that item from scratch. This change writes
    **no** harvest task.

## Decisions

### D-1 — The registry survivor is `tools/base.py:58`, and it is **populated before it is adopted**

D6.1 names the survivor. What D6.1 could not know is that the survivor is **empty at runtime**. So the work splits
into two commits that must not be merged into one:

1. **Populate.** Register the five decorated tools from `shell.py` and tag `web_search` (`web_search.py:80`) and
   `crawl_url` (`crawl.py:114`) on the survivor, while `tools/registry.py` still serves its consumers unchanged. Prove
   non-empty with an executable one-liner, not a code read.
2. **Adopt.** Move consumers (`agents/factory.py:53,146`), fix the package export inversion, keep
   `get_all_tools`/`get_web_tools` as thin aliases over `registry.by_tags("web")` for one commit, then delete
   `tools/registry.py`.

Adopting first is a live hazard, but **not the one earlier drafts of this document described**, and the corrected
version matters because it changes what the ordering protects against. Verified 2026-08-18:

```python
# src/app/shared/langchain_layer/agents/factory.py:146
resolved_tools.append(get_tool_registry().get(t))
```

`get_tool_registry()` (`registry.py:45`) returns the **loser** class, and `registry.py:9`'s class defines
`get_tools` / `get_tool` / `get_search_tool` / `get_crawl_tool` and **no `get`**. So `factory.py:146` is an
**unconditional `AttributeError` today, for every string-named tool** — not a silent miss. The earlier framing here
and in the proposal ("calls `.get_tool(...)` on a class that defines only `.get`" / "returns `None` on miss, and the
survivor's `get` raises `KeyError`") had the two classes **reversed**, and the "silent `None` → `KeyError`"
transition it described **does not exist**: `registry.py:24`'s `return None` on miss has **zero reachable callers**
(the class's only live uses are `get_tools()` at `:55,:60` and `.get(t)` at `factory.py:146`), so no code path
returns nothing for an unknown tool name.

The real hazard, re-justified on the true state:

- **Adoption is the moment `factory.py:146` becomes reachable at all.** Today it cannot resolve any name. Pointing
  `get_tool_registry()` at the survivor makes `.get(t)` a valid call for the first time — and `base.py:73` raises
  `KeyError` on a miss. Against an **empty** registry every name is a miss, so adoption without population converts
  one unconditional failure (`AttributeError`) into another (`KeyError`) while appearing to be a fix. Populating
  first is what makes adoption an actual repair rather than a relabelled break.
- **Fail-fast is the intended end state, and it is only correct once the registry has contents.** `KeyError` on a
  genuinely unregistered name is the behaviour `agent-tool-registry` requires; `KeyError` on every name because
  nothing was ever registered is not.
- **The breaking surface is the package symbol's identity, not miss semantics.** `...agents.tools.ToolRegistry` will
  name a different class with a different method set. That is the real behaviour change for importers, and the
  proposal's Impact section is corrected to say so.


*Alternatives considered.* (a) **Adopt then populate** — rejected: guarantees a window where every tool resolution
raises. (b) **Keep both registries behind a facade** — rejected: D6 mandates one, and a facade preserves the two
divergent `get`/`get_tool` contracts that produced the current bug. (c) **Populate by importing `shell.py` from the
package `__init__`** — rejected in favour of **explicit registration**: importing a module nothing has ever imported to
obtain a decorator side effect makes package import order load-bearing, and would run that module's top-level code in
every process including Celery workers. Explicit registration is order-independent and greppable. This also closes the
plan's Fog #2 without needing to audit `shell.py` for import-time I/O.

**Contradiction resolved (was blocking).** An earlier draft of `agent-tool-registry` carried a requirement titled
*"The registry is populated deterministically at import"* whose scenario read "**WHEN** the agent tools package is
imported **and nothing else** — **THEN** the registry SHALL report a non-empty set". That is satisfiable **only** by
alternative (c), the import side effect this decision rejects, so the spec mandated the design D-1 refused and the
requirement title named it. D-1(c) stands and **the spec was corrected**, not the decision: the requirement is now
*"The registry is populated by explicit registration before any consumer resolves a tool"*, with scenarios that
(i) prove the entry point populates, (ii) prove that a bare package import registers **nothing**, (iii) require
idempotent re-registration, and (iv) require a pre-registration resolve to raise. Every one is provable by a
bootstrap unit test with no import-order dependency.

Concretely, the entry point registers the five `@register_tool`-decorated tools from `shell.py` **and** tags
`web_search` (`web_search.py:80`) and `crawl_url` (`crawl.py:114`) with `web` — neither is decorated today, and none
of `shell.py`'s tags (`system`, `shell`, `filesystem`, `read`, `write`, `list`, `search`) is `web`. That is what makes
`get_all_tools` / `get_web_tools` viable as thin aliases over `registry.by_tags("web")` (`base.py:82`) and what makes
the spec's "Web-capable tools are reachable by their tag" scenario true. Under explicit registration the decorators
in `shell.py` remain harmless — they register into the survivor when the entry point imports the module — but nothing
depends on that import happening implicitly.

### D-2 — The third same-named class is **renamed, not deleted**

`shared/rag/graphiti/registry.py:56` is a third class called `ToolRegistry` and a **different concept**: an immutable
Pydantic bundle of four pre-built tools, constructed once (`build_tool_registry` `:98-122`) and consumed as a value
object. D6.1 forbids deleting the file. It is renamed (`AgentToolBundle`) and its four importers updated
(`agent_saul/factory.py:10,182`, `agent_saul/graph.py:16,91`), so the repo has exactly one `ToolRegistry`. **The name
is fixed by ADR 2 in `adrs.md`** (Accepted 2026-08-18), because change 4 also works in `shared/rag/graphiti/` and
inherits whatever this change calls the class. The module
docstring (`registry.py:9,25`) is corrected in the same commit — it currently points at the deleted-stub import path and
at an `app.state.saul_graph` assignment that exists nowhere, making it the most misleading comment in the layer.

Sequence it **after** D-1's adoption, so no window exists in which one imported name has two meanings.

*Alternatives considered.* (a) **Delete the file and fold the bundle into the registry** — rejected: D6.1 says not
deletable, and a bundle-of-four is genuinely not a registry; collapsing them would put per-agent tool assignment inside
a global registry. (b) **Leave the name collision** — rejected: it is the reason three scouts disagreed about which
class survives. (c) **Rename the survivor instead** — rejected: the survivor is the one with 20+ potential importers and
the public package symbol.

### D-3 — Availability is a **field** on the result envelope, not a metadata key, and the collapse is **four → one**

The survivor envelope is `langchain_layer/agents/tools/idempotency.py:34` (`extra="forbid"`, `frozen=True`,
`success`/`data`/`error`/`metadata`, `ok()`/`fail()`). The collapse is **four definitions into one**, not three:

| # | Definition | Disposition | Migrates |
|---|---|---|---|
| 1 | `langchain_layer/agents/tools/idempotency.py:34` `ToolResult` | **survivor**, gains availability | — |
| 2 | `shared/agents/tools/idempotency.py:11` `ToolResult` | dies with change 0's deletion of the shadow tree | no caller |
| 3 | `shared/rag/document_processing/models.py:318` `ToolResult` | deleted; its **only** importer is `todo_temp.py:8`, which D11 deletes | no caller |
| 4 | `langchain_layer/agents/tools/base.py:30` `ToolOutput` | **deleted; 13 call sites in `tools/shell.py` rewritten onto the survivor** | 13 sites, one module |

Definition 4 is the one earlier drafts of this document missed, and it is the worst of the four rather than the most
harmless: it is the only one with a `to_agent_string()` (`base.py:46`) that renders failure as `f"ERROR: {self.error}"`,
every one of its 13 sites calls it, and those sites are the five tools D-1 step 1 promotes into the registry of
record. Leaving it would make `agent-tool-contract`'s first requirement false the day this change ships. It also
means the **deployed** `typed-exception-handling` spec names a class this change removes — see **D-12**.

Because the fourth definition carries a different class *name*, any gate matching `^class ToolResult` passes while it
survives. The gate is therefore `rg -c "class Tool(Result|Output)\b" src/` → **4 → 1**; see the gate table.

Add an explicit unavailability state and a third constructor. A retry, escalation, or "do not tell the user this law
doesn't exist" decision must not be spelled in a free-form dict.

*Alternatives considered.* (a) **Carry it in `metadata`** — technically works (`**meta` already flows there) and was
rejected: an unkeyed convention in a dict is exactly how the current defect survived review, and no type checker can
enforce it. (b) **Raise an exception for unavailability** — rejected: these values are returned to a model as tool
output; an exception either aborts the run or is caught and re-stringified, which is the anti-pattern being removed.
(c) **A separate result type per outcome** — rejected: the envelope is persisted as JSON and read back with a single
validator; a union would need a discriminator, which is the field being added anyway. (d) **Leave `ToolOutput` alone
and narrow the envelope requirement to exclude `shell.py`** — rejected: the five tools it serves are exactly the ones
D-1 registers, and the deployed `typed-exception-handling` spec already governs its `fail()` in five scenarios, so
"out of scope" would leave a deployed spec pointing at a class in the package this change is correcting.

**Serialization hazard, stated because it is a rolling-deploy trap.** The guard persists this envelope as JSON in Redis
and Postgres with a 30-day TTL (`idempotency.py:30`, `_POSTGRES_TTL_DAYS = 30`; `:31` is `_REDIS_KEY_PREFIX`, the line
that must **also** be bumped) and reads it back with `model_validate_json` (`:83`). `extra="forbid"` means
**new-schema rows read by old code raise**. Adding a defaulted field is forward-safe but not backward-safe.
Resolution: bump `_REDIS_KEY_PREFIX` in the same commit as the key-shape change (D-4) and accept one cold cache.
**No dual-read.**

### D-4 — Trap2 is honoured by **splitting the key contract by tool kind**

Trap2 as literally worded — "hash structural IDs (`clause_id`, `doc_id`), never content" — is right for half the call
sites and actively harmful for the other half. `make_key` (`idempotency.py:65-76`) already does the correct
cryptography: `hashlib.sha256(json.dumps({...}, sort_keys=True, default=str))` — deterministic, not the salted `hash()`
builtin. The defect is **what callers put in `input_data`**: `precedent_tools.py:82` passes `{"query", "user_id",
"num_results"}`.

| Tool kind | Key inputs | Why |
|---|---|---|
| read / search | step id, user id, **canonicalised** query (case-folded, whitespace-collapsed) **and** structural scope (`doc_id`, `clause_id`) | the query text *is* the cache identity; dropping it makes two different questions return each other's answers |
| write / side-effect | step id, user id, structural ids **only** (`clause_id`, `doc_id`, `episode_id`) — never content | a node replays **from its first line** after `interrupt` (`brief:ref:1628`); a content-keyed write double-writes on a reworded retry |

Enforced mechanically: `make_key` becomes keyword-only with an explicit `structural: dict` and an optional
`content: dict | None`; the write path (`graphiti/write_clause_episodes.py` is the guard's consumer — `:35` is the *import* of `IdempotencyGuard`; the `make_key` call is elsewhere in that file and the implementer must locate it rather than trusting `:35`) passes `content=None`.
A single opaque `input_data` dict cannot express the distinction, which is precisely why it drifted.

*Alternatives considered.* (a) **Trap2 literally, everywhere** — rejected: search-result cache collisions across
distinct questions, in a legal product. (b) **Content everywhere** — the status quo; rejected: duplicate graph writes on
replay. (c) **Document the convention without changing the signature** — rejected: an unenforced convention is what
exists now.

**This is a deviation from a dispositioned item, and it is flagged as one rather than absorbed.** `dispositions.md`
Trap2 words the rule as "never content" and calls it "a one-line rule inside the surviving `IdempotencyGuard`"; the
table above makes it **two** rules behind a keyword-only signature. The technical argument is above and this change
proceeds on it, but dispositions belong to the orchestrator: **if the literal wording was intended as binding, this
decision is the one to reverse**, and reversing it costs only the search-path row (the write path is Trap2 verbatim).
No other decision in this change depends on the split.

### D-5 — `MessagesState` is rejected as a vehicle; sub-todo (i)'s intent is honoured

Sub-todo (i) names `MessagesState`. It is not adopted. Three reasons, in order of force:

1. The reference corpus **never uses it** — a single descriptive mention (`brief:ref:1479`) — and is explicit that
   *"custom state schemas must be `TypedDict`… Pydantic models and dataclasses are no longer supported"*
   (`brief:ref:1341-1345`).
2. `LegalAgentState` (`agent_saul/state.py:317`) **already has the mandated shape**: a `TypedDict` with
   `messages: Annotated[list[BaseMessage], add_messages]` (`:329`) and `operator.add` sibling channels
   (`:343-345,367`). Adopting `MessagesState` would be a lateral rename that **loses** the sibling channels or forces
   them into a subclass for no gain.
3. What is actually missing is not the state class but the **handoff convention**. The documented form is a message
   naming the recipient plus a router edge that reads it (`brief:ref:1473-1479`); `agent_saul` instead routes on custom
   router functions via `add_conditional_edges` (`graph.py:50,55,66,74`) with **zero** `transfer_to_*` and zero
   `Command(goto=...)` **in `agent_saul`**.

   *Correction, 2026-08-18.* An earlier draft of this decision claimed zero `Command(goto=...)` **anywhere in the
   repo**. That is wrong: there are **13** such lines in `shared/langgraph_layer/open_deep_search/graph.py`
   (`Command(goto="supervisor" | "researcher" | "researcher_tools" | "compress_research" | "write_research_brief" |
   "__end__")`). The conclusion for `agent_saul` is unchanged — it has none — but the repo **does** have prior art for
   exactly the routing convention this decision adopts, and D7 puts `open_deep_search` out of scope as a reason not to
   *edit* it, never as a claim that it does not exist. Whoever implements the router rule should read those 13 sites
   first: they are the in-repo reference for the `Command(goto=...)` form, and matching their conventions is cheaper
   than inventing a second dialect.

So: build one construction helper for the handoff message and one router rule that reads it, set an explicit
`recursion_limit` (`brief:ref:1492` requires it; `ref:1471` warns there is no loop detection outside the supervisor
state), and **convert no state class**.

**The handoff helper is also exposed as the orchestrator's tool set.** `agent-tool-registry` requires every
tool-using role to be constructed non-empty, and the gate takes `tools=[]` in `agent_saul/factory.py` from 3 to 0 —
but the specs previously named tool sets for the **compliance** and **risk** roles only, leaving the orchestrator's
`tools=[]  # TODO: add delegation tools when available` (`factory.py:116`) with nothing to close it and the gate
unreachable. Resolution: the handoff construction helper is registered as one `transfer_to_<role>` tool per
delegable role, tagged `handoff`, and the orchestrator is constructed with `registry.by_tags("handoff")`. This is
`brief:ref:1473-1479`'s own convention — the recipient-naming message *is* a tool call — so it invents no mechanism
beyond the helper this decision already commits to, and it makes the third `tools=[]` closable honestly rather than
by exception.

*Alternatives considered.* (a) **Replace `LegalAgentState` with `MessagesState`** — rejected on (1)–(3). (b) **Subclass
`MessagesState` and re-declare the siblings** — rejected: same shape as today plus an import, zero behaviour gained.
(c) **Keep the bespoke routers and skip the envelope** — rejected: it is the half of (i) that is genuinely missing, and
without it every new agent pair invents its own routing predicate.

### D-6 — Middleware owns model and tool retries; `tenacity` stays at I/O-client boundaries

Sub-todo (j) asks for `tenacity`. It is installed (9.1.4) and already used correctly at I/O-client boundaries —
`kb_retry.py`, `connections/redis.py`, `razorpay_client.py`. It is **not** extended into graph nodes.

| Option | Pros | Cons |
|---|---|---|
| `tenacity` inside graph nodes (sub-todo (j) as written) | familiar; already a dependency; decorator is one line | **zero** mentions of `tenacity` / `RetryPolicy` / `.with_retry()` in the reference corpus (`brief:301-305`); `ref:1633` forbids a bare `try/except` around `interrupt`, which pauses **by raising** — `tenacity`'s default `retry_if_exception_type(Exception)` is exactly that catch-all, so HITL silently stops pausing; and because a node replays from its first line (`ref:1628`), a node-local attempt counter is not a checkpointed channel and the retry budget silently multiplies on every resume |
| **Middleware at the model/tool seam (chosen)** | the seam already exists in this repo — `middlewares/guardrails.py:49,159` uses `@wrap_model_call`; retry state lives outside node bodies so replay does not multiply it; pause propagation is testable in isolation; **and the tool half is already built and wired** (see below) | one more middleware to configure on the second factory |
| Both | — | two retry budgets composing multiplicatively, invisibly |

**Mechanism, version-pinned (Q2, closed 2026-08-18).** `handle_tool_errors` is **not reachable** through
`create_agent` at langchain 1.2.12 — no such parameter, and no `tool_node` parameter to inject one
(`langchain/agents/factory.py:673-691`). Nothing in this change may be written against the 0.2-era
`ToolNode(handle_tool_errors=…)` shape. The reachable seam is `@wrap_tool_call` middleware, and the purpose-built
implementation is **`ToolRetryMiddleware`** (`middleware/tool_retry.py:30`), whose default `on_failure="continue"`
(`:134`) turns an exhausted failure into a `ToolMessage(status="error")` (`:273-286`) rather than re-raising. This
matters because the library default does the opposite: `_default_handle_tool_errors`
(`langgraph/prebuilt/tool_node.py:379-387`) **re-raises everything except `ToolInvocationError`**.

**This is not greenfield, and the scope is smaller than the todo implies.** `ToolRetryMiddleware` is already
constructed at `guardrails.py:345` and `:369`, and `build_default_middleware_stack` (`:304`) already reaches
`create_agent` as `middleware=` from `shared/langchain_layer/agents/factory.py:152,188`. The **only** gap is the second
factory: `shared/langgraph_layer/agent_saul/factory.py` contains **zero** occurrences of `middleware`, so its three
agents have no retry seam and a raising tool aborts the run there. The task is therefore *install the existing stack on
the second factory*, not *design a retry policy*.

Sub-todo (j)'s **intent** — bounded retry with backoff — is honoured; its named vehicle is not the documented one.
**Change 1 also touches `tenacity`** (its I/O-boundary usage in the ingestion path). This decision does not contradict
that work: it *confirms* the boundary as tenacity's home and only bars it from graph-node interiors.

If the user overrides this, the override needs its own task proving `interrupt` still propagates through the
`tenacity`-wrapped call — that test is the entire risk.

### D-7 — The prompt seam is built in the **opposite direction** from the todo's framing, and the ordering rule lives in a **new kinded seam**, not in `build()`

Todo (1) frames this as "Template vs `ChatPromptTemplate`" and the brief framed `render_prompt_sections` as a
"competing helper". Measured, it is the **dominant** one: `render_prompt_sections` (`langchain_layer/prompts.py:145`)
has **27** external call sites — `agent_saul/prompts.py` 11, `open_deep_search/prompts.py` 8, `ingestion_kb/prompts.py`
4, `retrieval_kb/nodes.py` 3, `reconciliation/prompts.py` 1 — and returns a bare `str`. (Earlier drafts said 26; the
itemisation always summed to 27.) `SystemPromptParts` (`prompts.py:19`) — which owns `build()` (`:99`),
`Template(...).safe_substitute` (`:122`) and `to_chat_template()` (`:126`) — has 3 construction sites
(`agents/factory.py:171`, `agents/registry.py:103,150`) plus the module-level `AGENT_SYSTEM_PROMPT` in `prompts.py`
itself.

So: **do not migrate 27 sites.** That part of the original decision stands.

**What did not stand, and is corrected here (was blocking).** The original decision then said "implement the Up#6
ordering rule **once**, inside `build()`". Two measured facts make that impossible:

1. **`render_prompt_sections` cannot order anything.** Its signature is
   `def render_prompt_sections(*sections: tuple[str, str | None]) -> str` (`:145`) and its body appends
   `f"{label}\n{normalized}"` in argument order. It is **positional, label-agnostic and order-preserving** — it has no
   notion of section *kind*, so making `SystemPromptParts` consume it does not move ordering authority anywhere. Each
   of the 27 callers keeps owning its own order, by construction.
2. **`build()` has nothing to order.** It emits a **fixed set of seven named fields** — `IDENTITY`, `OBJECTIVE`,
   `CONTEXT POLICY`, `EXECUTION POLICY`, `CONSTRAINTS`, `UNCERTAINTY POLICY`, `EXAMPLES` (`:99-124`) — in hardcoded
   positional order. There is **no evidence field and no task-restatement field**. "Retrieved evidence in the middle
   with highest-salience at the head and tail" cannot be implemented against fields that do not exist, and the
   original decision costed none of that work.

A third objection is independent of both and decides the shape of the fix: `SystemPromptParts` builds a **system**
prompt. Putting retrieved evidence there is where Lost-in-the-Middle ordering is *least* applicable — a preamble that
is byte-identical every turn has no contested middle — and where prompt-prefix reuse is *worst*, because per-turn
evidence changes the prefix on every call.

**Resolution — three parts, and the spec was narrowed to match:**

- **`render_prompt_sections` is unchanged and remains the label-agnostic primitive.** All 27 callers keep working and
  none is migrated. It is explicitly **not** the ordering seam.
- **A new kinded assembly seam owns the ordering rule.** It accepts sections by *kind* — standing instruction, output
  contract, **ranked** evidence sequence, task restatement — sorts them into the Up#6 order, and delegates the
  rendering of each resulting section to `render_prompt_sections`. Ordering is therefore implemented exactly once, in
  the one place that knows what a section *is*. Evidence arrives as a ranked sequence rather than opaque prose,
  because head/tail salience placement is not derivable from a single string.
- **The split follows the cache boundary.** `SystemPromptParts.build()` keeps the standing instructions and output
  contract — the stable, reusable preamble, which is already what its seven fields are. The evidence block and the
  task restatement are assembled per turn by the new seam and injected as turn message content (which is also where
  D-8's payload injection already lands). The composite order across both is the Up#6 order.
- **Scope is stated, not implied.** `agent-prompt-assembly` no longer claims "No caller assembles prompt sections
  independently" — 27 callers do, deliberately. Its ordering requirement is scoped to prompts assembled through the
  seam, and the boundary is recorded in the spec as a gap rather than left as a false universal.

*Alternatives considered.* (a) **Migrate all 27 onto `SystemPromptParts`** — rejected: 27 edits in files with zero
coverage, to reach a helper with 3 adopters. (b) **Delete `SystemPromptParts`** — rejected: it owns the only path to a
chat template and the only variable-substitution seam. (c) **Leave both** — rejected: the ordering rule would then have
two implementations, which is how "Lost in the Middle" ordering silently stops holding. (d) **Give
`render_prompt_sections` a section-kind parameter** — rejected: it is a 27-caller signature change to a function whose
current contract ("render these labels in this order") is correct for those callers and is what they want. (e) **Keep
the ordering rule in `build()` and add evidence + task fields to `SystemPromptParts`** — rejected on the third
objection: it would put per-turn evidence inside the cacheable system preamble.

### D-8 — Serialized payloads are injected as message content, never through brace substitution

`serialize_to_toon` (`langchain_layer/models.py:224`, one definition, **~11 measured calls** — `decisions.md:100` says
16, which counts name occurrences including the definition, the imports and `__all__`; the exact number is not
load-bearing, the escaping obligation is) emits the mandated
`key[N]{field1, field2}` form (`brief:ref:54`). `ChatPromptTemplate` reads `{field1, field2}` as a template variable, so
any such payload passed through `to_chat_template()` raises `KeyError` at format time. The repo corpus documents **no**
escaping convention: `{{`/`}}`, `partial_variables`, and `string.Template` are all zero-hit gaps (`brief:725-727`).

| Option | Pros | Cons |
|---|---|---|
| Escape `{`→`{{` at the injection boundary | one line; everything stays inside the templating engine | must be applied at all 16 sites; escaping is lossy to reason about; one missed site is a runtime `KeyError` in a path with no tests |
| **Inject the payload as pre-formatted message content (chosen)** — a placeholder slot or a ready-made message, so the payload never reaches brace substitution | byte-exact by construction; no per-site discipline; matches the docs' own convention of building dynamic prompts outside the templating engine (`brief:276-280`) | the two prompt entry points must distinguish "template text" from "data payload" — which is exactly the distinction that was missing |

**The proof must assert the payload verbatim**, not merely that rendering did not raise: a test that only asserts "no
exception" passes against a silently double-braced or mangled payload.

### D-9 — An uncited assertion fails **validation**, not a log line

**The type and all three of its fields already exist**; only the validator is new. Verified 2026-08-18,
`agent_saul/state.py:103`:

```python
class Citation(BaseModel, frozen=True):
    claim: str = Field(description="The specific claim being made")
    source: str = Field(description="Document section, statute, or precedent ID")
    confidence: float = Field(ge=0.0, le=1.0)
```

So claim, source and **bounded** confidence are all present today, and `RiskFinding.citations: list[Citation]`
(`state.py:203`) and `ComplianceFinding.citations: list[Citation]` (`:219`) are already required fields. An earlier
draft of this decision said `Citation` "**is extended** to carry claim / source / bounded confidence", which would
lead an implementer to redefine the type and fork it. It is not extended. Two consequences for the spec:
`agent-structured-output`'s "Confidence is bounded" and "A cited finding is accepted" scenarios **already pass
today** and their tasks are regression tests, not new behaviour.

The **only** new work for Up#11 is a model validator rejecting an assertion-bearing finding with an **empty** citation
list, on `RiskFinding` (`state.py:203`), `ComplianceFinding` (`:219`) and `GroundingVerificationOutput` (`:239`). The
obligation is also stated in the prompt sections (D-7), so the model is told the rule it will be held to.

*Alternatives considered.* (a) **Warn and continue** — rejected: for a legal product an uncited assertion *is* the
failure mode, and at 3 a.m. a warning is indistinguishable from success. (b) **Enforce in a downstream verifier node** —
rejected: the invalid object would already exist and be persistable; validation is the cheapest place with no bypass.
(c) **Make citations optional with a flag** — rejected: the flag's default becomes the real policy.

### D-10 — D17: the wiring stays commented, and that makes fail-closed **more** important

D17 is binding: `lifespan.py:234-247` and `:294-305` were commented **deliberately**, so they are not a regression.
Consequences taken here:

- **No restoration, and no flag that defaults on.** The proposal's task set does not enable the graph. This change also
  introduces no `SAUL_GRAPH_ENABLED`-style toggle defaulting to `True`; an earlier draft of the plan proposed exactly
  that and it is withdrawn.
- **Fail-closed is now the primary justification for the dependency fix, not a side effect.**
  `features/agent_saul/dependencies.py:40-41` returns `request.app.state.saul_graph` unguarded, and `:45` reads
  `request.app.state.langgraph_checkpointer` unguarded — the `is None` check at `:46` does **not** protect against the
  attribute never being assigned, which is the actual state (`findings-database.md` §5: the checkpointer short-circuits
  to `None` because `psycopg`'s driver cannot load, and nothing assigns the graph at all). Both, plus `get_redis`
  (`:53`), become `getattr(..., None)` + `ServiceUnavailableException`, matching the sibling's existing shape. An
  intentional gap must answer 503; today it answers `AttributeError` 500 on **already-mounted** router surface
  (`api/v1.py:4,17`).
- **Proofs on commented code are import-level and type-level only.** Commented code cannot be type-checked, linted, or
  tested, so it will rot. Any task that touches the construction path proves itself by `ty`, by import, and by a unit
  test against the constructor's signature — **never** by running the graph. No task's Proof may read "the graph
  produces X".
- **What "the graph" means, stated because the discipline above otherwise forbids the one test that cannot be
  dropped.** The prohibition is on executing **the application's agent graph** — `build_saul_graph`, anything reached
  from the commented `lifespan.py:234-247` block, and anything requiring the real checkpointer, real settings, or a
  provisioned `app.state`. It is **not** a prohibition on executing a graph. Six scenarios are behavioural at runtime
  and have no import- or type-level witness:

  | Scenario | Capability |
  |---|---|
  | "A pause raised inside a wrapped invocation is not retried" | `agent-runtime-resilience` |
  | "The graph actually pauses" | `agent-runtime-resilience` |
  | "A raising tool does not terminate the run" | `agent-runtime-resilience` |
  | "A run exceeding its step budget terminates explicitly" | `agent-state-handoff` |
  | "Older recognised version is upgraded … before any other step reads it" | `agent-state-handoff` |
  | "The version check runs before any reasoning step" | `agent-state-handoff` |

  Worse, `interrupt` cannot pause at all without a checkpointer, and `findings-database.md` §5 records that the real
  one short-circuits to `None` because `psycopg`'s driver cannot load — so read literally, the pause-propagation test
  that this change calls non-negotiable had **no runnable path**, and an implementer facing that marks the scenario
  satisfied by inspection.

  **Authorised proof vehicle:** a throwaway `StateGraph` built inside the test itself — two nodes, a minimal state
  `TypedDict`, compiled with `langgraph.checkpoint.memory.InMemorySaver`. It is constructed by the test, imports
  nothing from `agent_saul/graph.py`, touches no `app.state`, requires no database, and does not uncomment or exercise
  the lifespan wiring. It exercises the **seam under test** — the retry middleware, the tool-error path, the
  `recursion_limit`, the hydration node — against a harness, which is the only way a seam's runtime behaviour can be
  asserted at all. Tasks using it must say so in their Proof, and must not import the application graph.

  **Still forbidden:** importing or invoking `build_saul_graph`; any Proof whose text reads "the agent graph produces
  X"; any Proof requiring a provisioned checkpointer, a live database, or a mounted route.
- **Nothing here may make re-enabling harder.** D17's *"at that time"* is noted. The renames (D-2), the registry
  adoption (D-1) and the tool binding all keep the construction entry points intact and correctly typed, so re-enabling
  stays a matter of uncommenting plus provisioning, not a redesign. That property is asserted by the import/type proof
  above.

*Alternatives considered.* (a) **Restore behind a flag defaulting off** — the plan's original step 18; rejected because
D17 forbids introducing the enabling machinery here at all, and a flag defaulting off still adds a startup path that
can raise. (b) **Delete the commented block** — rejected: it is the only record of intended wiring, and deleting it
makes re-enabling harder, which D17 forbids. (c) **Leave the dependency unguarded until wiring returns** — rejected:
the router is mounted now, so the 500 is live now.

### D-11 — Statute retrieval is retargeted onto `chunks`, and BM25 is **harvested, not rewritten**

`statutes` has no model and no migration, and per `findings-database.md` §4 it — along with `clauses`,
`parent_documents`, `entities`, `relationships`, `events`, `memory_versions` — **does not exist and never has**. Under
**D15** the target is `chunks`, never `clauses`. Zero rows exist anywhere in this subject area, so this is a retarget
with **nothing to migrate**.

Three things make it more than a table rename:

1. **The full-text engine changes.** The old precedent SQL as executed
   (`search_legal_precedents.py:182-200`) ranks with `ts_rank(fts_vector, plainto_tsquery(...))` against a
   **pre-existing `fts_vector` column** — `to_tsvector('english', body)` appears only in that module's docstring as
   suggested DDL, not in the executed query, and an earlier draft of this document conflated the two.
   `retrieve_statute_section.py:128-146` is an `ILIKE`-on-three-predicates point lookup with
   `ORDER BY year DESC LIMIT 1`. The target's index is `pg_textsearch` BM25 (`chunks_bm25_idx`, `a71f0d7d9c12:97`),
   queried as `<@> to_bm25query(...)` — the working reference is `features/search/repository.py:415-419`, in scope
   under D5.1, with RRF fusion at `features/search/fusion.py:28` (`k=60`).
   **A third BM25 implementation in this repo would be a planning failure.**

   **The harvest is not a lift-and-shift, and the difference is one line.** The working reference is
   `c.content <@> to_bm25query(:query, 'search_chunks_bm25_idx')` against `search_chunks`; the target index is
   `CREATE INDEX chunks_bm25_idx ON chunks USING bm25(search_text) …` (`a71f0d7d9c12:97`) — a **different column**
   (`search_text`, not `content`), a **different index name**, on a **different table**. So the harvest carries a
   column-name and index-name change. That is still a harvest, not a second implementation: the ranking expression,
   the parameter binding and the fusion are reused unchanged, and the requirement forbidding "a second ranking
   implementation" is satisfied. It is stated here so the implementer does not copy the reference verbatim and get a
   silent "column does not exist" at first execution.
2. **Statute identity attributes are consumed from change 2, not shipped here — Coordination point.** Change 3 ships
   **no DDL**. `legal-corpus-retrieval`'s requirement *"Statute identity attributes are addressable and efficiently
   retrievable"* mandates that the corpus carry the instrument name, the section reference and the year under a
   documented contract, with an index-served point lookup on the first two. Both clauses are DDL and both belong to
   the change that owns the retrieval schema.

   **Provider:** change 2 (`openspec/changes/documents-unified-schema/`), under its ADR **"`documents` / `chunks` is
   the sole retrieval schema"** (`adrs.md`, status **Accepted**) and its `document-retrieval-schema` capability. The
   ADR is the authority on the attribute names and on whether they land as columns or as a typed `metadata_`
   sub-object with a documented key contract; the nearest existing carriers are `UnifiedChunk.clause_type`
   (`model.py:92`), `UnifiedChunk.metadata_` (`:95`) and `UnifiedDocument.metadata_` (`:50`).

   **Consumer obligation on this change:** `legal-corpus-retrieval` states the requirement at the **attribute** level
   and names no column, deliberately — naming columns here would pre-empt the provider. The retarget task is
   explicitly gated on change 2 in `tasks.md`, by name, and no task in this change creates, alters or indexes a
   relation.

   **The gate is directional, which constrains the Proof and not just the order.** The requirement is *unsatisfiable*
   until change 2's migration lands, so any proof for it either runs **after** that migration or is import/type-level.
   A Proof asserting an index-served lookup **today** would be unexecutable against the deployed database — a failure
   that has already occurred twice in this refactor — so the retarget task carries an import-level Proof plus a
   database-level Proof marked as blocked on change 2, never a single Proof that silently assumes the relation exists.

   *An earlier draft filed these as "asks A1/A2 against change 2" in this document's prose only.* That was the
   "ask floats in prose" failure mode: change 2 contained zero occurrences of `A1`, `A2`, or this change's name, so
   the only implementer neither knew about nor had scheduled the work. The asks are now routed through the
   orchestrator to change 2's own remediation, and this change's position is downgraded from *asserting* the schema
   to *consuming* a contract whose provider is named.
3. **It floats.** This is the only part of the change that cannot be written today. If change 2 slips, the rest ships
   and the tools honestly report unavailability (D-3). That is the entire reason the honesty work precedes the
   retarget: an honest failure is a shippable state, a fabricated legal conclusion is not.

*Alternatives considered.* (a) **Create a `statutes` table** — rejected: D15 and change 2 consolidate onto
`documents`/`chunks`; a new table would be a fifth schema for the same content. (b) **Keep `tsvector` on the new
table** — rejected: `content_tsv` is being removed as a zero-reader generated column (D5.1) and `pg_trgm`/`tsvector`
paths are not the target engine. (c) **Write a fresh BM25 query for the tools** — rejected per (1).

### D-12 — **REVERSED.** `typed-exception-handling` is **modified**, not merely cited; `pattern-matching-standard` is still only cited

**Original decision (superseded):** both adjacent specs were "cited, not edited", on two rationales — (i)
`typed-exception-handling` governs *which* exception type is caught and annotated, whereas this change's requirements
govern *what the tool reports to the model*, so nothing in the existing requirement's behaviour changes; and (ii)
editing it would move a pre-existing validation-failure baseline other authors are measuring against.

**Both rationales are false, and the first is falsified by the spec's own text.** Verified 2026-08-18 against the
deployed `openspec/specs/typed-exception-handling/spec.md`:

- Its requirement `### Requirement: Agent tools SHALL catch OS-level and library-specific exceptions` (`:207-239`)
  **prescribes the return value**, not only the caught type. Five of its six scenarios end
  "*… and returns a `ToolOutput.fail()` result*" — `:219`, `:223`, `:227`, `:235`, `:239`. It governs both halves, and
  the distinction rationale (i) rested on does not exist.
- Worse, it names **`ToolOutput`** — the fourth envelope definition (D-3), the one this change **deletes**. So five
  scenarios of a **deployed** spec become false the day this change ships. That is not a citation relationship; it is
  a breaking edit to a deployed contract, and leaving it unrecorded would leave the deployed spec pointing at a class
  that no longer exists.
- Rationale (ii) is a rule this refactor does not itself follow: **change 0 already edits this same spec**
  (`openspec/changes/cleanup-foundation/specs/typed-exception-handling/spec.md`, a `MODIFIED` delta on the
  `asyncpg`/`PostgresError` requirement), and **change 1 adds four requirements to it**
  (`openspec/changes/ingestion-pipeline-unification/specs/typed-exception-handling/spec.md`, `## ADDED Requirements`).
  Empirically it confounds nothing: `openspec validate --all` reports the same six failures with both present.
- The original alternative (a) was rejected partly because "`MODIFIED` requires copying the entire original
  requirement block into a file whose parent spec already fails validation". Copying the block is the format's
  requirement, not an argument against it, and a change delta validates independently of its parent spec's
  pre-existing failure — change 0's delta demonstrates exactly that.

**Decision:** this change carries a `MODIFIED` delta on `typed-exception-handling`, at
`specs/typed-exception-handling/spec.md`, scoped to the single requirement *"Agent tools SHALL catch OS-level and
library-specific exceptions"*. It reproduces the requirement with **every scenario title verbatim** (all six), keeps
every caught-exception-type clause unchanged, replaces the five `ToolOutput.fail()` return clauses with the surviving
envelope's failure constructor, adds the rule that a catch site returns the envelope rather than a rendered string,
and adds the rule that a corpus-unreachable failure uses the unavailability constructor rather than the generic one.

**Collision discipline — three changes now touch this spec, and this is the third.** The delta touches **only** the
agent-tools requirement. It does **not** touch change 0's `asyncpg`/`PostgresError` requirement, and it does **not**
duplicate change 1's four `ADDED` requirements. Requirement ownership across the three deltas is therefore disjoint:

| Change | Operation | Requirement(s) |
|---|---|---|
| 0 `cleanup-foundation` | `MODIFIED` | Database operations SHALL catch `asyncpg.exceptions.PostgresError` |
| 1 `ingestion-pipeline-unification` | `ADDED` | four new retry/embedding-failure requirements |
| **3 this change** | **`MODIFIED`** | **Agent tools SHALL catch OS-level and library-specific exceptions** |

`pattern-matching-standard` remains **cited, not edited**: it governs the shape of the three-state return
(`found` / `absent` / `unavailable`) this change introduces, that shape conforms to it, and no requirement of it
becomes false. `llm-injection` was also evaluated and does not fit (see the proposal's Capabilities section).

*Alternatives considered.* (a) **Keep it cited only** — rejected: five scenarios of a deployed spec would name a
deleted class. (b) **Leave `ToolOutput` in place so the deployed spec stays true** — rejected: see D-3 alternative (d);
it would leave the string-as-error envelope in the package this change exists to unify, and make
`agent-tool-contract`'s first requirement false instead. (c) **`REMOVED` the whole requirement and re-add it under this
change's own capability** — rejected: the caught-exception-type clauses are correct, deployed, and not this change's
subject; removing them to restate them would discard a working contract and require a Reason and Migration for a
deletion nobody wants.

## Risks / Trade-offs

- **[Change 0 deletes `shared/agents/**` before this change's importer rewrite lands, and `import app.main` raises
  `ImportError` through `shared/rag/graphiti/registry.py:40-45`.]** → The rewrite is task 1 here and is cited by number
  from change 0's deletion task; this design states the cross-change predecessor explicitly. The paired restore is
  `git revert` of the deletion commit, then land the rewrite, then re-delete. **Boot risk ends at that deletion** —
  after it, nothing imports the stub tree.
- **[Adopting the registry turns a silent `None` into a `KeyError` for every tool name, because the survivor is
  empty.]** → D-1's two-commit order, with an executable non-empty proof before any consumer moves.
- **[Deleting `tools/registry.py` and changing package exports breaks a package `__init__` that
  `shared/rag/graphiti/registry.py:40` imports eagerly.]** → Keep `get_all_tools`/`get_web_tools` as thin aliases over
  the tag query in the same commit so the public surface is unchanged; revert is a one-line re-add.
- **[Renaming the Graphiti bundle misses one of four importers and boot fails at import.]** → Ship
  `ToolRegistry = AgentToolBundle` as an alias in the same module for one commit; assert exactly one `class ToolRegistry`
  remains repo-wide.
- **[The new envelope field breaks 30-day-TTL persisted rows on a rolling deploy, because `extra="forbid"`.]** → Bump
  `_REDIS_KEY_PREFIX` together with D-4's key-shape change and accept one cold cache. No dual-read.
- **[The key-shape change silently invalidates every cached tool result.]** → Same mitigation, stated in the Migration
  Plan rather than discovered as a latency spike.
- **[A retry wrapper swallows the HITL pause and the graph silently stops pausing.]** → The pause-propagation test
  exists solely for this and is the one test that cannot be dropped for time.
- **[Retaining usage accounting changes a return shape that four `cast(...)` calls currently hide from `ty`.]** →
  Decide per chain, test the shape, do not blanket-apply; where the metadata is genuinely unused, keep the current shape
  and record why at the call site.
- **[Renaming inside `shared/rag/graphiti/` collides with change 4, which also works there.]** → **Settled, not
  deferred:** the name is fixed by the second ADR in `adrs.md` (*"The Graphiti tool bundle is named `AgentToolBundle`"*,
  **Accepted** 2026-08-18), so change 4 writes the new name from its first line without waiting on this change's code.
  The earlier form of this risk said "land the rename before change 4 starts, **or agree the name in `adrs.md` first**"
  while `adrs.md` contained nothing about it — a dangling pointer. The mitigation is now the ADR plus the
  alias-for-one-commit it commits to.
- **[Zero coverage means no regression signal for any of this.]** → Eleven tasks carry mandatory tests; the two
  riskiest structural ones carry executable proofs independent of the suite.
- **[Making the layer honest reveals that with Graphiti degraded both evidence backends are dead.]** →
  `lifespan.py:220-223` already degrades silently (Graphiti failure sets `app.state.graphiti = None` and continues).
  D-3's unavailability signal plus D-10's 503 are what make that state honest instead of fabricated; under D17 the graph
  is not wired, so the exposure is bounded to the dependency surface.

## Migration Plan

**No database migration. No new dependency. No route mounted. Startup behaviour unchanged (D-10).**

**Ordering** — four phases; every task independently committable:

1. **Fail-closed and the cycle predecessor.** The dependency guards (D-10), then the three import-line rewrites
   (`get_obligation_chain.py:29`, `precedent_tools.py:21,22`) onto the D6/D6.1 survivors. Zero call-site edits are
   needed: both files already call the survivor API (`IdempotencyGuard.make_key` at `get_obligation_chain.py:67`,
   `precedent_tools.py:80,188`; `ToolResult.ok`/`fail`, which the stub lacks entirely) and `precedent_tools.py` already
   treats its scope as the real `MemoryScope` (`scope.top_k` at `:104`, `scope=scope` into `expand_from_seeds` at
   `:115`, the call opening at `:113`) against a 1-line `str` stub. **It is a strict bug fix that also unblocks change 0.**
2. **Shape.** Populate → adopt → rename (D-1, D-2), one envelope with availability (D-3) — **four definitions to one,
   including the rewrite of `shell.py`'s 13 `ToolOutput` sites onto the survivor and the deletion of
   `to_agent_string()`** — and the key contract (D-4). Provable by `ty` and by import; no runtime dependency.
3. **Honesty.** The unavailability register across `retrieve_statute_section.py:170-172`,
   `search_legal_precedents.py:227-229`, `precedent_tools.py:221-240`, and the deletion of the docstring sentence at
   `search_legal_precedents.py:179-180` that licensed the whole failure class. Then prompts (D-7, D-8), citations
   (D-9), declared output schemas, the retry seam (D-6), tool binding, the hydration step and the single version
   constant (D-5). The relocation of `shared/rag/rag_agent_advanced.py` to `src/app/examples/` (Q1) also lands here —
   a pure move with an import-level proof, no body edits.
4. **Floating.** The corpus retarget (D-11), gated on change 2's retrieval-schema migration (the Coordination point in
   D-11 §2 names the provider) plus change 0's alembic head merge. Because that gate is **directional**, the retarget
   task carries an import-level Proof that runs today and a database-level Proof marked blocked on change 2 — never a
   single Proof that assumes the relation exists.

**Measurable gates** — these are gates, not aspirations. Each is a number a task must produce:

| Check | Baseline | Gate | Basis |
|---|---|---|---|
| `uv run ty check src/` | **46** | **≤31** immediately after the import-line rewrite — **and if not, enumerate the residue** | `baseline-tests.md:167,169` localises 15 diagnostics to `precedent_tools.py` (11) + `get_obligation_chain.py` (4). Per-file counts re-measured 2026-08-18 and exact. The gate **assumes all 15 are caused by the stub imports** — plausible (all 15 sites touch `IdempotencyGuard` / `ToolResult` / `MemoryScope`) but **not proven**. If two survive, the task does not fail: it enumerates the residue and the gate becomes ≤33 with the residue named |
| `uv run ty check src/` | — | **≤28** after registry adoption | plus ≥3 in `agents/factory.py` (`baseline-tests.md:170`) |
| `uv run ty check src/` | — | **no increase** for every later task | the retry work perturbs 7 `unused-type-ignore-comment` diagnostics in both directions; check that removing a `# type: ignore` from `guardrails.py` did not convert a suppressed error into a live one |
| `rg -c "^class ToolRegistry" src/` | 3 | **1** | D-1, D-2 |
| `rg -c "class Tool(Result\|Output)\b" src/` | **4** | **1** | D-3. **Was `rg -c "^class ToolResult" src/` 3 → 1, which was satisfiable while the defect survived**: `ToolOutput` (`base.py:30`) does not match that pattern, so the change could pass its own gate and still leave two envelope shapes 28 lines apart in the same module. The gate must count all four |
| `rg -n "app\.shared\.agents\." src/` | non-zero | **0** after phase 1 | the cycle predecessor |
| `rg -n "tools\.registry\|get_tool_registry" src/` | non-zero | **0** after adoption | D-1 |
| `rg -n "tools=\[\]" .../agent_saul/factory.py` | 3 (`:116,122,128`) | **0** | tool binding. Reachable for all three roles only because D-5 exposes the handoff helper as the orchestrator's `handoff`-tagged tool set; before that correction the specs named tool sets for 2 of 3 roles and this gate was unreachable. **The gate is file-scoped on purpose**: a fourth `tools=[]` exists at `shared/langchain_layer/agents/registry.py:149` (a code-review agent spec, `# Could add linter tools, security scanners`), which no requirement in this change touches. Repo-wide the count is 4 — do not widen this gate to `src/`, or it becomes unreachable for a reason unrelated to tool binding |
| `rg -n "FROM statutes" src/` | non-zero | **0** after the retarget | D-11 |
| `uv run pytest` summary | **55 passed** | **≥75 passed**, same failures | ~20 new tests. **Compare the summary line, never `$?`** — `pyproject.toml:752-760` puts `--cov-fail-under=80` in `addopts` against 18.38% coverage, so the suite exits 1 even when every test passes |
| `uv run ruff check src/` | **123**, re-measured 2026-08-18 | **≤121** after change 0 lands; **≤123 and no increase** before it | An earlier draft said baseline **125 → gate ≤123**. Measured: `Found 123 errors` **with `todo_temp.py` still present**, and the two `invalid-syntax` errors (D11) are **inside** the 123. So a ≤123 gate would have permitted this change to add **two new lint errors and still pass**. After change 0 deletes `todo_temp.py` the value is **≤121**. Any task running before change 0 uses "no increase against the value measured at task start" |
| `ast-grep scan src/` | **4 errors** (exit **0**) | **4** | none of the vendored rules touch this layer; compare the printed count, not the exit code |
| `openspec validate --all` | **21 passed / 6 failed of 27**, re-measured 2026-08-18 | **no new failures — 6, not 7** | D12. An earlier draft recorded "16 passed / 6 failed of 22", faithfully quoting D12 but measured before the five refactor changes existed; the number would read as a regression to whoever ran it. The failing set is unchanged: `spec/cognee-v1-api`, `change/mintlify-documentation`, `spec/noqa-documentation`, `spec/pattern-matching-standard`, `spec/transactional-outbox`, `spec/typed-exception-handling`. Note that `spec/typed-exception-handling` is a **pre-existing** failure of the deployed spec and is **not** caused by this change's `MODIFIED` delta — `change/agent-tools-unification` passes `--strict` |
| `uv run python -c "import app.main"` | exits 0 | exits 0 after **every** boot-risk task | the four-row boot-risk ledger |

**Proof discipline under D17:** no task's Proof may involve executing **the application's agent graph**.
Construction-path tasks prove by `ty`, by import, and by a unit test against the constructor. Six runtime scenarios
have no import- or type-level witness and are proven against a **throwaway two-node `StateGraph` compiled with
`InMemorySaver`, constructed inside the test** — see D-10 for exactly what that authorises and what remains forbidden.

**Proof dependency to restate, not assume:** `outbox_events` and `dead_letter_events` do not exist either
(`findings-database.md` §8, change 0 owns that fix). No Proof in this change depends on an outbox event firing, and none
may be written that way — if a later task needs one, its Proof is blocked on change 0 and must say so.

**Rollback.** Per task, from the boot-risk ledger: revert the deletion commit (phase 1); the one-line alias re-add
(registry exports); the `ToolRegistry = AgentToolBundle` alias (rename); cold cache is accepted, not rolled back
(envelope + key shape). Startup is not modified, so there is no startup rollback to plan.

## Open Questions

One remains genuinely open. None blocks starting phase 1.

- **Q1 — the disposition of `shared/rag/rag_agent_advanced.py` (~600 lines). CLOSED 2026-08-18 by user decision.**
  **The file is moved to `src/app/examples/`. It is not deleted, and it is not harvested into change 1.**
  The author's recommendation was *harvest then delete*; the user chose **relocation**, and the decision is recorded
  here as taken, not as a preference. Per `CLAUDE.md`, examples belong in `src/app/examples/` — which already exists
  and already holds `logger_usage_example.py`, `redis_examples.py` and three guides — so the move needs no new
  directory and no new convention. The file's facts are unchanged and are what make the move safe: **zero importers**,
  its entry point is a CLI (`run_cli()` `:517`), it is pydantic-ai rather than langchain, it queries a `match_chunks()`
  function defined in no migration and no source file, and it imports `from ingestion.embedder import create_embedder`
  (`:119,198,267,373`) — **a package that does not exist in this repo** — so every one of its tools `ImportError`s on
  first call regardless of where the file lives.

  **Two losses are accepted on the record, and both are Non-Goals above rather than deferred work:**
  1. **The string-as-error anti-pattern survives**, at **five** sites: `f"Search error: {e!s}"` at `:172,244,293,481`
     plus `f"Error retrieving document: {e!s}"` at `:345`. (Both this document and `review.md` first cited
     `:172,244,293,345,481` (re-measured; earlier drafts were off by three); **re-measured 2026-08-18, every one was off by three**, and the fifth site is a
     differently-worded string, so a grep for `Search error` alone finds only four.) It is **quarantined**, not
     fixed: under `src/app/examples/` it stops reading as production code, but the strings are still there. No task in
     this change edits the file's bodies.
  2. **The iterative-RAG prior art stays unused.** `search_with_self_reflection` (`:353`, grading at `:420`, query
     refinement at `:460`) and `expand_query_variations` (`:52`) are the repo's only prior art for iterative RAG, and
     `dispositions.md` routes agentic query rewriting to change 1's item 195. Change 1 therefore receives **no** design
     note from this file and will design that item from scratch. **This change writes no harvest task.**

  The one task Q1 produces is a **pure move** — `git mv` plus an import-path fix if any exists — with an import-level
  proof and no body edits. See `tasks.md`.
- **Q2 — how tool-error handling is reached under the installed versions. CLOSED 2026-08-18 by reading the installed
  packages** (langchain **1.2.12**, langgraph **1.1.2**, langgraph-prebuilt **1.0.8**).

  **`handle_tool_errors` is unreachable through `create_agent`.** The signature
  (`langchain/agents/factory.py:673-691`) has no `handle_tool_errors` parameter and no `tool_node` parameter; passing
  `tools=ToolNode(...)` raises `TypeError` and `tools=[ToolNode(...)]` raises `ValueError`; the internal tool-node
  construction (`factory.py:909-928`) threads only `tools` plus `wrap_tool_call` / `awrap_tool_call`. **The 0.2-era
  `ToolNode(handle_tool_errors=…)` shape is therefore not writable against this codebase** — the binding constraint
  in the brief is confirmed, not merely assumed.

  **What that means for today's behaviour:** the default handler `_default_handle_tool_errors`
  (`langgraph/prebuilt/tool_node.py:379-387`) **re-raises everything except `ToolInvocationError`**. A tool that raises
  aborts the run wherever no middleware intercepts it.

  **The reachable seam is middleware**, in two forms: a custom `@wrap_tool_call`
  (`langchain/agents/middleware/types.py:649-653`, exported at `middleware/__init__.py:80`), or the purpose-built
  **`ToolRetryMiddleware`** (`middleware/tool_retry.py:30`, `__init__` at `:128-139`), whose default
  `on_failure="continue"` (`:134`) converts an exhausted failure into a `ToolMessage(status="error")` (`:273-286`)
  instead of re-raising. Write `"continue"` / `"error"`, never the deprecated `"return_message"` / `"raise"` spellings,
  which warn and are rewritten at `:191-204`.

  **This mechanism is already in the repo and already wired — the resilience work is not greenfield.**
  `ToolRetryMiddleware` is constructed at `guardrails.py:345` (`max_retries=3, backoff_factor=1.5`) and `:369`
  (`max_retries=2`); `build_default_middleware_stack` (`:304`) is called from
  `shared/langchain_layer/agents/factory.py:152` and reaches `create_agent` as `middleware=` at `:188`. Neither site
  passes `on_failure`, so both already take the non-fatal default.

  **The actual gap is `agent_saul`.** `shared/langgraph_layer/agent_saul/factory.py` contains **zero** occurrences of
  `middleware` — its three `create_agent` calls install no stack at all, so a raising tool there aborts the run while
  the same tool under the survivor factory returns an error message. The task this produces **adds the existing stack to
  the second factory**; it does not design a retry mechanism. See D-6.
- **Q3 — provider-native structured output per model. CLOSED 2026-08-18 by reading the installed profile table. The
  silent fallback is real, for exactly the two models this repo configures.**

  `settings.py:191-192` defaults to **`gemini-3.1-flash`** and **`gemini-3.1-pro`**. Neither key exists in
  `langchain_google_genai/data/_profiles.py`. Of its 28 entries the only `gemini-3` keys are
  `gemini-3-flash-preview`, `gemini-3-pro-preview`, `gemini-3.1-pro-preview` and
  `gemini-3.1-pro-preview-customtools` — **all `-preview`-suffixed; the GA names are absent.**

  A miss returns **`{}`, not `None`**, which is the trap: `{}` passes an `is not None` guard but fails
  `.get("structured_output")` inside `_supports_provider_strategy` (`langchain/agents/factory.py:509-522`), so
  `AutoStrategy` resolves to **`ToolStrategy`** (`factory.py:1190-1200`) — tool-calling emulation. Passing a bare
  schema therefore yields something weaker than it appears, with no warning. A further exclusion compounds it: models
  below 3.x are excluded whenever tools are present, even when their profile does advertise `structured_output`.

  **Three fixes exist, and the specs must not claim strictness without one:** pass `ProviderStrategy(schema=…)`
  explicitly; override `profile={"structured_output": True}`; or configure a key the table knows —
  `gemini-3.1-pro-preview` and `gemini-3-flash-preview` both carry `structured_output=True` (verified).
  Changing the configured model is a **settings** change and out of scope here, so this change records the constraint
  and any `agent-structured-output` proof asserts the strategy actually selected rather than assuming the native path.

  **Incidental blocker for any probe task:** `models.py:126-138` defaults `implementation="generic"`, which infers
  provider `google_vertexai` (`chat_models/base.py:539-540`) — not installed, so the probe `ImportError`s unless it
  names the provider explicitly. A Proof that runs `init_chat_model` must pass the provider.
- **Q4 — message trimming for the handoff envelope. DELIBERATELY OPEN — a follow-up, not a blocker.** `RemoveMessage`,
  `REMOVE_ALL_MESSAGES`, and `add_messages` ID-collision semantics are all zero-hit gaps in the repo corpus
  (`brief:718-719`), and trimming tool output in a multi-step conversation (`brief:ref:58`) only becomes concrete once
  agents actually exchange messages under D-5's envelope. It needs the library source as the authority, and the
  envelope has to exist before the trimming policy can be designed against real message volume. No task in this change
  addresses it; it is not scheduled, and that is the intent.
- **Q5 — whether `get_research_agent` / `get_code_review_agent` are reached from outside `src/app`. CLOSED 2026-08-18
  by measurement — they are not.** A repo-root search returns **zero code callers** outside
  `src/app/shared/langchain_layer/agents/registry.py`, where they are defined (`:97`, `:144`). Every other hit is
  planning prose about this very question (`docs/relay/plan-change3.md:753,756`,
  `docs/relay/scout-tools-duplicates.md:124,129,160`, `docs/relay/open-questions.md:74`, and this document). No
  notebook, root script, `src/app/examples/` module, or test reaches them.

  **Consequence: the priority of the registry adoption is unchanged.** `factory.py:146`'s `AttributeError` stays
  classified as *reachable only by a developer or a new feature calling the registry directly* — it does **not** get
  promoted to "breaks on first request". It also means the second `AttributeError` this change unmasks
  (`MemoryManager.inject_long_term_context` / `save_session`, Non-Goals above) is equally unreachable today, so
  unmasking it is a recorded gap rather than a live regression.

**Closed since the plan was written:** whether `statutes`, `clauses`, `parent_documents`, `entities`, `relationships`,
`events`, `memory_versions` or `match_chunks` exist in the deployed database — **they do not**
(`findings-database.md` §4, §7, live probe). The retarget is therefore a retarget, not a data migration, and the
unavailability path is not currently masking a transient error.
