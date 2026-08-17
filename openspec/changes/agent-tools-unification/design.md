> Change class: **L** — cross-cutting (multi-module, breaking internal import surface, a retarget onto a different schema and full-text engine, a correctness boundary in a legal product).
> The proposal covers *why* and *what*; this covers *how*. Reference the proposal - do not restate it.

## Context

The layer this change touches has never executed. Three independent proofs, each sufficient on its own:

| Proof | Evidence |
|---|---|
| Tools are never bound to agents | `shared/langgraph_layer/agent_saul/factory.py:114,120,126` — all three `create_agent` calls pass `tools=[]` with `# TODO: add ... when available  # noqa: FIX002`, while `build_saul_graph` (`agent_saul/graph.py:91`) threads a tool bundle down through `_build_graph_nodes(:95)` to them |
| The survivor registry is empty | `shared/langchain_layer/agents/tools/base.py:99` is a module-level `registry = ToolRegistry()`; the only module using `@register_tool` is `tools/shell.py` (`:41,96,114,132,174`) and it has **zero importers** |
| The builder has no callers | `shared/rag/graphiti/registry.py:98` `build_tool_registry` — zero callers; the lifespan block that would call it (`lifespan.py:234-247`) is commented out, deliberately (D17) |

Two more facts shape every decision below.

**The export inversion.** `shared/langchain_layer/agents/tools/__init__.py:7-12` re-exports `ToolRegistry` from
`.registry`, **not** from `.base`. The scout report claimed the opposite. So `...agents.tools.ToolRegistry` names the
**loser** class today, `base.py` is not re-exported by its own package at all, and the D6.1 survivor is the harder of
the two to reach. Correcting that line is a behaviour change for every importer of the package symbol.

**The import cycle that breaks boot.** `shared/rag/graphiti/registry.py:41-46` eagerly imports the four `make_*_tool`
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
- **Editing `typed-exception-handling` or `pattern-matching-standard`.** See decision **D-12**.

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

Adopting first is a live hazard, not a theoretical one: `factory.py:146` currently calls `get_tool_registry().get_tool(t)`
which returns `None` on miss, and the survivor's `get` **raises `KeyError`** (`base.py:73`). Against an empty registry
that converts a silent miss into a hard failure for *every* tool name. `registry.py:9`'s class also defines only
`get_tool`, never `get` — so `factory.py:146` is a confirmed live `AttributeError` today. Fail-fast is the intent; it is
only safe once the registry has contents.

*Alternatives considered.* (a) **Adopt then populate** — rejected: guarantees a window where every tool resolution
raises. (b) **Keep both registries behind a facade** — rejected: D6 mandates one, and a facade preserves the two
divergent `get`/`get_tool` contracts that produced the current bug. (c) **Populate by importing `shell.py` from the
package `__init__`** — rejected in favour of **explicit registration**: importing a module nothing has ever imported to
obtain a decorator side effect makes package import order load-bearing, and would run that module's top-level code in
every process including Celery workers. Explicit registration is order-independent and greppable. This also closes the
plan's Fog #2 without needing to audit `shell.py` for import-time I/O.

### D-2 — The third same-named class is **renamed, not deleted**

`shared/rag/graphiti/registry.py:56` is a third class called `ToolRegistry` and a **different concept**: an immutable
Pydantic bundle of four pre-built tools, constructed once (`build_tool_registry` `:98-122`) and consumed as a value
object. D6.1 forbids deleting the file. It is renamed (`AgentToolBundle`) and its four importers updated
(`agent_saul/factory.py:10,182`, `agent_saul/graph.py:16,91`), so the repo has exactly one `ToolRegistry`. The module
docstring (`registry.py:9,25`) is corrected in the same commit — it currently points at the deleted-stub import path and
at an `app.state.saul_graph` assignment that exists nowhere, making it the most misleading comment in the layer.

Sequence it **after** D-1's adoption, so no window exists in which one imported name has two meanings.

*Alternatives considered.* (a) **Delete the file and fold the bundle into the registry** — rejected: D6.1 says not
deletable, and a bundle-of-four is genuinely not a registry; collapsing them would put per-agent tool assignment inside
a global registry. (b) **Leave the name collision** — rejected: it is the reason three scouts disagreed about which
class survives. (c) **Rename the survivor instead** — rejected: the survivor is the one with 20+ potential importers and
the public package symbol.

### D-3 — Availability is a **field** on the result envelope, not a metadata key

The survivor envelope is `langchain_layer/agents/tools/idempotency.py:34` (`extra="forbid"`, `frozen=True`,
`success`/`data`/`error`/`metadata`, `ok()`/`fail()`). Its twin at `shared/agents/tools/idempotency.py:11` dies with
change 0's deletion. The third at `shared/rag/document_processing/models.py:318` has **exactly one importer** —
`todo_temp.py:8`, which D11 deletes — so removing it migrates **no caller**.

Add an explicit unavailability state and a third constructor. A retry, escalation, or "do not tell the user this law
doesn't exist" decision must not be spelled in a free-form dict.

*Alternatives considered.* (a) **Carry it in `metadata`** — technically works (`**meta` already flows there) and was
rejected: an unkeyed convention in a dict is exactly how the current defect survived review, and no type checker can
enforce it. (b) **Raise an exception for unavailability** — rejected: these values are returned to a model as tool
output; an exception either aborts the run or is caught and re-stringified, which is the anti-pattern being removed.
(c) **A separate result type per outcome** — rejected: the envelope is persisted as JSON and read back with a single
validator; a union would need a discriminator, which is the field being added anyway.

**Serialization hazard, stated because it is a rolling-deploy trap.** The guard persists this envelope as JSON in Redis
and Postgres with a 30-day TTL (`idempotency.py:31`, `_POSTGRES_TTL_DAYS = 30`) and reads it back with
`model_validate_json` (`:77`). `extra="forbid"` means **new-schema rows read by old code raise**. Adding a defaulted
field is forward-safe but not backward-safe. Resolution: bump `_REDIS_KEY_PREFIX` in the same commit as the key-shape
change (D-4) and accept one cold cache. **No dual-read.**

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
`content: dict | None`; the write path (`graphiti/write_clause_episodes.py:35` holds the guard) passes `content=None`.
A single opaque `input_data` dict cannot express the distinction, which is precisely why it drifted.

*Alternatives considered.* (a) **Trap2 literally, everywhere** — rejected: search-result cache collisions across
distinct questions, in a legal product. (b) **Content everywhere** — the status quo; rejected: duplicate graph writes on
replay. (c) **Document the convention without changing the signature** — rejected: an unenforced convention is what
exists now.

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
   naming the recipient plus a router edge that reads it (`brief:ref:1473-1479`); saul instead routes on custom router
   functions via `add_conditional_edges` (`graph.py:50,55,66,74`) with **zero** `transfer_to_*` or `Command(goto=...)`
   anywhere in the repo.

So: build one construction helper for the handoff message and one router rule that reads it, set an explicit
`recursion_limit` (`brief:ref:1492` requires it; `ref:1471` warns there is no loop detection outside the supervisor
state), and **convert no state class**.

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
| **Middleware at the model/tool seam (chosen)** | the seam already exists in this repo — `middlewares/guardrails.py:49,159` uses `@wrap_model_call`; retry state lives outside node bodies so replay does not multiply it; pause propagation is testable in isolation | one more middleware to configure; `ToolNode(handle_tool_errors=...)` reach under the installed version needs verification (Open Question **Q2**) |
| Both | — | two retry budgets composing multiplicatively, invisibly |

Sub-todo (j)'s **intent** — bounded retry with backoff — is honoured; its named vehicle is not the documented one.
**Change 1 also touches `tenacity`** (its I/O-boundary usage in the ingestion path). This decision does not contradict
that work: it *confirms* the boundary as tenacity's home and only bars it from graph-node interiors.

If the user overrides this, the override needs its own task proving `interrupt` still propagates through the
`tenacity`-wrapped call — that test is the entire risk.

### D-7 — The prompt seam is built in the **opposite direction** from the todo's framing

Todo (1) frames this as "Template vs `ChatPromptTemplate`" and the brief framed `render_prompt_sections` as a
"competing helper". Measured, it is the **dominant** one: `render_prompt_sections` (`langchain_layer/prompts.py:145`)
has **26 call sites** (`agent_saul/prompts.py` 11, `open_deep_search/prompts.py` 8, `ingestion_kb/prompts.py` 4,
`retrieval_kb/nodes.py` 3, `reconciliation/prompts.py` 1) and returns a bare `str`. `SystemPromptParts`
(`prompts.py:19`) — which owns `build()` (`:99`), `Template(...).safe_substitute` (`:122`) and `to_chat_template()`
(`:126`) — has **2** real sites (`agents/factory.py:171`, `agents/registry.py:103,150`).

So: **do not migrate 26 sites.** Make `render_prompt_sections` the section-assembly primitive it already is, make
`SystemPromptParts` consume it, and implement the Up#6 ordering rule **once**, inside `build()`.

*Alternatives considered.* (a) **Migrate all 26 onto `SystemPromptParts`** — rejected: 26 edits in files with zero
coverage, to reach a helper with 2 adopters. (b) **Delete `SystemPromptParts`** — rejected: it owns the only path to a
chat template and the only variable-substitution seam. (c) **Leave both** — rejected: the ordering rule would then have
two implementations, which is how "Lost in the Middle" ordering silently stops holding.

### D-8 — Serialized payloads are injected as message content, never through brace substitution

`serialize_to_toon` (`langchain_layer/models.py:224`, one definition, 16 call sites) emits the mandated
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

The type already exists: `agent_saul/state.py:103` defines `class Citation(BaseModel, frozen=True)`. It is extended to
carry claim / source / bounded confidence, and a model validator rejects an assertion-bearing finding with an empty
citation list, on `RiskFinding` (`state.py:203`), `ComplianceFinding` (`:219`) and `GroundingVerificationOutput`
(`:239`). The obligation is also stated in the prompt sections (D-7), so the model is told the rule it will be held to.

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
- **Nothing here may make re-enabling harder.** D17's *"at that time"* is noted. The renames (D-2), the registry
  adoption (D-1) and the tool binding all keep the construction entry points intact and correctly typed, so re-enabling
  stays a matter of uncommenting plus provisioning, not a redesign. That property is asserted by the import/type proof
  above.

*Alternatives considered.* (a) **Restore behind a flag defaulting off** — the plan's original step 18; rejected because
D17 forbids introducing the enabling machinery here at all, and a flag defaulting off still adds a startup path that
can raise. (b) **Delete the commented block** — rejected: it is the only record of intended wiring, and deleting it
makes re-enabling harder, which D17 forbids. (c) **Leave the dependency unguarded until wiring returns** — rejected:
the router is mounted now, so the 500 is live now.
