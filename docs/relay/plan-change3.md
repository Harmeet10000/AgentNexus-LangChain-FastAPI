# Plan — openspec change 3: agent tools

Planner leg. Read-only on `src/`. Authored 2026-08-17 against
`docs/relay/{decisions,dispositions,scout-tools-duplicates,scout-tools-schema,brief-langgraph-practices,conventions-openspec-skeleton,baseline-tests}.md`.

Locked inputs: **D6 / D6.1** (survivors), **D7** (`open_deep_search` out), **D8** (registry unification gates
change 3; change 2's schema consolidation determines what the retargeted tools query), **D11** (`todo_temp.py`
deleted in change 0), **D13** (DROP/DEFER become Non-Goals). None re-litigated below.

## Shape

Change 3 is **not** a tools rewrite. It is a *truth-telling* change over a layer that is currently unreachable
three times over: the lifespan wiring is commented out (`lifespan.py:235-249`), `build_tool_registry`
(`shared/rag/graphiti/registry.py:98`) has zero callers, **and** — a fact no scout stated — all three
`create_agent` calls in `agent_saul/factory.py:114,120,126` pass `tools=[]` with `# TODO` comments, so the
`tool_registry` threaded through `build_saul_graph(:91)` is never bound to any agent. The four legal tools are
dead at the wiring layer, the registry layer, *and* the agent layer. Nothing here is a production incident, and
nothing here yields a test signal today.

That reorders the work. The cheap, safe, high-yield half is **shape**: one `ToolResult`, one `ToolRegistry`, one
`IdempotencyGuard`, one `MemoryScope`, one prompt seam — all provably correct by `ty` and by import, with no
runtime dependency. The expensive half is **honesty**: two tools currently report a missing table as
"this section of law does not exist", and one computes `insufficient_basis` from a single surviving source. Those
must be fixed *before* the wiring is restored, because restoring the wiring is what makes them fire.

The steps below therefore run shape-first (1-8), honesty-second (9-14), schema-retarget third and floating
(15, change-2 gated), and wiring dead last behind a settings flag (18). Every step is independently committable.
Boot is at risk in exactly four of them, all named in the ledger.

Two findings materially change the work list as handed to me, and both are load-bearing:

1. **`shell.py` has zero importers**, so the five `@register_tool`-decorated tools never register, so
   `base.py:99`'s module singleton `registry` is **empty at runtime**. Swapping `factory.py:146` from
   `get_tool_registry().get_tool(t)` (returns `None`) to `registry.get(t)` (raises `KeyError`) therefore fails
   *every* tool name, not zero. The survivor must be **populated** before it is adopted — two commits, not one.
   This closes `scout-tools-duplicates.md` Fog #3.
2. **`tools/__init__.py:7-12` re-exports `ToolRegistry` from `.registry`, not `.base`.** The scout stated the
   opposite. So `app.shared.langchain_layer.agents.tools.ToolRegistry` is today the **loser** class, and
   `base.py` is not re-exported by its own package at all. The D6.1 survivor is currently the harder of the two
   to reach.

Third finding, cheaply resolving `scout-tools-duplicates.md` Fog #1: `precedent_tools.py` uses `scope.top_k`
(`:104`) and passes `scope=scope` into `subgraph_expander.expand_from_seeds` (`:117`). Against the 1-line
`str` stub both are `AttributeError` / wrong-type. The file **already targets the survivor's `MemoryScope` API** —
so the import swap needs **zero** call-site edits and is a strict bug fix. That is why it is step 2, not step 12.

## Ordering constraints

Inbound, cross-change, and internal. Each step below names which of these it consumes.

| id | Constraint | Source |
|---|---|---|
| **X1** | **The cycle.** `shared/rag/graphiti/registry.py:41-46` eagerly imports the four `make_*_tool` factories from the `tools` package at module import. Two of those modules import the `shared/agents/**` stubs (`get_obligation_chain.py:29`, `precedent_tools.py:21,22`). So change 0's `shared/agents/**` deletion raises `ImportError` at *import* time unless change 3's importer rewrite lands first. **Change 3 step 2 is a blocking predecessor of change 0's deletion task.** | D6.1, `scout-tools-duplicates.md:153` |
| **X2** | **D8.** Registry unification gates the rest of change 3 — steps 4-6 precede everything that resolves a tool by name (13, 16, 18). | D8 |
| **X3** | **D8 / change 2.** The `statutes` retarget (step 15) cannot be written until `UnifiedDocument`/`UnifiedChunk` is the consolidated target *and* change 2 has answered two schema asks filed by this plan (below). Step 15 is the only step in change 3 that cannot be done today. | D8, `scout-tools-schema.md:36-37` |
| **X4** | **Change 1.** `build_saul_graph(checkpointer=...)` requires `app.state.langgraph_checkpointer` to be set, which is change 1's item 138 residue (a). Step 18 is blocked on it. | `dispositions.md` change 1 row 138(a) |
| **X5** | **Change 0.** The third `ToolResult` (`shared/rag/document_processing/models.py:318`) has **exactly one importer**: `todo_temp.py:8`, which D11 deletes. After change 0, that class is zero-importer dead code. Step 7 therefore only removes an orphan; it does **not** have to migrate any caller. Verified: `rg ToolResult src/` outside `langchain_layer/agents/tools/` hits `todo_temp.py` only. | D11, verified |
| **X6** | **Internal.** The survivor registry is empty (Shape finding 1) → populate (step 4) before adopt (step 5). |  verified |
| **X7** | **Internal.** Fix invisible failures (step 9) before restoring wiring (step 18), or the restoration ships a compliance path that silently claims statutory basis. Orchestrator's explicit instruction; accepted as stated. | brief |
| **X8** | **Internal.** `ToolResult` must carry an availability signal (step 7) before step 9 can express "corpus unavailable" distinctly from "not found". |  design |

### Two schema asks filed against change 2

Change 3 files no migration of its own. It requires change 2 to deliver, or change 3 adds one narrow migration on
the merged alembic head as a conditional sub-step of step 15:

- **A1** — carriers for `act_name`, `section_ref`, `year`. `scout-tools-schema.md:36` confirms **no equivalent
  column** on `UnifiedDocument`/`UnifiedChunk`; nearest are `UnifiedChunk.clause_type` (`model.py:92`),
  `UnifiedChunk.metadata_` (`:95`), `UnifiedDocument.metadata_` (`:50`). Recommend a typed `metadata_` sub-object
  with a documented key contract, not three new columns.
- **A2** — a point-lookup index for `(act_name, section_ref)`. `retrieve_statute_section.py:128-146` is an
  equality-ish lookup (`ILIKE` on all three predicates, `ORDER BY year DESC LIMIT 1`) and
  `scout-tools-schema.md:37` records that `documents`/`chunks` has no index for it. Without A2 the retarget is a
  sequential scan on the largest table in the schema.

Also note for step 15: this is **not** a table rename. The `statutes` SQL uses `to_tsvector`/`ts_rank`
(`search_legal_precedents.py:193-199`); the target's FTS is `pg_textsearch` BM25 — `chunks_bm25_idx`
(`a71f0d7d9c12:102`) queried as `c.content <@> to_bm25query(...)` per `features/search/repository.py:415`. The
retarget changes the **FTS engine**, and its reference implementation lives in `features/search/`, which D5.1 put
back in scope. Harvest it; do not write a third one.

## Steps

**Baseline traps that apply to every Proof line below.** `pyproject.toml:752-760` puts `--cov-fail-under=80` in
`addopts` and total coverage is **18.38%**, so `uv run pytest` **exits 1 even when every test passes**. Every
pytest Proof compares the **summary line** (`N passed`), never `$?`. Likewise `ast-grep scan src/` exits **0** with
error-level diagnostics — compare the printed count against **4**. Baselines: **55 passed** / **ruff 125** (→ 123
after change 0 deletes `todo_temp.py`, D11) / **ty 46** / **ast-grep 4**.

---

### 1. `get_saul_graph` fails closed with 503, not `AttributeError`

`features/agent_saul/dependencies.py:40-41` is `return request.app.state.saul_graph` with no guard, while its
sibling `get_saul_checkpointer` (`:44-49`) already raises `ServiceUnavailableException` on `None`. Since nothing
assigns `app.state.saul_graph`, **every request to the mounted `agent_saul_router` (`api/v1.py:4,17`) dies with an
`AttributeError` 500.** Make it symmetric with its sibling: `getattr(..., None)` + `ServiceUnavailableException`.

This is the same defect family as D5.2's `UserIdDep` and belongs to change 3 only because it is this router. It
converts an unhandled 500 into an honest 503 on **already-shipped surface**, so it goes first and alone.

- **Inbound:** none. Independent of everything else in this change.
- **Boot risk:** none.
- **Tests:** mandatory — a route-level test asserting 503 with the router mounted and `app.state.saul_graph` unset.
  This module has zero coverage today.
- **Proof:** `uv run pytest tests/unit/ -k saul_graph_unavailable` → `1 passed` in the summary line; and
  `rg -n "app\.state\.saul_graph" src/app/features/agent_saul/dependencies.py` shows the access is guarded.

### 2. Rewrite the three stub imports onto the D6/D6.1 survivors — **the cycle predecessor**

Three import lines, zero call-site edits:

| File:line | From | To |
|---|---|---|
| `shared/langchain_layer/agents/tools/get_obligation_chain.py:29` | `app.shared.agents.tools.idempotency` | `.idempotency` |
| `shared/langchain_layer/agents/tools/precedent_tools.py:22` | `app.shared.agents.tools.idempotency` | `.idempotency` |
| `shared/langchain_layer/agents/tools/precedent_tools.py:21` | `app.shared.agents.memory.memory_scope` | `..memory.memory_scope` |

No other change is needed. Both files already call the survivor API (`IdempotencyGuard.make_key` at
`get_obligation_chain.py:67`, `precedent_tools.py:80,188`; `ToolResult.ok`/`fail`, which the stub lacks entirely)
and `precedent_tools.py` already treats `PRECEDENT_SCOPE` as a `MemoryScope` — `scope.top_k` (`:104`) and
`scope=scope` into `expand_from_seeds` (`:117`). The 1-line `str` stub makes both wrong. **This step is a strict
bug fix that happens to also unblock change 0.**

- **Inbound:** none. **Blocks change 0's `shared/agents/**` deletion (X1).** Change 0's `tasks.md` must cite this
  task ID by number.
- **Boot risk:** none — the stub tree is still on disk; this step only stops pointing at it.
- **Tests:** optional. `ty` is the stronger signal here and it is free.
- **Proof:** `uv run ty check src/` → **31 diagnostics or fewer** (baseline 46 minus the 15 that
  `baseline-tests.md:167,169` localises to `precedent_tools.py` (11) + `get_obligation_chain.py` (4)); and
  `rg -n "app\.shared\.agents\." src/` returns **zero** hits; and
  `uv run python -c "import app.shared.rag.graphiti.registry"` exits 0.

### 3. [change 0 owns the edit; change 3 owns the assertion] delete `src/app/shared/agents/**`

Five files, all created in one commit `c228398` (2026-07-02) *after* their real counterparts existed — verdict
**accidents, not shims** (`scout-tools-duplicates.md:108`). Listed here only to fix its position in the order.

- **Inbound:** step 2. **This is the last step at which boot is at risk from the X1 cycle.**
- **Boot risk:** **YES.** If step 2 has not landed, `import app.main` raises `ImportError` transitively through
  `shared/rag/graphiti/registry.py:40-46`. Paired restore: `git revert` the deletion commit, or land step 2 first.
- **Tests:** none new.
- **Proof:** `rg -n "shared\.agents" src/` → zero; `uv run python -c "import app.main"` exits 0;
  `uv run pytest` summary still reads **55 passed**.

### 4. Populate the survivor registry (no consumer moves yet)

`base.py:99`'s `registry = ToolRegistry()` is empty at runtime because `shell.py` — the only module using
`@register_tool` (`:41,96,114,132,174`) — **has zero importers**. Fix the supply side first, while
`tools/registry.py` still serves its consumers:

- import `shell` from `tools/__init__.py` (or register the five explicitly) so the decorator runs at package import;
- give `web_search` (`web_search.py:80`) and `crawl_url` (`crawl.py:114`) tags on the survivor, since
  `tools/registry.py:53,58`'s `get_all_tools`/`get_web_tools` (identical bodies — written twice) are exactly
  `registry.by_tags("web")`;
- re-export `registry`, `register_tool`, `make_structured_tool` from `tools/__init__.py` (today they are reachable
  only by direct module import).

- **Inbound:** none strictly, but sequence after step 2 so the package imports cleanly.
- **Boot risk:** low but real — importing `shell.py` at package-import time runs module-level code in a file
  nothing has ever imported. Read it before importing it; if it does I/O at import, register explicitly instead.
- **Tests:** mandatory — assert the singleton's contents, because "empty registry" is the exact failure this
  change exists to prevent and nothing else will catch a regression.
- **Proof:** `uv run python -c "from app.shared.langchain_layer.agents.tools.base import registry; print(sorted(registry.names()))"`
  prints a **non-empty** list containing `web_search` and `crawl_url`; and
  `uv run pytest tests/unit/shared/ -k tool_registry` → the new assertions pass in the summary line.

### 5. Adopt the survivor; delete `tools/registry.py`

Now the consumers move.

- `agents/factory.py:53` `from .tools.registry import get_tool_registry` → `from .tools.base import registry`;
  `:146` `get_tool_registry().get(t)` → `registry.get(t)`. Note `factory.py:146` is a **confirmed live
  `AttributeError`** today: `registry.py:9`'s class defines `get_tool`, never `get`
  (`scout-tools-duplicates.md:63`). The survivor's `get` raises `KeyError` on miss (`base.py:73`), which is the
  desired fail-fast — and is only safe because step 4 populated it.
- `tools/__init__.py:7-12` — drop the `.registry` import block, export `ToolRegistry` from `.base`. **Correcting
  the scout: the package currently exports the loser class**, so this line is a behaviour change for any importer
  of `...agents.tools.ToolRegistry`, not a no-op.
- keep `get_all_tools`/`get_web_tools` as thin aliases over `registry.by_tags("web")` for one commit, then remove.
  `get_web_search_tool`/`get_crawl_url_tool` live in `web_search.py:80`/`crawl.py:114` and survive independently.
- delete `tools/registry.py`.
- **D7 hazard, recorded not fixed:** `open_deep_search/utils.py:260` defines a *different, async* `get_all_tools`
  taking a `RunnableConfig`, consumed at `open_deep_search/graph.py:46,281,344,391`. Nothing crosses today. Do not
  unify the names.

- **Inbound:** steps 2, 4 (X6), X2.
- **Boot risk:** **YES** — removing exports from a package `__init__` that `shared/rag/graphiti/registry.py:40`
  imports eagerly. Paired restore: keep the aliases (above) so the public surface is unchanged in this commit, and
  revert is a one-line re-add.
- **Tests:** mandatory — a test that resolving an unknown tool name raises `KeyError` (the failure mode changed
  from silent `None` to loud `KeyError`; that is the point, and it must be pinned).
- **Proof:** `rg -n "tools\.registry|get_tool_registry" src/` → zero; `uv run ty check src/` → **28 or fewer**
  (baseline 46 minus step 2's 15 minus ≥3 of `factory.py`'s, `baseline-tests.md:170`);
  `uv run python -c "import app.main"` exits 0.

### 6. Rename the graphiti DTO to `AgentToolBundle`

`shared/rag/graphiti/registry.py:56` is a third class named `ToolRegistry` and it is a **different concept** — an
immutable Pydantic bundle of four pre-built tools, built once at lifespan (`build_tool_registry` `:98-122`). D6.1
says the file is **not deletable**. Rename the class so the repo has one `ToolRegistry`; update
`agent_saul/factory.py:10,182` and `agent_saul/graph.py:16,91`. Also fix the module docstring at
`registry.py:9,25`, which still points at `app.shared.agents.tools.registry` and at an
`app.state.saul_graph` assignment that exists nowhere — it is the single most misleading comment in the layer.

- **Inbound:** step 5 (do the rename after the survivor is the package's `ToolRegistry`, so no window has two
  meanings for one imported name).
- **Boot risk:** **YES** at import time if any of the four call sites is missed → `ImportError`. Paired restore:
  `ToolRegistry = AgentToolBundle` alias in the same module for one commit.
- **Tests:** optional.
- **Proof:** `rg -c "^class ToolRegistry" src/` → **1**; `rg -n "AgentToolBundle" src/` → 5 sites (definition +
  four importers); `uv run python -c "import app.main"` exits 0.

### 7. One `ToolResult`, with availability as a first-class field (Up#10)

`langchain_layer/agents/tools/idempotency.py:34` is the survivor (D6.1): `extra="forbid"`, `frozen=True`,
`success`/`data`/`error`/`metadata`, `ok()`/`fail()`. Work:

- the `shared/agents/tools/idempotency.py:11` twin dies with step 3;
- the third at `shared/rag/document_processing/models.py:318` has **exactly one importer, `todo_temp.py:8`**, which
  change 0 deletes (X5) — so this step deletes an orphaned class definition and migrates **no caller**;
- **add the availability signal** step 9 needs. `metadata` would technically carry it (`**meta` flows there), but a
  retry/escalation decision must not be spelled in a free dict. Add an explicit `unavailable: bool = False` and a
  third constructor `ToolResult.unavailable(reason, **meta)`.

**Serialization hazard, name it in `design.md`:** `IdempotencyGuard` persists `ToolResult` as JSON in Redis and
Postgres with a 30-day TTL (`idempotency.py:31`, `_POSTGRES_TTL_DAYS = 30`) and reads it back with
`model_validate_json` (`:77`). `extra="forbid"` means **new-schema rows read by old code raise**; adding a
defaulted field is forward-safe but not backward-safe. Either deploy readers before writers, or bump
`_REDIS_KEY_PREFIX`. Do not silently rely on rolling-deploy luck.

- **Inbound:** step 3 (twin gone), change 0's `todo_temp.py` deletion (X5).
- **Boot risk:** none.
- **Tests:** mandatory — round-trip `ToolResult.unavailable(...)` through `model_dump_json` /
  `model_validate_json`, and assert `extra="forbid"` still rejects unknown keys.
- **Proof:** `rg -c "^class ToolResult" src/` → **1**; `uv run pytest tests/unit/shared/ -k tool_result` passes in
  the summary line.

### 8. Structural idempotency keys (Trap2) — **and a conflict inside Trap2 itself**

`IdempotencyGuard.make_key` (`idempotency.py:65-76`) already does the right *cryptography*:
`hashlib.sha256(json.dumps({...}, sort_keys=True, default=str))` — deterministic, not the salted `hash()` builtin.
So Trap2 is not about the hash function. It is about **what callers put in `input_data`**, and today they put
content: `precedent_tools.py:82` passes `{"query": query, "user_id": ..., "num_results": ...}`.

**Trap2 as literally worded ("hash structural IDs, never content") is wrong for half the call sites.** For a
*read/search* tool, the query text **is** the cache identity — removing it makes two different questions collide
and return each other's answers. For a *write/side-effect* tool (`graphiti/write_clause_episodes.py:35` holds the
guard), the structural ID is the identity and hashing content means a re-worded retry double-writes the graph.

Resolution on record — split the key contract by tool kind:

| Tool kind | Key inputs | Rationale |
|---|---|---|
| read / search | `step_id`, `user_id`, **and** a canonicalised query (case-folded, whitespace-collapsed) plus structural scope (`doc_id`, `clause_id`) | content is the identity; canonicalise so trivial re-wording hits cache |
| write / side-effect | `step_id`, `user_id`, structural IDs **only** (`clause_id`, `doc_id`, `episode_id`) — never content | replay after `interrupt` must not double-write |

Enforce mechanically: make `make_key` keyword-only with an explicit `structural: dict` and an optional
`content: dict | None`, and have the write path pass `content=None`. A dict-shaped `input_data` cannot express the
distinction, which is why it drifted. Docs corroborate the need independently: nodes replay **from the beginning of
the node** on resume (`brief:ref:1628`) and the remedy is idempotency keys (`brief:ref:1614`), not a bigger retry.

- **Inbound:** step 7 (same module, keep the diff coherent).
- **Boot risk:** none. **Cache-invalidation risk: yes** — every existing key changes shape. Bump
  `_REDIS_KEY_PREFIX` in the same commit and accept one cold cache; do not attempt a dual-read.
- **Tests:** mandatory. Three cases: (a) same structural IDs + differently-worded query → **different** key for a
  read tool; (b) same structural IDs + different content → **same** key for a write tool; (c) key is stable across
  two separate `python -c` processes (guards against any future `hash()` regression).
- **Proof:** `uv run pytest tests/unit/shared/ -k idempotency_key` → the three cases pass in the summary line; and
  `rg -n "make_key\(" src/` shows every call site passing keyword args.

### 9. The invisible-failure register: unavailability must never be reported as absence

The worst defect in the repo, and it is two lines. `retrieve_statute_section.py:_fetch_statute_section` returns
`None` for **both** "no matching row" (`:159`) and "`SQLAlchemyError`" (`:170-172`), and `:87-92` converts `None`
into `ToolResult.fail("Section {x} of {y} not found in {z}")`. **A missing table is reported to the LLM as
"this section of law does not exist."** For a legal product that is not a bug, it is a fabricated legal conclusion.

Fix all four sites in one commit — they are one defect wearing four hats:

| Site | Today | Target |
|---|---|---|
| `retrieve_statute_section.py:170-172` | `except SQLAlchemyError → return None` | three-state return (`found` / `absent` / `unavailable`); `unavailable` → `ToolResult.unavailable("statute corpus unavailable")`, **never** the "not found" string |
| `search_legal_precedents.py:227-229` | `except SQLAlchemyError → return []` | propagate an `unavailable` flag; **`insufficient_basis` (`:110`) must not be computed from `len(graphiti) + len(statutes)` when the statute leg failed.** Today `total_sources` (`:109`) is graphiti-only on failure, so ≥2 graphiti hits set `insufficient_basis=False` and the compliance agent proceeds believing it has statutory basis it never retrieved. Add a distinct `basis_unknown` that is **true whenever any leg was unavailable**, and surface it in `data` |
| `precedent_tools.py:221-237` | `_vector_search_clauses` unconditional `return []` with a `TODO` at `:234` and **no log line** | log at `warning` with an explicit `not_implemented` reason and stop counting the pgvector leg in `total_sources` (`:129`) while the docstring at `:62` advertises it |
| `search_legal_precedents.py:179-180` | docstring says the fallback "lets you deploy before the statutes table is populated" | delete that sentence. It is the written-down permission slip for this whole failure class |

Also delete the dead `_MIN_SOURCE_THRESHOLD` reasoning path if the retarget (step 15) changes the source count
semantics — check, do not assume.

- **Inbound:** step 7 (X8 — needs `ToolResult.unavailable`).
- **Boot risk:** none.
- **Tests:** **mandatory, and this is the single most important test in change 3.** With a mocked `AsyncEngine`
  whose `connect()` raises `SQLAlchemyError`: assert `retrieve_statute_section` returns a result whose `error`
  string does **not** contain `not found`, and that `search_legal_precedents` sets `basis_unknown=True`. Zero
  coverage today, so there is no regression net other than the one this step writes.
- **Proof:** `uv run pytest tests/unit/shared/ -k statute_unavailable` → all new cases pass in the summary line; and
  `rg -n "not found in" src/app/shared/langchain_layer/agents/tools/` returns **zero** hits on a
  `SQLAlchemyError`-reachable path.

### 10. Disposition `rag_agent_advanced.py` — harvest, then delete

`shared/rag/rag_agent_advanced.py` returns `f"Search error: {e!s}"` as tool output at `:169,241,290,342,478`, which
is the string-as-error anti-pattern the docs rule out (`brief`: "use structured output everywhere for llm output,
tool output, MCP output", `ref:45`). But fixing it in place is wasted work: the module is **pydantic-ai, not
langchain**, it is imported by nothing, its entry point is a CLI (`run_cli()` `:517`), it queries a `match_chunks()`
function defined in **no migration and no source file**, and it imports `from ingestion.embedder import
create_embedder` (`:119,198,267,373`) — **a package that does not exist in this repo**, so every tool ImportErrors
on first call (`scout-tools-schema.md:72`).

Recommendation: **delete it, after harvesting two things that exist nowhere else** — `search_with_self_reflection`
(`:353`, grades results at `:420`, refines the query at `:460`) and `expand_query_variations` (`:52`) are the repo's
only iterative/agentic-RAG prior art, and `dispositions.md` routes agentic query rewriting to change 1's item 195.
Copy the algorithm into change 1's design notes, then delete. If the user wants it kept, the fallback is to move it
under `src/app/examples/` (per `CLAUDE.md`) so it stops looking like production code.

- **Inbound:** none. **Requires a user decision** — it is a deletion of ~600 lines that D4 would authorise
  (proven-empty caller set) but which no locked decision names explicitly.
- **Boot risk:** none (zero importers).
- **Tests:** none.
- **Proof:** `rg -l "rag_agent_advanced" src/` → zero; `uv run pytest` summary still **55 passed** (nothing covers
  it); `uv run ruff check src/` count does not increase.

### 11. Prompt assembly: one seam, the ordering rule, and the TOON brace collision (todo 1 + Up#6)

D6/decisions already settle that both paths were built and the defect is **adoption**. The measured shape is more
specific than "~30 bare-string sites vs a competing helper", and it inverts which one is the competitor:

- `render_prompt_sections` (`langchain_layer/prompts.py:145`) is the **dominant** helper — **26 call sites** across
  `agent_saul/prompts.py` (11), `open_deep_search/prompts.py` (8), `ingestion_kb/prompts.py` (4),
  `retrieval_kb/nodes.py` (3), `reconciliation/prompts.py` (1). It returns a bare `str`.
- `SystemPromptParts` (`prompts.py:19`) is the **richer but barely-adopted** one — **2 real sites**
  (`agents/factory.py:171`, `agents/registry.py:103,150`) plus the `AGENT_SYSTEM_PROMPT` constant (`:162`). It owns
  `build()` (`:99`), `Template(...).safe_substitute` (`:122`) and `to_chat_template()` (`:126`).

So do **not** migrate 26 sites onto `SystemPromptParts`. Make `render_prompt_sections` the section-assembly
primitive it already is, and make `SystemPromptParts` consume it, so there is one seam with two entry points and the
ordering rule is implemented **once**, inside `build()`.

**Up#6 "Lost in the Middle" ordering**, as a deterministic reorder in `build()`: standing instructions and the
output contract first; retrieved evidence in the middle **with the highest-salience items at the head and tail of
the evidence block**; the task restatement last. Assert it with a test on section order, not by eyeballing a prompt.

**The TOON brace collision — how it will be proved.** `serialize_to_toon` (`langchain_layer/models.py:224`, 16 call
sites, one definition — the defect is import inconsistency, not duplication) emits the mandated format
`key[N]{field1, field2}` (`brief:ref:54`). `ChatPromptTemplate` treats `{field1, field2}` as a template variable, so
any TOON payload passed through `to_chat_template()` raises `KeyError` at format time. The brief records
`{{`/`}}` escaping, `partial_variables`, and `string.Template` as **zero-hit gaps** in the repo corpus
(`brief:725-727`) — there is no house precedent to follow.

Two mechanisms, pick the second:

| Option | Pros | Cons |
|---|---|---|
| Escape `{`→`{{` at the injection boundary | one-line; keeps everything inside `ChatPromptTemplate` | escaping is lossy to reason about and must be applied at every one of the 16 sites; a missed site is a runtime `KeyError` in a prompt path with no tests |
| **Inject TOON as pre-formatted message content** — a `MessagesPlaceholder` slot or a ready-made `HumanMessage`, so the payload never passes through brace substitution | payload is bytes-exact by construction; no per-site discipline; matches the docs' own observable convention of building dynamic prompts outside the templating engine (`brief:276-280`) | requires the two prompt entry points to distinguish "template text" from "data payload" — which is the distinction that was missing |

**Proof of the fix (this is the answer to "say how you will prove it works"):** a unit test that builds a TOON
payload containing a literal `results[2]{name, score}` header, passes it through
`SystemPromptParts(...).to_chat_template().format_messages(...)`, and asserts (a) no exception, and (b) the rendered
message content contains the substring `results[2]{name, score}` **verbatim, with single braces**. A test that only
asserts "no exception" passes against a silently-mangled payload, so the substring assertion is load-bearing.

- **Inbound:** none (independent of the registry work). Sequence before step 12, which adds a section to the prompt.
- **Boot risk:** none.
- **Tests:** **mandatory** — section-ordering test plus the brace round-trip above. `prompts.py` has zero coverage.
- **Proof:** `uv run pytest tests/unit/shared/ -k "prompt_ordering or toon_braces"` → both pass in the summary line;
  and `rg -c "render_prompt_sections" src/` still shows the 26 sites unbroken.

### 12. Citation enforcement (Up#11)

Cheaper than it looks, because the type already exists: `agent_saul/state.py:103` defines
`class Citation(BaseModel, frozen=True)`. Work is (a) confirm/extend it to carry **claim / source / confidence**,
(b) add a model validator that rejects an assertion-bearing finding whose citation list is empty, (c) add the
"every assertion carries a citation" clause to the prompt sections from step 11.

Target the three output models that carry legal assertions: `RiskFinding` (`state.py:203`), `ComplianceFinding`
(`:219`), `GroundingVerificationOutput` (`:239`). Fail **validation**, not a log line — for a legal product an
uncited assertion is the failure mode, and a warning is indistinguishable from success at 3 a.m.

- **Inbound:** step 11 (the prompt clause), step 13 (the schema must be what `response_format` enforces).
- **Boot risk:** none.
- **Tests:** **mandatory** — a `RiskFinding` with `citations=[]` must raise `ValidationError`; one with a valid
  `Citation` must construct. Also assert `confidence` is bounded, or the field is decorative.
- **Proof:** `uv run pytest tests/unit/shared/ -k citation_required` → passes in the summary line; and
  `uv run python -c "import json,app.shared.langgraph_layer.agent_saul.state as s; print('citations' in json.dumps(s.RiskFinding.model_json_schema()))"`
  prints `True`.

### 13. `response_format` on the agents; stop discarding usage metadata (Up#9, cheap half only)

`dispositions.md` SPLITs Up#9: the cheap half is IN, the Accept/Retry/Escalate state machine is **DEFERRED** and
becomes a Non-Goal in `design.md`. The cheap half is already half-present and half-missing:

- **Present:** `langchain_layer/agents/factory.py:189` already passes `response_format=spec.response_format` into
  `create_agent`.
- **Missing:** `agent_saul/factory.py:114,120,126` — the orchestrator, risk, and compliance `create_agent` calls
  pass **no `response_format`**, while their output models (`RiskAnalysisOutput` `state.py:213`, `ComplianceOutput`
  `:232`) already exist. Add them.
- **Missing:** the Flash chains at `agent_saul/factory.py:132+` call `flash_llm.with_structured_output(QnAOutput)`
  **without `include_raw=True`**, so token counts and the raw `AIMessage` are discarded (`brief:ref:942-943`,
  `04-...:79`). **Caveat that must be handled in the same commit:** `include_raw=True` changes the return to a dict
  (`{"raw", "parsed", "parsing_error"}`), which invalidates the `cast("Runnable[list[Any], QnAOutput]")` casts
  wrapping every one of those chains. Either unwrap at the call site or keep `include_raw=False` where the metadata
  is genuinely unused — but decide per chain and write the reason down.
- `ProviderStrategy.strict` requires `langchain>=1.2` and installed is **1.2.12** (`brief:231-234`), so strict mode
  is available. Prefer passing the bare schema type and letting LangChain pick provider-native output
  (`brief:226-227`) over hand-picking a strategy.

- **Inbound:** step 12 (the schemas being enforced must already require citations, or this locks in the weaker
  contract), step 6 (touching `agent_saul/factory.py`, which the rename also edits — sequence to avoid a conflict).
- **Boot risk:** none.
- **Tests:** mandatory for the `include_raw` shape change (it is a silent type change with `cast` hiding it from
  `ty`); optional for `response_format` itself.
- **Proof:** `rg -n "create_agent\(" -A4 src/app/shared/langgraph_layer/agent_saul/factory.py` shows
  `response_format=` on all three; `uv run ty check src/` count does not increase; `uv run pytest` summary shows the
  new chain-shape test passing.

### 14. Middleware owns retries; `ToolNode(handle_tool_errors=...)` (172, narrow) — **and the tenacity conflict**

Scope, per `dispositions.md`: `@wrap_model_call` plus tool-error handling. Nothing else from item 172.

The seam already exists in this repo — `middlewares/guardrails.py:49,159` already uses `@wrap_model_call` (with
`# type: ignore`, which is 5 of the 46 `ty` diagnostics, `baseline-tests.md:168`). The retry middleware is the
missing sibling; `middlewares/` holds only `evals.py` and `guardrails.py` today. `ToolNode` has **zero occurrences
in `src/`**, so `handle_tool_errors` is genuinely absent — and because saul uses `create_agent` (which builds its
own tool node), reaching it means either `create_agent`'s tool-node configuration or passing a pre-built `ToolNode`.
**Verify which against installed langchain 1.2.12 before writing the step; do not assume the 0.2-era signature.**

**Conflict with sub-todo (j), surfaced not resolved.** Sub-todo (j) asks for `tenacity`. The brief finds `tenacity`,
`RetryPolicy`, and `.with_retry()` at **zero mentions** across the entire repo doc corpus (`brief:301-305`), while
`tenacity` 9.1.4 **is** installed and already used at I/O-client boundaries (`kb_retry.py`, `connections/redis.py`,
`razorpay_client.py`). More decisively, `brief:ref:1633` forbids wrapping `interrupt` in a bare `try/except` —
`interrupt` pauses by *raising*, so a catch-all swallows it and the graph never pauses. `tenacity`'s default
`retry_if_exception_type(Exception)` is exactly that catch-all. And the replay trap compounds it: a node restarts
**from its first line** on resume (`brief:ref:1628`), so a `tenacity` attempt counter held in a node local is not a
checkpointed channel — the retry budget silently multiplies on every replay.

**Recommendation on record (matches `dispositions.md`'s):** middleware owns model and tool retries; `tenacity` stays
at I/O-client boundaries where it already is. Sub-todo (j)'s intent (bounded retry with backoff) is honoured; its
named vehicle is not the documented one. **This is a flag for the user, not a decision I am taking alone** — if the
user wants `tenacity` inside graph nodes, the plan needs a step to prove `interrupt` still propagates.

- **Inbound:** step 7 (`ToolResult.unavailable` is what a retry decision reads), step 9 (retrying a call that
  reports "not found" for a missing table retries nothing useful).
- **Boot risk:** none.
- **Tests:** **mandatory, two cases.** (a) a transient failure is retried up to N and then surfaces; a permanent one
  is not retried. (b) **the interrupt-safety test**: a middleware-wrapped call that raises LangGraph's interrupt
  exception must **propagate**, not be retried. Without (b) this step can ship the exact bug `ref:1633` warns about,
  and nothing else in the suite would catch it.
- **Proof:** `uv run pytest tests/unit/shared/ -k "model_retry or interrupt_propagates"` → both pass in the summary
  line; `rg -n "handle_tool_errors" src/` → at least one hit; `uv run ty check src/` count does not increase — and
  check specifically that removing any `# type: ignore` from `guardrails.py` did not convert a suppressed error into
  a live one, since 7 of the 46 baseline diagnostics are `unused-type-ignore-comment` and this step perturbs them.

### 15. Retarget the two statute tools off the nonexistent `statutes` table — **change-2 gated, floating**

`statutes` has **no model and no migration** — confirmed against all 10 files in `src/alembic/versions/`. Raw SQL at
`search_legal_precedents.py:182-200` and `retrieve_statute_section.py:128-146`. Retarget onto
`UnifiedDocument`/`UnifiedChunk`, using `scout-tools-schema.md:36-37`'s column map.

Three things make this bigger than a table rename:

1. **The FTS engine changes.** The old SQL is `to_tsvector('english', body)` + `ts_rank`. The target's full-text
   index is `pg_textsearch` BM25 (`chunks_bm25_idx`, `a71f0d7d9c12:102`), queried as
   `c.content <@> to_bm25query(:query, 'chunks_bm25_idx')` — the working reference is
   `features/search/repository.py:415-419`, in scope per D5.1, with RRF fusion at `features/search/fusion.py:28`
   (`k=60`). **Harvest that; a third implementation of BM25 in this repo would be a planning failure.**
2. **Schema asks A1 and A2** (see Ordering constraints). Without A1, `act_name` / `section_ref` / `year` have
   nowhere to live; without A2, `retrieve_statute_section` sequential-scans.
3. **Alembic ambiguity.** `2bc7726317f6` has two children (`8a7d9b1c2e3f`, `a71f0d7d9c12`) and the unified
   `documents`/`chunks` migration sits on the **unmerged** head, so `alembic upgrade head` is ambiguous today. The
   head merge is change 0's; this step consumes it.

**Positioning:** this step is gated only by change 2 (X3), not by steps 16-18. If change 2 slips, **do not block** —
steps 16-18 proceed and the tools honestly report `unavailable` thanks to step 9. That is the whole reason step 9
comes first: an honest failure is a shippable state; a fabricated legal conclusion is not.

- **Inbound:** change 2's `UnifiedDocument`/`UnifiedChunk` consolidation + A1 + A2; change 0's alembic head merge;
  step 9 (the availability semantics this rewrites into).
- **Boot risk:** none.
- **Tests:** mandatory — the SQL must at minimum be asserted to compile and bind, and the column map asserted
  against the ORM models so a change-2 rename breaks a test rather than production. A live-DB integration test is
  better but no database is running (`baseline-tests.md` service matrix: **zero** containers up), so make the unit
  test the gate and mark the integration test as follow-up.
- **Proof:** `rg -n "FROM statutes" src/` → **zero**; `uv run pytest tests/unit/shared/ -k statute_sql` passes in the
  summary line; and `uv run alembic upgrade head` resolves to a single head (consumes change 0's merge).

### 16. Bind the tools to the agents; add the hydration node; hoist `schema_version` (153 + Up#7)

Three things that only make sense together, because each is a lie the state layer currently tells.

**(a) The `tools=[]` lie.** `agent_saul/factory.py:114,120,126` pass `tools=[]` with `# TODO: add ... when
available  # noqa: FIX002` on all three `create_agent` calls, while `build_saul_graph(:91)` faithfully threads a
`tool_registry` down through `_build_graph_nodes(:95)`. **No scout reported this.** Bind the four legal tools from
the (now renamed, step 6) bundle to the agents that `graphiti/registry.py:78-95`'s own comments assign them to:
`compliance_tools` → `search_legal_precedents` + `retrieve_statute_section`; `risk_tools` → `query_knowledge_graph` +
`get_obligation_chain`.

**(b) The hydration node.** `agent_saul/state.py:9` documents that "`schema_version` guards the AsyncPostgresSaver
hydration node", `:322` declares the field — and **no such node exists**. Add it as the first node after
checkpoint load: read `state["schema_version"]`, and on a mismatch either upgrade deterministically or fail with a
typed exception. Never proceed on an unknown version. The docs independently require this: *"always normalise agent
state after fetching from checkpointer so that there is no version mismatch"* (`brief:ref:105`), and a checkpoint
written under one schema and read after the schema changed is called out as a live hazard for this refactor
(`brief:446-450`).

**(c) The magic number.** `features/agent_saul/service.py:401` writes `"schema_version": 1` as a literal while
`state.py:384` defaults it to `1` separately. Two independent `1`s that must agree is a bug waiting for a bump.
Hoist a single `LEGAL_AGENT_STATE_SCHEMA_VERSION` constant in `state.py` and reference it from both.

**Sub-todo (i)'s conflict is resolved here, and it is smaller than it looked.** `LegalAgentState`
(`state.py:317`) is **already** a `TypedDict` with `messages: Annotated[list[BaseMessage], add_messages]` (`:329`)
and `operator.add` siblings (`:343-345,367`) — i.e. it already matches what the docs mandate
(`brief:ref:1341-1345`, *"Pydantic models and dataclasses are no longer supported"*). **No state conversion is
needed.** What is missing is the *handoff convention*: the documented form is an `AIMessage` carrying a
`transfer_to_*` tool call with a router edge reading it (`brief:ref:1473-1479`), and saul instead routes via
`add_conditional_edges` on custom router functions (`graph.py:50,55,66,74`) with **zero** `transfer_to_*` or
`Command(goto=...)` anywhere. Standardise the envelope (one helper that constructs the handoff `AIMessage`, one
router that reads it) rather than renaming the state class. Also set a `recursion_limit` — `brief:ref:1492` requires
it and `ref:1471` warns there is no loop detection outside `SupervisorState`.

- **Inbound:** steps 5, 6 (a populated, single-named registry), step 9 (bind tools only after they stop lying),
  step 13 (same file, `agent_saul/factory.py` — sequence to avoid conflicts).
- **Boot risk:** low. `graph.py` is only imported by callers that themselves have no caller — but if step 18 lands
  first this becomes a startup-path edit. Keep 16 before 18.
- **Tests:** **mandatory.** (a) a checkpoint payload with `schema_version=0` is upgraded or rejected
  deterministically — never silently accepted; (b) the handoff helper produces an `AIMessage` the router routes on;
  (c) the three agents receive non-empty tool lists.
- **Proof:** `rg -n "tools=\[\]" src/app/shared/langgraph_layer/agent_saul/factory.py` → **zero**;
  `rg -n '"schema_version": 1' src/` → zero (constant referenced instead);
  `uv run pytest tests/unit/shared/ -k "hydration or handoff"` → passes in the summary line.

### 17. `Field(description=...)` on agent config and tool argument schemas (todo g)

D-record: todo (g) is **not** a Pydantic deprecation fix (already resolved by an earlier reorg); it is adding
`Field(description=...)` to agent config models, starting with `AgentSpec`. Verified: `AgentSpec`
(`langchain_layer/agents/factory.py:~95-127`) documents all 16 of its fields in **trailing `#` comments** —
`temperature`, `memory_backend: str = "memory"  # memory | postgres | redis`, `max_tokens_before_summary`, etc. —
none of which reach a JSON schema or a validation error message.

Extend the same pass to tool argument schemas, where the docs are emphatic and the payoff is model behaviour, not
tidiness: *"define a Pydantic model and pass it to the `@tool` decorator... effectively 'forcing' the LLM to adhere
to your structure"* (`brief:ref:1481-1490`) and note 1 at `ref:43`, *"add Field description for tool instead of
simple docstrings."* Also adopt `InjectedToolArg` for `user_id` — `brief:ref:1480` names it as the documented way to
pass user scope into a tool **without exposing it to the model**, and every one of the four legal tools currently
takes `user_id` as an ordinary LLM-visible argument, which is both a prompt-injection surface and a waste of tokens.

- **Inbound:** steps 12, 13, 16 (so the pass covers the schemas those steps add rather than needing a second pass).
- **Boot risk:** none.
- **Tests:** optional but cheap and self-maintaining — iterate `AgentSpec.model_fields` and assert every field has a
  non-empty `description`; do the same over each tool's `args_schema.model_json_schema()["properties"]`. A test is
  the only thing that stops the next field being added without one.
- **Proof:** `uv run pytest tests/unit/shared/ -k field_descriptions` → passes in the summary line; and
  `uv run python -c "from app.shared.langchain_layer.agents.factory import AgentSpec; print([n for n,f in AgentSpec.model_fields.items() if not f.description])"`
  prints `[]`.

### 18. Restore the lifespan wiring — **last, and behind a flag**

Only now. `lifespan.py` contains no match for `tool_registry`, `idempotency`, `saul_graph`, or `IdempotencyGuard`;
the block that would build them (`:235-249`) is commented out, and `graphiti/registry.py:98`'s `build_tool_registry`
has zero callers. Restoring it means constructing `IdempotencyGuard(redis, db_engine)`, calling
`build_tool_registry(...)`, calling `build_saul_graph(...)`, and assigning `app.state.saul_graph`.

**This step is what makes every latent failure in this layer live.** That is why it is last, and why it ships behind
a settings flag (e.g. `SAUL_GRAPH_ENABLED`, default `False`) in the same commit that adds it, flipped in a separate
follow-up commit once the health probe (change 0's item 198.2) reports Graphiti green. Note `lifespan.py:220-223`
already degrades silently — a Graphiti startup failure sets `app.state.graphiti = None` and continues — so with the
flag on and Graphiti down, the compliance path runs with **both** backends dead. Step 1's 503 guard plus step 9's
`unavailable` semantics are what make that state honest rather than fabricated; both must be in before the flip.

- **Inbound:** **change 1's item 138 residue (a)** — `app.state.langgraph_checkpointer` must be set, since
  `build_saul_graph` takes a checkpointer (X4). Plus steps 1, 5, 6, 9, 16.
- **Boot risk:** **YES, and of a different kind** — this adds work to startup that can raise, on a code path that
  has never executed. Paired restore: the flag itself (`SAUL_GRAPH_ENABLED=False` restores today's behaviour with no
  revert), plus step 1's guard means an unset `app.state.saul_graph` is a clean 503 rather than a 500.
- **Tests:** mandatory — a lifespan test asserting that with the flag off nothing is constructed, and with the flag
  on and Graphiti unavailable, startup **still completes** and the route returns 503 rather than the process dying.
- **Proof:** with the flag on, `uv run python -c "import app.main"` exits 0; a startup log line names the graph;
  `uv run pytest tests/unit/ -k lifespan_saul` passes in the summary line; and with the flag **off**,
  `uv run pytest` summary is unchanged from the step-17 run.

## Boot-risk ledger

Every step where `import app.main` might fail, with its paired restore. Steps not listed cannot break import.

| Step | Why boot is at risk | Blast radius | Paired restore |
|---|---|---|---|
| **3** — delete `shared/agents/**` | `shared/rag/graphiti/registry.py:41-46` eagerly imports the four `make_*_tool` factories from the `tools` package; `get_obligation_chain.py:29` and `precedent_tools.py:21,22` import the stub tree. Deleting it first is a hard `ImportError` at package-import time. | **total** — `app.main` does not import | Land **step 2** first (three import lines). If already deleted: `git revert` the deletion commit, apply step 2, re-delete. |
| **5** — delete `tools/registry.py`, change package exports | `tools/__init__.py:7-12` is the only re-export of `ToolRegistry`/`get_all_tools`/`get_tool_registry`/`get_web_tools`, and `graphiti/registry.py:40` imports the package eagerly. | **total** | Keep `get_all_tools`/`get_web_tools` as thin aliases over `registry.by_tags("web")` in the same commit, so the public surface is unchanged and revert is a one-line re-add. |
| **6** — rename the graphiti `ToolRegistry` | four importers (`agent_saul/factory.py:10,182`, `agent_saul/graph.py:16,91`); one missed site is `ImportError` at import. | **total** | `ToolRegistry = AgentToolBundle` alias in the same module for one commit. |
| **18** — restore the lifespan wiring | different kind: not an import failure but **startup** failure. Constructs `IdempotencyGuard`, `build_tool_registry`, `build_saul_graph` on a path that has never executed; any raise aborts startup. | app starts but is unhealthy | `SAUL_GRAPH_ENABLED=False` restores today's behaviour with **no code revert**; step 1's guard turns an unset graph into a 503. |

**The last step at which boot is at risk from the X1 cycle is step 3.** After step 3 the `shared/agents/**` tree is
gone and no import points at it. Steps 5 and 6 are fresh, self-inflicted risks with their own aliases; step 18 is a
runtime risk, not an import one.

Ordered resolution of the cycle, stated once: **step 2 (rewrite three importers) → step 3 (change 0 deletes the
tree).** Change 0's `tasks.md` must reference change 3's step-2 task by number, and change 3's `design.md` must state
that its first task is a predecessor of another change's task. Any other order raises `ImportError` before FastAPI
constructs.

## Conflicts surfaced

Five, none resolved unilaterally. Each has a recommendation on record and a named place it must be written down.

| # | Conflict | Evidence | Recommendation | Lands in |
|---|---|---|---|---|
| **C1** | Sub-todo (i) names `MessagesState`. | The docs never use it — one descriptive mention at `brief:ref:1479`. They mandate `TypedDict` + `Annotated[list, add_messages]` plus sibling channels, *"Pydantic models and dataclasses are no longer supported"* (`ref:1341-1345`). Documented handoff is an `AIMessage` with a `transfer_to_*` tool call + a router edge (`ref:1473-1479`). **And `LegalAgentState` (`state.py:317`) already has the mandated shape.** | Honour the intent (standardise A→B passing), reject the vehicle. Build the handoff envelope + router convention; convert nothing. | `design.md` Decisions; step 16 |
| **C2** | Sub-todo (j) asks for `tenacity`. | Zero mentions of `tenacity`/`RetryPolicy`/`.with_retry()` in the corpus (`brief:301-305`), though it is installed at 9.1.4 and used at I/O boundaries. `ref:1633` forbids bare `try/except` around `interrupt` (which pauses by raising) — condemning `tenacity`'s default catch-all inside nodes. Replay restarts a node from line 1 (`ref:1628`), so a node-local attempt counter multiplies the retry budget. | Middleware (`@wrap_model_call`) owns model/tool retries; `tenacity` stays at I/O-client boundaries. **If the user overrides, add a step proving `interrupt` still propagates.** | `design.md` Decisions; step 14 test (b) |
| **C3** | Trap2 — "hash structural IDs, never content" — **contradicts itself across tool kinds.** | `make_key` (`idempotency.py:65`) already uses `sha256` + `sort_keys` (deterministic), and callers pass content (`precedent_tools.py:82` passes `query`). Dropping content makes two different search queries return each other's cached answers; keeping it makes a re-worded write retry double-write. | Split the key contract by tool kind (table in step 8): content **in** for read/search (canonicalised), structural IDs **only** for writes. | `design.md` Decisions; step 8 |
| **C4** | Todo (1) frames prompts as "Template vs ChatPromptTemplate"; the real defect is adoption — but **in the opposite direction from the brief's framing.** | `render_prompt_sections` has **26** call sites and returns `str`; `SystemPromptParts` (which owns `safe_substitute` at `:122` and `to_chat_template()` at `:126`) has **2**. The "competing helper" is the dominant one. | Do not migrate 26 sites. Make `render_prompt_sections` the section primitive and `SystemPromptParts` its consumer; implement Up#6 ordering once inside `build()`. | `design.md` Decisions; step 11 |
| **C5** | TOON's mandated format `key[N]{field1, field2}` (`ref:54`) collides with `ChatPromptTemplate` brace substitution, and the repo corpus documents **no** escaping convention (`brief:725-727`: `{{`/`}}`, `partial_variables`, `string.Template` all zero-hit). | 16 `serialize_to_toon` call sites; `to_chat_template()` would raise `KeyError` on any of them. | Inject TOON as pre-formatted message content (never through brace substitution) rather than escaping at 16 sites. Prove with a **verbatim substring** assertion, not merely "no exception". | `design.md` Decisions; step 11 test |

Two smaller items recorded rather than fixed, both per D7/D13:

- **D7 hazard:** `open_deep_search/utils.py:260` defines a second, **async** `get_all_tools` taking a
  `RunnableConfig`. Nothing crosses today. Step 5 must not unify the names.
- **`MemoryManager` (`factory.py:69-74`)** is a self-documented stub whose two called methods —
  `inject_long_term_context` (`:246`) and `save_session` (`:256`) — **do not exist**, guarded by
  `enable_long_term_memory` which defaults `True` (`:113`). `factory.py:146` fires first and masks it. Not a
  duplicate, so not in the unification workstream; it is a **replace** candidate that belongs to change 4's memory
  build. Record as a Non-Goal here with the `AttributeError` named, or step 5 will look like it fixed `factory.py`.

## Openspec mapping

**Checked `openspec/specs/` first, as `config.yaml:39-43` requires.** Twenty capabilities exist:
`cognee-v1-api`, `datetime-utc-cleanup`, `llm-injection`, `mcp-context-di`, `mcp-directory-restructure`,
`mcp-server-codemode`, `mcp-server-composition`, `mcp-server-pagination`, `mcp-server-prompts`,
`mcp-server-resources`, `mcp-telemetry`, `mcp-testing`, `noqa-documentation`, `outbox-helper-extraction`,
`pattern-matching-standard`, `session-required`, `settings-validation`, `test-mock-isolation`,
`transactional-outbox`, `typed-exception-handling`. **None covers agent tools, agent state, prompts, or retrieval.**
Every delta below is therefore **ADDED under a new capability**, and each new capability's `spec.md` needs a
`## Purpose` of 50+ characters.

Change ID: **`agent-tools-unification`** (bare slug per D12 — the archive adds the `YYYY-MM-DD-` prefix).
Class **L**; `design.md` mandatory. `.openspec.yaml` must say `schema: spec-gated` (**not** `spec-driven` — the
cognee change's stale value is the trap, per conventions FINDING 2), `created: 2026-08-17`, and **no**
`skip_specs: true` — this change has plenty of observable behaviour change.

### New capabilities

| Capability | Covers steps | Observable behaviour it contracts |
|---|---|---|
| `agent-tool-registry` | 4, 5, 6 | one tag-based registry; resolving an unknown tool name fails loudly rather than returning nothing |
| `agent-tool-contract` | 7, 8, 9, 10 | one result envelope; **backend unavailability is never reported as absence**; idempotency keys are structural for writes and content-canonical for reads |
| `legal-corpus-retrieval` | 15 | statute and precedent retrieval resolves against the unified corpus; a query that cannot reach the corpus says so |
| `agent-prompt-assembly` | 11 | one prompt seam; deterministic section ordering; serialized data payloads survive prompt rendering byte-exact |
| `agent-structured-output` | 12, 13 | agent outputs are schema-validated; **an assertion without a citation is rejected**; usage metadata is retained where captured |
| `agent-runtime-resilience` | 1, 14, 18 | model and tool retries are bounded and never swallow a pause; an unavailable agent graph yields a service-unavailable response, not an internal error |
| `agent-state-handoff` | 16 | agent-to-agent handoff uses one message envelope with an explicit router; a state schema version mismatch is resolved or refused, never ignored |

### Modified capabilities

**None.** Two existing specs are adjacent and must be **cited, not edited**:
`typed-exception-handling` governs step 1's and step 9's exception choices, and `pattern-matching-standard` governs
step 9's three-state return. **Both are among D12's six pre-existing `validate --all` failures**, so editing them
risks confusing the acceptance signal. Cite them in `design.md` Decisions instead.

### Acceptance criterion

Per D12: **`openspec validate --all` shows no new failures beyond the existing 6** (baseline 16 passed / 6 failed of
22). Never "validate --all passes". Formatting traps that fail *silently*: scenario headers take **exactly four
hashtags** (`schema.yaml:164-165`), every requirement needs ≥1 scenario, and `MODIFIED` must copy the entire original
requirement block. Artifact order is gated: `proposal` → `specs` + `design` → **`review` (with a `VERDICT:` line)** →
`tasks`. `review.md` is written by a **fresh subagent, never the author** (`schema.yaml:321`; the gate is
instructional, not CLI-enforced — we honour it by choice per D12).

### ADR

One candidate that outlives the change: **the tool-result contract** — that a tool must distinguish *absent* from
*unavailable*, and that unavailability is a first-class field rather than a log line. Everything built on these tools
depends on it, and it is the fix for the worst defect in the repo. The other decisions (registry survivor, prompt
seam, retry ownership) are change-scoped; `adrs.md` should carry the one and say so about the rest.

## Expected effect on the baselines

State this up front so the verifier is not guessing.

| Check | Baseline | Expected after change 3 | Reasoning |
|---|---|---|---|
| `ty check src/` | **46** | **≤28**, and **≤31 after step 2 alone** | `baseline-tests.md:167-170`: 11 diagnostics in `precedent_tools.py` + 4 in `get_obligation_chain.py` = **the 15 in `agents/tools/`**, all caused by the stub imports (`scope.top_k` on a `str`, `ToolResult.ok`/`fail` on a class lacking them). Step 2 should clear all 15. `factory.py`'s 3 clear with step 5. **Do not expect below ~25** without deliberately touching `guardrails.py`'s 5 — and note step 14 perturbs the 7 `unused-type-ignore-comment` diagnostics in both directions. |
| `ruff check src/` | **125** → **123** after change 0 (D11) | **≤123**, flat is acceptable | change 3 adds no syntax fixes. Step 10's deletion may remove a few. Two of the 125 are `invalid-syntax` in `todo_temp.py` and belong to change 0. |
| `pytest` | **55 passed** / 22 failed / 13 errors, **exit 1** | **≥75 passed**, same failures | ~20 new tests across the six mandatory-test steps. **The suite still exits 1** because `--cov-fail-under=80` vs 18.38% coverage — compare the summary line, never `$?`. |
| coverage | **18.38%** | measurably higher, no target | these modules are at **0%** today, so every new test is net-new coverage. Do not set a numeric goal; 80% is unreachable in one change and pretending otherwise invites padded tests. |
| `ast-grep scan src/` | **4 errors**, exit **0** | **4**, unchanged | none of the five vendored rules (`no-raw-httpexception`, `no-raise-app-error-mapper`, …) touch this layer. Exit code is not a gate — compare the printed count. |
| `openspec validate --all` | 16 passed / **6 failed** | no new failures | D12. |

**Where new tests are mandatory, not optional** (these modules have **zero** coverage, so there is no existing net):
steps **1** (503 guard), **4** (registry non-empty — the exact failure this change exists to prevent), **5**
(`KeyError` on unknown tool), **7** (envelope round-trip), **8** (three key-semantics cases), **9** (**the most
important test in the change** — unavailability is not absence), **11** (section ordering + verbatim TOON braces),
**12** (uncited assertion rejected), **14** (retry bounds **and** interrupt propagation), **16** (schema-version
mismatch + handoff envelope), **18** (startup with flag off/on). Optional: steps 2, 6, 10, 15's integration half, 17.

## Risks

`[Risk] → Mitigation`, in `design.md` format.

- **[Change 0 deletes `shared/agents/**` before change 3's step 2 lands, and the app will not import.]** → Change 0's
  `tasks.md` cites change 3's step-2 task by number; change 3's `design.md` states the cross-change predecessor
  explicitly. The ledger names step 3 as the last boot-risk point of the cycle.
- **[Step 5 adopts a registry that is empty at runtime, turning silent `None` into `KeyError` for every tool.]** →
  Step 4 populates and *proves* the singleton non-empty before any consumer moves; the proof is an executable
  one-liner, not a code read.
- **[Change 2 slips and step 15 blocks the rest of change 3.]** → Step 15 is explicitly floating. Step 9 makes the
  un-retargeted tools honest, so shipping steps 16-18 without 15 is a defensible state.
- **[Step 18 makes previously-invisible failures start firing, in production, at once.]** → Ships behind
  `SAUL_GRAPH_ENABLED=False` in the same commit; flipped separately, only after change 0's health probe (item 198.2)
  reports Graphiti green. Step 1 turns the unwired case into a 503.
- **[Zero test coverage means no regression signal for any refactor here.]** → Six steps carry mandatory tests, and
  the two riskiest structural steps (4, 5) have executable proofs independent of the suite.
- **[`ToolResult`'s new field breaks 30-day-TTL idempotency rows on a rolling deploy, because `extra="forbid"`.]** →
  Bump `_REDIS_KEY_PREFIX` (step 8 already bumps it for the key-shape change — do both in one commit) and accept one
  cold cache. Do not attempt a dual-read.
- **[Step 8's key-shape change silently invalidates every cached tool result.]** → Same mitigation; state it in
  `design.md` Migration Plan rather than discovering it as a latency spike.
- **[Step 14 ships a retry wrapper that swallows `interrupt`, and HITL silently stops pausing.]** → Test (b) of step
  14 exists solely for this; it is the one test that cannot be dropped for time.
- **[Step 13's `include_raw=True` changes a return type that four `cast(...)` calls currently hide from `ty`.]** →
  Decide per chain and test the shape; do not blanket-apply.
- **[Renaming the graphiti `ToolRegistry` (step 6) collides with change 4, which also touches `graphiti/`.]** → Land
  step 6 before change 4 starts, or agree the name in `adrs.md` first.

## Fog

Open, with what it would take to close each. None of these blocks starting at step 1.

1. **`docs/relay/plan-change0.md` does not exist.** The orchestrator's brief said to read it if present; it is not in
   `docs/relay/` (15 files, none named `plan-change0.md`). So change 0's exact task numbering — which step 3 above must
   cite, and which must cite step 2 back — **cannot be written yet**. Close by: authoring change 0's plan, then adding
   the two cross-references in both directions. **This is the highest-priority Fog because the X1 cycle is only safe
   if both plans name each other.**
2. **Does `shell.py` do work at import time?** Step 4 proposes importing it from `tools/__init__.py` so its five
   `@register_tool` decorators run. If the module opens a subprocess, reads a config, or touches the filesystem at
   module scope, that becomes startup work in every process including celery workers. Close by: reading
   `shell.py:1-40` before writing step 4. **Fallback that needs no reading: register the five explicitly from
   `__init__.py` instead of relying on import side effects** — which is arguably better anyway.
3. **How `handle_tool_errors` is reached under langchain 1.2.12 + `create_agent`.** `ToolNode` has zero occurrences in
   `src/`, and saul uses `create_agent`, which constructs its own tool node. Whether the option is exposed via
   `create_agent`, requires passing a pre-built `ToolNode`, or has moved entirely is unverified — the brief's citation
   (`ref:38`) is a cross-checked note, not a version-pinned signature. Close by: reading
   `.venv/.../langchain/agents/` and `langgraph/prebuilt/tool_node.py` at the installed versions. **Do not write step
   14 against the 0.2-era signature.**
4. **Whether `statutes` and `match_chunks` exist in the deployed database.** No migration and no model define either,
   but **no database is running** (`baseline-tests.md` service matrix: zero containers, no port bound), so absence
   from the repo is the only evidence. If `statutes` was hand-created outside alembic, step 15 is a data migration
   rather than a retarget, and step 9's `unavailable` path may currently be firing on a *transient* error rather than
   a missing table. Close by: `\dt` against a real instance. **Inherited unclosed from `scout-tools-schema.md:84-85`.**
5. **Whether `clauses` survives.** `9f4a1b7c6d2e` added `chunk_id`/`chunk_text` to `clauses`, which looks like a
   half-finished migration toward `chunks`. `precedent_tools.py:221-237`'s stub is the only reader and it returns
   `[]`. If `clauses` is being retired by change 2, step 9's third row and step 15 merge. Close by: change 2's plan
   stating the disposition of `clauses` explicitly.
6. **Whether `get_research_agent` / `get_code_review_agent` are reached from outside `src/app`.** The scout swept
   `src/app` and `tests/` only. A notebook, a root script, or `src/app/examples/` could reach them, which would
   promote `factory.py:146`'s `AttributeError` from "rare" to "breaks on first request" and reorder steps 5 and 17.
   Close by: `rg -n "get_research_agent|get_code_review_agent"` over the repo root, `docs/`, and any notebooks.
7. **`ProviderStrategy` vs `ToolStrategy` per model.** `strict` needs `langchain>=1.2` and 1.2.12 is installed, but
   whether the configured Gemini models expose native structured output through `model.profile` on
   `langchain-google-genai` 4.2.1 is unverified — `brief:234-239` says profile data is read on `langchain>=1.1` and
   that a custom profile must be supplied when it is unavailable. If Gemini has no profile data, step 13's bare-schema
   approach silently falls back to tool-calling. Close by: `python -c "...init_chat_model(...).profile"`.
8. **Whether `RemoveMessage` / message-trimming is needed for the handoff envelope.** `brief:718-719` records
   `RemoveMessage`, `REMOVE_ALL_MESSAGES`, and `add_messages` ID-collision semantics as **zero-hit gaps** in the repo
   corpus, and note 8 (`ref:58`) asks to "trim/remove tool output in a multi-step agent conversation" — which step 16's
   envelope will make concrete once agents actually exchange messages. Deliberately **not** planned: it needs library
   source (`langgraph.graph.message.add_messages`) as the authority, and it is a follow-up, not a blocker.
9. **Item 151's `compact-middleware` / `langchain-collapse` / `langchain-cisco-aidefense`.** `dispositions.md` DEFERs
   all three and notes they partially overlap step 14's middleware work. Recorded here so the overlap is visible when
   step 14 is written; no work scheduled.

---

**Orchestrator correction (2026-08-17):** 2 occurrence(s) of the vendor name "VectorChord" were
replaced with `pg_textsearch` throughout this file. The BM25 index access method and `to_bm25query()` come from
**`pg_textsearch`** (Timescale/TigerData), verified present at version 1.3.0 on the live server. `vchord` 0.5.3
is available on that server but unused, and `vchord_bm25` is **not available at all** — so the earlier
attribution was not merely a naming slip, it named an extension the deployment cannot install. See
`docs/relay/findings-database.md` §3.
