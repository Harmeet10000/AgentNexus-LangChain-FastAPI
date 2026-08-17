# Scout: duplicate symbol definitions in the agent/tools layer

Decision in force: **ToolRegistry unifies in `langchain_layer`, tags preserved.**
`open_deep_search` is out of scope (recorded hazard only).

All paths relative to repo root `/home/harmeet/Desktop/Projects/langchain-fastapi-production`.

---

## Duplicate register

### 1. `IdempotencyGuard` — 2 definitions

| Definition | Lines | Surface |
|---|---|---|
| **SURVIVOR** `src/app/shared/langchain_layer/agents/tools/idempotency.py:56` | 56-207 (file 211) | `__init__(redis: Redis, db_engine: AsyncEngine)`, `@staticmethod make_key(step_id, input_data, user_id) -> str` (`:65`), `async get(key) -> ToolResult\|None` (`:78`), `async set(key, result, *, tool_name, user_id, thread_id, step_id)` (`:93`), private `_warm_redis_cache` (`:130`), `_get_from_postgres` (`:143`), `_set_in_postgres` (`:167`); module fn `_redis_key` (`:210`) |
| **STUB** `src/app/shared/agents/tools/idempotency.py:19` | 19-29 (file 29) | `model_config`, `_cache: dict[str, ToolResult]`, `async execute(key, fn, *args, **kwargs)`. **No** `make_key` / `get` / `set`. |

Co-located `ToolResult` also duplicated: survivor `langchain_layer/.../idempotency.py:34` (`extra="forbid"`, frozen, `metadata`, `ok`/`fail`) vs stub `shared/agents/tools/idempotency.py:11` (frozen, **no `metadata`, no `ok`/`fail`**). A third lives at `src/app/shared/rag/document_processing/models.py:318` (out of this scope; already logged in `docs/relay/todo-overlap.md:116`).

Import sites:

| Site | Currently imports | Needs |
|---|---|---|
| `src/app/shared/langchain_layer/agents/tools/get_obligation_chain.py:29` | `from app.shared.agents.tools.idempotency import IdempotencyGuard, ToolResult` | `from .idempotency import IdempotencyGuard, ToolResult` |
| `src/app/shared/langchain_layer/agents/tools/precedent_tools.py:22` | same stub path | `from .idempotency import IdempotencyGuard, ToolResult` |
| `src/app/shared/agents/tools/__init__.py:1` | re-exports stub | package deleted (see Overlap) |
| already correct: `search_legal_precedents.py:30`, `query_knowledge_graph.py:24`, `retrieve_statute_section.py:27` (`from .idempotency import ...`), `src/app/shared/rag/graphiti/write_clause_episodes.py:35` (absolute survivor path), `src/app/shared/rag/graphiti/registry.py:51` (TYPE_CHECKING, survivor path) | | no change |

Call-site changes needed in the two rewritten importers: **none beyond the import line.** Both already call the survivor API — `IdempotencyGuard.make_key(...)` at `get_obligation_chain.py:67` and `precedent_tools.py:80,188`; both also use `ToolResult.ok`/`fail`, which the stub lacks. After the import swap the call sites type-check as-is. Confirmed the survivor's `.get`/`.set` are used identically in the already-correct siblings (`query_knowledge_graph.py:69`, `retrieve_statute_section.py:66`, `search_legal_precedents.py:83`).

> Correction to Claim 1: the scout brief cited `.get` at `:77` and `.set` at `:104` of the *importer*. Those line numbers are the survivor's own definitions (`get`→`:78`, `set`→`:93`). The importers reference `make_key` and `ToolResult.ok/fail`, not `.get`/`.set` directly. The AttributeError is real; the cited lines were the wrong file.

### 2. `memory_scope` / `PRECEDENT_SCOPE` — 2 definitions

| Definition | Lines | Surface |
|---|---|---|
| **SURVIVOR** `src/app/shared/langchain_layer/agents/memory/memory_scope.py` | 238 | `MemoryEntityType:28`, `MemorySource:36`, `MemoryTimeFilter:42`, `MemoryScope:47`, `_coerce_*:96-113`, `_read_int_field:119`, `_build_scope:133`, and five `MemoryScope` constants: `RISK_SCOPE:152`, `COMPLIANCE_SCOPE:161`, `PRECEDENT_SCOPE:174`, `ORCHESTRATOR_SCOPE:183`, `GROUNDING_SCOPE:192`; `scope_from_router_decision:202`, `_read_iterable_field:223` |
| **STUB** `src/app/shared/agents/memory/memory_scope.py:1` | 1 (whole file) | `PRECEDENT_SCOPE = "precedent"` — a **`str`**, not a `MemoryScope` |

Import sites:

| Site | Currently imports | Needs |
|---|---|---|
| `src/app/shared/langchain_layer/agents/tools/precedent_tools.py:21` | `from app.shared.agents.memory.memory_scope import PRECEDENT_SCOPE` | `from ..memory.memory_scope import PRECEDENT_SCOPE` (or `from ..memory import PRECEDENT_SCOPE`, re-exported at `src/app/shared/langchain_layer/agents/memory/__init__.py:14,27`) |
| `src/app/shared/agents/memory/__init__.py:1` | re-exports stub | package deleted |
| `src/app/shared/rag/graphiti/subgraph.py:30` | survivor path (TYPE_CHECKING) | no change |

Call-site change: the survivor is a `MemoryScope` model, the stub a `str`. Any use of `PRECEDENT_SCOPE` in `precedent_tools.py` that treats it as a string must be re-typed. **Fog:** I did not read every use of `PRECEDENT_SCOPE` inside `precedent_tools.py` beyond the import; the planner must diff the attribute access.

Confirms Claim 2 (stub is one line).

### 3. `ToolRegistry` — 3 definitions, but NOT the split the brief assumed

| Definition | Lines | Surface | Verdict |
|---|---|---|---|
| **SURVIVOR** `src/app/shared/langchain_layer/agents/tools/base.py:58` | 58-95 (file 172) | `__init__` (`_tools: dict[str, BaseTool]`, `_tags: dict[str, set[str]]`), `register(t, *, tags)` `:68`, `get(name) -> BaseTool` raises `KeyError` `:73`, `all()` `:79`, **`by_tags(*tags)` `:82`**, `by_names(names)` `:88`, `names()` `:91`, `descriptions()` `:94`; module singleton `registry = ToolRegistry()` `:99`; decorator `register_tool(*tags)` `:125`; `make_structured_tool` `:149` auto-registers | keeps tags → matches user decision |
| `src/app/shared/langchain_layer/agents/tools/registry.py:9` | 9-39 (file 60) | `__init__` (`_tools: list[Any]`), `get_tools()` `:15`, `get_tool(name)` `:24` returns `None`, `@staticmethod get_search_tool()` `:31`, `@staticmethod get_crawl_tool()` `:36`; module singleton `_tool_registry` `:42`, `get_tool_registry()` `:45`, `get_all_tools()` `:53`, `get_web_tools()` `:58` (both bodies identical) | **no tags, no `.get`** |
| `src/app/shared/rag/graphiti/registry.py:56` | Pydantic `BaseModel`, immutable tool bundle for saul; `build_tool_registry(...) -> ToolRegistry` `:101-104` | different concept (a DTO of pre-built tools), same name |

> **Correction to Claim 3.** All three `ToolRegistry` definitions live *outside* `src/app/shared/agents/**` — there is no `shared/agents/tools/registry.py` on disk (only the docstring at `src/app/shared/rag/graphiti/registry.py:9` *claims* that path). "Unify in `langchain_layer`" is therefore ambiguous as written: **two** of the three are already in `langchain_layer`. The tag-preserving one is `tools/base.py:58`; the one `factory.py` imports is `tools/registry.py:9`.

The `factory.py:146` bug is **confirmed**: `src/app/shared/langchain_layer/agents/factory.py:53` imports `get_tool_registry` from `.tools.registry`, and `:146` calls `get_tool_registry().get(t)`. `registry.py:9`'s class defines `get_tool`, never `get` → `AttributeError: 'ToolRegistry' object has no attribute 'get'`.

Import sites:

| Site | Currently | Needs |
|---|---|---|
| `src/app/shared/langchain_layer/agents/factory.py:53` + `:146` | `from .tools.registry import get_tool_registry`; `get_tool_registry().get(t)` | import the `base.py` survivor: `from .tools.base import registry` (module singleton) → `:146` becomes `resolved_tools.append(registry.get(t))`. `registry.get` already raises `KeyError` on miss, which is the desired fail-fast. |
| `src/app/shared/langchain_layer/agents/tools/__init__.py:8,10,20,25` | re-exports both `ToolRegistry` (from `base`) and `get_tool_registry`, `get_all_tools`, `get_web_tools` (from `registry`) | the `__init__` **already** exports `ToolRegistry` from `base.py` (`:8`). Drop the `registry.py` re-exports once its consumers move. |
| `src/app/shared/langgraph_layer/agent_saul/factory.py:10` + `:182` | `from app.shared.rag.graphiti.registry import ToolRegistry` | out of the unification (graphiti DTO). Rename that class to avoid the name collision, or leave — planner's call. |
| `src/app/shared/langgraph_layer/agent_saul/graph.py:16` + `:91` | same graphiti DTO | same |
| `src/app/shared/langgraph_layer/open_deep_search/graph.py:46,281,344,391` | `get_all_tools` from `open_deep_search/utils.py:260` — a **different, async** `get_all_tools` | **HAZARD, out of scope.** Two `get_all_tools` names: `langchain_layer/.../registry.py:53` (sync) and `open_deep_search/utils.py:260` (async, takes `RunnableConfig`). Nothing crosses today; a careless unification of `registry.py` could collide. Schedule no work. |

### 4. `MemoryManager` — 1 definition, and it is a stub

`src/app/shared/langchain_layer/agents/factory.py:69-74`. Surface: `__init__(backend: str)` setting `self.backend` and `self.checkpointer = None`. Nothing else.

Claim 4 **confirmed in full**:
- instantiated at `factory.py:164-166` with `backend=spec.memory_backend`
- typed field on `ProductionAgent` at `factory.py:220`
- `self.memory.inject_long_term_context(...)` at `factory.py:246` — **not defined**
- `self.memory.save_session(...)` at `factory.py:256` — **not defined**
- both guarded by `self.spec.enable_long_term_memory`, whose default is `True` (`factory.py:113`)
- `checkpointer` is unconditionally `None` (`factory.py:74`)

There is no second `MemoryManager` anywhere in `src/app`. This is a **missing implementation**, not a duplicate — it does not belong in the unification workstream, only in the same plan's stub-removal bucket.

---

## Shim vs accident verdict

`git log -1` per file:

| File | Commit | Date | Subject |
|---|---|---|---|
| `src/app/shared/agents/tools/idempotency.py` | `c228398` | 2026-07-02 | fix: Fixed MCP lifecycle and more |
| `src/app/shared/agents/tools/__init__.py` | `c228398` | 2026-07-02 | same commit |
| `src/app/shared/agents/memory/memory_scope.py` | `c228398` | 2026-07-02 | same commit |
| `src/app/shared/agents/memory/__init__.py` | `c228398` | 2026-07-02 | same commit |
| `src/app/shared/langchain_layer/agents/tools/idempotency.py` (real) | `2beddca` | 2026-07-16 | feat: add 53 ty rules + fix 147 type errors |
| `src/app/shared/langchain_layer/agents/memory/memory_scope.py` (real) | `e0ee291` | 2026-06-14 | fix: Fixed S3 config and more |
| `src/app/shared/langchain_layer/agents/tools/base.py` | `bdb9664` | 2026-06-22 | outbox relay call-site migration |
| `src/app/shared/langchain_layer/agents/tools/registry.py` | `2beddca` | 2026-07-16 | ty rules |
| `src/app/shared/rag/graphiti/registry.py` | `c8e6075` | 2026-06-21 | refactor: langgraph saul nodes / crawler |
| `src/app/shared/langchain_layer/agents/factory.py` | `2beddca` | 2026-07-16 | ty rules |

**All four files in `src/app/shared/agents/**` were created in one commit, `c228398`, two-to-three weeks AFTER their real counterparts already existed** (`memory_scope` 2026-06-14, `base.py` 2026-06-22). They were not scaffolding that the real code later replaced — they were written *on top of* working code.

- `src/app/shared/agents/tools/idempotency.py` → **ACCIDENT.** Import-satisfying placeholder. Zero docs, zero tests (`grep IdempotencyGuard tests/` → no hits), an API (`execute`) that no caller anywhere uses, and it shadows a 211-line Redis+Postgres implementation. Created after the real one.
- `src/app/shared/agents/memory/memory_scope.py` → **ACCIDENT.** One line, wrong type (`str` where `MemoryScope` is expected), same commit, no docs, no tests.
- `src/app/shared/langchain_layer/agents/tools/registry.py` → **ACCIDENT-adjacent (parallel evolution).** Not a shim of `base.py`: it is an independently written, tag-less second registry that only knows about web-search and crawl tools. Both `get_all_tools` and `get_web_tools` have identical bodies (`:53-60`) — the signature of code written twice without looking.
- `MemoryManager` at `factory.py:69` → **DELIBERATE STUB, self-documented.** Docstring "Stub for LangChain 1.0 MemoryManager - replace when available" (`:70`) plus two `# ponytail:` markers at `:166` and `:220`. It is honestly labelled and still broken; the label does not make the two missing methods safe.

Only the `MemoryManager` stub is documented. Nothing in `tests/` covers any of these five symbols; `codegraph_explore`'s blast radius flags "no covering tests found" for `ToolRegistry` (both), `get_tool_registry`, `get_tools`, and `make_key`.

---

## Reachability ranking

Two facts govern everything below:

1. **`app.state.saul_graph` is never assigned.** The assignment exists only inside the module docstring at `src/app/shared/rag/graphiti/registry.py:25`. `build_saul_graph` (`src/app/shared/langgraph_layer/agent_saul/graph.py:86`) has zero non-docstring callers; `build_tool_registry` (`src/app/shared/rag/graphiti/registry.py:101`) has zero callers; `src/app/lifecycle/lifespan.py` contains no match for `tool_registry`, `idempotency`, `saul_graph`, or `IdempotencyGuard`. Matches the sibling finding at `docs/relay/todo-overlap.md:31,135`.
2. **`create_production_agent` has no route.** Its only callers are `get_research_agent` (`src/app/shared/langchain_layer/agents/registry.py:97`, returns at `:135`) and `get_code_review_agent` (`:144`, returns at `:182`) — and neither of those has any caller in `src/app`. No router, dependency, or Celery task reaches them.

| Rank | Break | Reached by | Evidence |
|---|---|---|---|
| **Breaks on first request** | *none of the four* | — | see below |
| **Breaks on a rare path** | `factory.py:146` `get_tool_registry().get(t)` → `AttributeError`; then `factory.py:246/256` `MemoryManager.inject_long_term_context` / `save_session` → `AttributeError` | only `get_research_agent` / `get_code_review_agent`, which nothing calls | zero callers outside `src/app/shared/langchain_layer/agents/registry.py`. Reachable **only** by a developer or new feature calling the registry directly. `:146` fires *before* `:246`, so the memory break is masked. |
| **Latent / never reached** | stub `IdempotencyGuard` at `get_obligation_chain.py:67` and `precedent_tools.py:80,188`; stub `PRECEDENT_SCOPE` at `precedent_tools.py:21` | the agent_saul route — mounted, but dead upstream | `src/app/api/v1.py:4,17` mounts `agent_saul_router` under `/api/v1`; `src/app/features/agent_saul/dependencies.py:40-41` `get_saul_graph` does `return request.app.state.saul_graph`, injected at `:90`. Since nothing sets `app.state.saul_graph`, the request dies with `AttributeError` in the dependency, **before** any tool executes. These tool bodies are unreachable until the lifespan wiring in the `graphiti/registry.py` docstring is actually implemented. |

Consequence for the planner: **every break in this workstream is a latent-or-rare break, not a production incident.** The `/api/v1` agent_saul route is already non-functional for a reason that has nothing to do with these duplicates. That reorders the risk: the import unification is *cheap and safe* (no live traffic), and its real value is that it stops the stubs from silently absorbing the lifespan wiring when someone finally writes it.

---

## Overlap with deletion

`docs/relay/todo-overlap.md` (sibling scout) already records the graphiti-registry docstring-only wiring at `:31` and `:135` and the competing `ToolResult` classes at `:116`.

Files in **both** this unification list and a plausible deletion manifest:

| File | Status here | Deletion status |
|---|---|---|
| `src/app/shared/agents/tools/idempotency.py` | stub, all importers rewritten away | **delete after rewrite** — zero remaining importers |
| `src/app/shared/agents/tools/__init__.py` | re-export of the stub | delete |
| `src/app/shared/agents/memory/memory_scope.py` | stub | delete |
| `src/app/shared/agents/memory/__init__.py` | re-export of the stub | delete |
| `src/app/shared/agents/__init__.py` | 0 bytes | delete — the whole `src/app/shared/agents/` tree goes |
| `src/app/shared/langchain_layer/agents/tools/registry.py` | loser `ToolRegistry` | **delete only after** `factory.py:53,146` moves to `base.py` AND the `__init__.py:9-11,23,27` re-exports of `get_all_tools`/`get_web_tools` are resolved. Note `get_web_search_tool`/`get_crawl_url_tool` are its only real content and live in `web_search.py:80` / `crawl.py:114` — they survive independently. |
| `src/app/shared/rag/graphiti/registry.py` | third `ToolRegistry`, name collision only | **do NOT delete** — it is the intended lifespan DTO. Its docstring paths are stale and point at the stub tree. |
| `factory.py:69-74` `MemoryManager` | stub with 2 missing methods | not a deletion candidate; it is a *replace* candidate. Deleting it breaks `factory.py:164,220`. |

**Contradiction risk:** a deletion manifest that removes `src/app/shared/agents/**` before `get_obligation_chain.py:29` and `precedent_tools.py:21-22` are rewritten produces an `ImportError` at module import — a **hard** break at `app.shared.langchain_layer.agents.tools` package import time, which `src/app/shared/rag/graphiti/registry.py:41-46` imports eagerly. Ordering is load-bearing: **rewrite imports first, delete second.**

---

## Fog

- **`PRECEDENT_SCOPE` usage shape inside `precedent_tools.py`.** I confirmed the import (`:21`) and the type mismatch (`str` vs `MemoryScope`) but did not enumerate every attribute access on it. If the file treats it as a string, the survivor swap needs code changes, not just an import change. Establishing this needs a full read of `precedent_tools.py:35-237`.
- **Whether `get_research_agent` / `get_code_review_agent` are reached from outside `src/app`.** I searched `src/app` only. A notebook, script, `src/app/examples/`, or `tests/e2e` entry point could reach them, which would promote the `factory.py:146` break from "rare" to "breaks on first request". `grep` over `tests/` for the symbol names returned nothing, but I did not sweep repo root scripts or `docs/`.
- **`ToolRegistry` survivor ambiguity.** The user decision says "unify in `langchain_layer`, keep tags", which uniquely selects `tools/base.py:58`. But `tools/base.py` exposes a **module-level singleton** `registry` (`:99`) populated by the `register_tool` decorator (`:125`) and `make_structured_tool` (`:171`) — not a `get_tool_registry()` accessor. Whether the plan wants `factory.py` to use that singleton or a fresh instance is a design choice I am not making. I also could not establish **which tools actually register themselves** into that singleton at import time — if none do, `registry.get(t)` raises `KeyError` where `get_tool_registry().get_tool(t)` returned `None`, changing the failure mode. Establishing this needs a sweep of `@register_tool` / `make_structured_tool` call sites.
- **The graphiti `ToolRegistry` name collision.** Whether the plan renames it (e.g. `AgentToolBundle`) is a decision, not a fact. I recorded only that three classes share the name and two of the three are unrelated concepts.
- **`open_deep_search`.** Confirmed **zero** definitions of `ToolRegistry`, `IdempotencyGuard`, or `MemoryManager` there. The only overlap is the name `get_all_tools` (`open_deep_search/utils.py:260`, async) vs (`langchain_layer/agents/tools/registry.py:53`, sync). Recorded hazard; no work scheduled.
