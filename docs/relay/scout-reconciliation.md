# Scout — Reconciliation subsystem (Leg 1, item 155)

Date: 2026-08-17 · Branch: main

## 1. Inventory

### 1a. `langgraph_layer/` — the reconciliation package (618 lines, CONFIRMED)

`src/app/shared/langgraph_layer/reconciliation/`

| File | Lines | Public symbols |
|---|---|---|
| `__init__.py` | 33 | re-exports only (`ReconciliationGraph`, `build_reconciliation_graph`, state models, node factories) |
| `state.py` | 78 | `ReconciliationEntityRecord` (:18), `MergeDecision` (:30), `ReconciliationDecision` (:53), `ReconciliationState` (:61) |
| `nodes.py` | 442 | `make_fetch_existing_node` (:62), `make_reconcile_node` (:135), `make_apply_changes_node` (:205), `make_write_versions_node` (:274); privates `_reconciliation_failure` (:40), `_infrastructure_failure` (:44), `_row_to_record` (:359), `_parse_reconciliation_decision` (:374), `_read_llm_content` (:380), `_strip_markdown_fences` (:385) |
| `graph.py` | 42 | `build_reconciliation_graph` / `ReconciliationGraph` |
| `prompts.py` | 23 | reconciliation LLM prompt text |

42+33+442+23+78 = **618**. Claim CONFIRMED — but note 618 counts the package **only**; the subsystem also spans `src/tasks/memory_decay_reconciliation_tasks.py` (209) and `src/database/schemas/memory_schema.py` (302), for **1129 lines** total.

### 1b. `features/` — **CLAIM CORRECTION**

There is **no reconciliation code under `src/app/features/` at all.** `find src/app/features -ipath "*reconcil*"` returns nothing. The only `features/` file matching `reconcil` is `src/app/features/billing/models/audit.py`, and that is billing payment reconciliation — unrelated domain (see §2). Sub-todo 1's "and `features/`" has no referent; the reconciliation area is `shared/langgraph_layer/` + `src/tasks/` + `src/database/schemas/`.

### 1c. Satellites

- `src/tasks/memory_decay_reconciliation_tasks.py` (209): `DecayStats` (:24), `ReconciliationSummary` (:32), `_compute_decay` (:51), `_run_decay_async` (:64), `_run_reconciliation_async` (:145), `run_memory_decay` (:180), `run_reconciliation_for_user` (:186), `run_reconciliation_for_active_users` (:198)
- `src/database/schemas/memory_schema.py` (302): orphan `Base` (:51), `Entity` (:55), plus `relationships`/`events`/`memory_versions` models

## 2. Capability ledger — CORE DELIVERABLE

**Read this first:** every row is *potential* capability, never observed behaviour. Proof: no `@celery_app.task` decorator anywhere in `memory_decay_reconciliation_tasks.py` (only bare `def` at :51, :64, :145, :180, :186, :198), the module is absent from `celery.py:191-196`, and the four tables it queries were never created (§4). Nothing in this subsystem has ever executed in this repo. The "what is lost" column describes design work discarded, not a regression a user would notice.

| Capability | Where implemented | Anything else provides it? | If simply gone |
|---|---|---|---|
| **Audit trail / entity versioning** — append-only `memory_versions` rows per entity change | `nodes.py:274` `make_write_versions_node`; table model `memory_schema.py` | Partially. `src/app/features/billing/models/audit.py:48` `AuditLog` + `repositories/audit_repository.py:30,45,73` is a *live, migrated* audit trail — but scoped to billing only, not memory entities. Cognee: **unverified** (§7 Fog) | Nothing observable. Loses the only design for memory-entity version history; billing audit unaffected |
| **Edge-preserving merge** — merge duplicate entities while re-pointing `relationships` rows | `nodes.py:205` `make_apply_changes_node`; decision model `state.py:30` `MergeDecision` | **Nothing** in-repo. Nearest neighbour is `src/app/shared/rag/graphiti/subgraph.py:133` `expand_from_seeds` / `:178` `get_obligation_chain` — those *traverse* a graph, they never merge nodes | Nothing observable. Loses the merge-vs-keep-both decision schema, which is the non-trivial part |
| **Near-duplicate detection** — raw SQL self-join over `entities e1 JOIN entities e2` (`nodes.py:94-95`) with LLM adjudication (`nodes.py:135` `make_reconcile_node`, prompt at `prompts.py:23`) | `nodes.py:62` `make_fetch_existing_node` | **Nothing** with this shape. `memory_scope.py:79 allows_entity_type` / `:73 allows_source` filter memory, they do not dedupe. Cognee dedupe: **unverified** | Nothing observable. Loses the LLM-adjudicated dedupe prompt + JSON parse chain (`nodes.py:374,380,385`) |
| **Memory decay** — exponential score from age/access_count/confidence | `src/tasks/memory_decay_reconciliation_tasks.py:51` `_compute_decay`, driver `:64` `_run_decay_async`, entry `:180` `run_memory_decay` | **Nothing.** No other decay implementation in the repo | **This is the one that matters for backlog item 170** (memory-decay cron). Deleting `_compute_decay` deletes the only decay formula in the repo; item 170 would restart from a blank page unless the ~13-line function is preserved somewhere |
| **Per-user / fleet-wide batch orchestration** | `:186` `run_reconciliation_for_user`, `:198` `run_reconciliation_for_active_users` | Pattern exists in billing: `src/tasks/billing_tasks.py:71` `_renewal_job`, `:253` `_reconciliation_job` — copyable shape, different domain | Nothing observable. Shape is re-derivable from billing_tasks |
| **LangGraph node-factory + Result-pattern conformance** | `graph.py:42` builder; `nodes.py:40,44` `_reconciliation_failure`/`_infrastructure_failure` | Yes — this is the house pattern, spec'd at `openspec/changes/archive/2026-06-14-result-adoption-phases-2-5/specs/langgraph-node-result-pattern/spec.md` | Nothing lost; pattern lives in the spec |

**Naming collision to avoid mis-deleting:** `beat_schedule` entry `"billing-reconciliation-daily"` → task `billing.reconciliation` (`celery.py:272-275`) is **billing payment reconciliation** (`src/tasks/billing_tasks.py:253`), a live, scheduled, unrelated subsystem. It shares only the word.

## 3. Proof of deadness

Verbatim `graphify affected` — both claims **CONFIRMED**:

```
Affected nodes for reconciliation/graph.py
Depth: 2
- reconciliation/__init__.py [re_exports] src/app/shared/langgraph_layer/reconciliation/__init__.py:L1

Affected nodes for memory_decay_reconciliation_tasks.py
Depth: 2
- tasks/__init__.py [re_exports] src/tasks/__init__.py:L6
```

- **Tests:** no test imports reconciliation. `rg "reconcil|memory_schema" tests/` hits only prose in `tests/performance/todo.md` (the backlog itself).
- **Routers / lifespan:** no reach. Nothing under `src/app/features/` or `src/app/lifecycle/` references it.
- **Celery decorators — CONFIRMED:** zero `@celery_app.task` in the module; include list `celery.py:191-196` is 4 entries, none of them `tasks.memory_decay_reconciliation_tasks`.
- **beat_schedule — CONFIRMED:** `celery.py:259-276` is exactly 4 billing entries.
- **One live edge, and it is load-bearing for the delete:** `src/tasks/__init__.py:6-9` *does* `from .memory_decay_reconciliation_tasks import (...)` and re-exports all three names in `__all__` (`:18-20`). So `import tasks` imports the module (no task registration, but a real import). Deleting the file without editing `tasks/__init__.py` breaks every celery worker at import time.

## 4. Orphan-schema teardown

`src/database/schemas/memory_schema.py` declares its own `Base` at **:51** — `declarative_base()` local to the module. All four claims **CONFIRMED**:

- `src/database/__init__.py:3-6` imports `Base` from `.base` and only `ChatMessage, ChatSession, DocumentVector` from `.schemas`. `memory_schema` is never imported.
- `src/alembic/env.py:11` `from database import Base`; `:23-24` registers only `app.features.billing.models` and `app.shared.outbox.model`; `:27` `target_metadata = Base.metadata`. The orphan `Base` at `memory_schema.py:51` is never in that metadata.
- `rg "create_table" src/alembic/versions/*.py | rg -i "entit|relationship|event|memory_version"` → **exit 1, zero hits.** No CREATE TABLE for `entities`, `relationships`, `events`, `memory_versions` anywhere.
- `memory_schema.py:15` even carries the never-run instruction as a docstring: `Then: uv run alembic revision --autogenerate -m "add_memory_schema"`.

**The distinction you asked for — does anything import the module for its model classes?** **No. Nothing imports `memory_schema` at all**, not even the reconciliation code that logically owns it: `nodes.py:14-25` imports `sqlalchemy.text` and hand-writes raw SQL (`FROM entities` at `:77`, the self-join at `:94-95`). The ORM models are **decorative** — declared, never used, never migrated. That places the file fully in the delete column, not the gut column: there is no model class to preserve for another importer.

Teardown surface: delete the file; no `database/__init__.py` edit needed (it never appeared there); no migration needed (no table exists to drop); no downgrade path.

## 5. The remove-vs-break-up tension — presented, not resolved

Verbatim, `tests/performance/todo.md`:

- **:285** — `155. complete the ingestion pipeline to working condition and see where reconciliation comes init. i want to remove reconciliation and replace it with agent memory made with cognee entirely.`
- **:287** — `1. toons reusable , point 138,  break the code for reconcilliation inside langgraph_layer/ and features/, ...`
- **:265** — `170. write cron job for memory decay and then send to celery for off loading for cognee`

Concretely, in files:

| Reading | Files touched | Consequence |
|---|---|---|
| **Remove** (155) | Delete `reconciliation/` (618 L), `memory_decay_reconciliation_tasks.py` (209 L), `memory_schema.py` (302 L); edit `src/tasks/__init__.py:6-9,18-20` | 1129 lines gone. `_compute_decay` (`:51`) — the only decay formula in the repo — goes with it, and **item 170 needs exactly that** |
| **Break up** (sub-todo 1) | Split/redistribute the same package; `and features/` has **no referent** (§1b) | Reads as the *already-completed* work of `docs/superpowers/plans/2026-04-13-...md`, which split the monolith into `state.py`/`prompts.py`/`graph.py`/`nodes.py`. Re-doing it re-splits a package that is being deleted |

Third data point the decision needs: `openspec/changes/cognee-saul-memory-migration/proposal.md:20-21` states **"Cognee v1.1 has no built-in curation/decay/dedup"** and marks `saul-cognee-maintenance-worker` and `saul-cognee-reconciliation` **deferred**. So "replace with cognee entirely" (155) is not a like-for-like swap by that change's own analysis — decay, dedupe, and merge have no cognee-side equivalent named anywhere.

## 6. Git archaeology

```
git log --oneline -- <reconciliation paths> | head -20
2beddca feat: add 53 ty type-checker rules + fix 147 type errors
5c67c7d fix: migrate all BLE001 blind-except violations to typed exceptions
2178c36 refactor(result): eliminate dual-method pattern across all repos and callers
22650cd chore: Result migration complete
e0ee291 fix: Fixed S3 config and more
7cfb667 refactor: Updated auth, ingestion, user services and langgraph layer with result pattern
b337613 refactor: Updated langgraph_layer and ingestion_kb
1d5be4f refactor: Refctored langgraph_layer
a5d0024 refactor: Simplified models and improved system prompt parts
acac4a7 refactor: Docling and Graphiti refactor + Graphify init
20fed4b refactor: Corrected imports and corrected code
f579b66 refactor: Refactored langGraph code
7d817f2 feat: Added better support and security for WS
0397b90 feat: Added knowledge tools and memory component
```

14 commits, **one** feature commit (`0397b90 Added knowledge tools and memory component`) and 13 sweeps that dragged it along (Result pattern, BLE001, ty rules). Verdict: **written once, never wired, then maintained-by-sweep.** Not a half-migration from something older.

Against the prior plan `docs/superpowers/plans/2026-04-13-reconciliation-langgraph-package-split.md`: the code **does match its intent** — the plan renames misspelled `reconsiliation/` → `reconciliation/` and splits monolithic `pipeline_node.py` into `state.py` + `prompt.py` + `graph.py`, and all of that is present. Two naming drifts from plan to code: plan says `prompt.py`, repo has `prompts.py`; plan keeps `pipeline_node.py`, repo has `nodes.py`. The plan's checkboxes are still unticked (`- [ ]`) despite the work appearing done.

## 7. Fog — what I could NOT prove dead

- **Cognee's actual capability surface.** I read only the openspec proposal's *claim* that Cognee v1.1 lacks decay/dedup/curation (`openspec/changes/cognee-saul-memory-migration/proposal.md:20-21`). I did not read `.venv/lib/python3.12/site-packages/cognee/` to verify. Establishing it: grep cognee's own API for decay/dedupe/merge entry points.
- **`docs/plan-ingestion.md` and `docs/fuzzy-crafting-cookie.md` are untracked working files** mentioning reconciliation; I did not read them. They may hold a newer intent than todo.md.
- **`src/app/shared/langchain_layer/agents/memory/memory_scope.py`** (`MemoryScope` :47, `_build_scope` :133, `scope_from_router_decision` :202) is adjacent memory infrastructure. `graphify affected` was not run on it. **It is NOT a deletion candidate** — I have no evidence it is dead, and it appears to be live scope-filtering.
- **`src/app/shared/rag/graphiti/subgraph.py`** likewise: adjacent graph code, not proven dead, not a deletion candidate.
- **Whether the `entities`/`relationships`/`events` tables exist in a live database** created out-of-band (by hand, or by cognee's own alembic at `.venv/.../cognee/alembic/`). Absent from *this repo's* migrations is proven; absent from a running Postgres is not. Establishing it: `\dt` against the dev database.

**Deletion candidates I can prove dead:** `src/app/shared/langgraph_layer/reconciliation/` (all 5 files), `src/tasks/memory_decay_reconciliation_tasks.py`, `src/database/schemas/memory_schema.py` — with the mandatory paired edit to `src/tasks/__init__.py:6-9,18-20`.
