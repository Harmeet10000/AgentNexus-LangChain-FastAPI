# Plan — openspec change 0 (`cleanup-foundation`)

Planner leg, 2026-08-17. Read-only on `src/`. Every claim below was re-verified against the tree,
not just inherited from the scout reports; where a scout report is **wrong or incomplete** it is
labelled **SCOUT CORRECTION**.

> **READ THE ADDENDUM FIRST (`# ADDENDUM — live-database supersession`, at the end of this file).**
> A live-database probe landed after the body was written. It supersedes Steps 7-9, R1, R2, F1, F2, F6, F7,
> adds Steps 8b and 17, and adds one new confirmed runtime break. It also invalidates every baseline figure in
> the body: the working tree was refactored underneath this plan. Where addendum and body conflict, the
> **addendum is binding**.

## Shape

Change 0 is two unrelated workstreams that happen to share a change directory: a **migration-chain
repair** (merge two alembic heads, give `clauses` a `CREATE` so the chain runs on a clean DB, teach
`env.py` about the live models) and a **subtraction pass** (delete nine proven-dead trees plus the
four `__init__`/config edits that are coupled to them). Bolted on are four small behaviour fixes that
are live `AttributeError`s or blind spots on already-mounted routers. The plan sequences the migration
work first — because `alembic merge` and the `--sql` dry run are the only proofs in this change that
are both cheap and DB-free, and because one deletion target (`memory_schema.py`) is the sole
declarative description of two tables the migration repair needs. Everything else is cut so each step
is one commit that leaves the repo bootable; **no step in this plan temporarily breaks boot**, which
is achieved by pairing every deletion with its coupled `__init__` edit inside the same step.

## Ordering constraints

**OC0 — Baselines to disk before the first edit.** `--cov-fail-under=80` is in `addopts` and coverage
is 18.38%, so exit code is meaningless (`baseline-tests.md`). Every later proof compares a *summary
line* or a *count* against a file in `/tmp`. If Step 0 is skipped, nothing in this plan is provable.

**OC1 — RESOLUTION OF THE `shared/agents/**` CYCLE: it is NOT a change-0 deletion.**
D6.1 says change 3 must rewrite importers off `shared/agents/**` before the tree is deleted.
The cycle dissolves once you read the manifest carefully: `scout-deletion-manifest.md` §2 **never
lists `shared/agents/**`**, and §4 says verbatim *"Goes to Fog, not the manifest."* So the manifest
and D6.1 already agree — only the orchestrator's scope summary implied otherwise. I re-verified the
import chain and it is worse than "an ordering nicety":

```
app.shared.rag.graphiti.registry            (:40-45, module level, NOT under TYPE_CHECKING)
  → app.shared.langchain_layer.agents.tools (package __init__)
    → …/tools/precedent_tools.py:21
      → app.shared.agents.memory.memory_scope   ← the 30-byte SHADOW
```

`registry.py` is reached at boot (`ToolRegistry` is consumed by `agent_saul/graph.py:16,91` and
`agents/factory.py:182,205`; `agent_saul_router` is mounted at `api/v1.py:17`). Deleting
`shared/agents/**` in change 0 is therefore an `ImportError` at import time, and it cannot be fixed
by a coupled `__init__` edit — the fix is *retargeting `precedent_tools.py` to the real 7189-byte
`langchain_layer/agents/memory/memory_scope.py`*, which is change 3's registry/memory-scope
unification work (D6.1, dispositions row Up#10). **Resolution: `shared/agents/**` moves to change 3.
Change 0 deletes only trees whose importer set is provably empty *after* its own coupled edits.**
Verified today: `uv run python -c "import app.main, app.api.v1, app.features, tasks,
app.shared.rag.graphiti.registry, app.shared.langchain_layer.agents.tools"` → all six `OK`.

**OC2 — `memory_schema.py` cannot be deleted before its two live models are harvested. SCOUT
CORRECTION (the two scouts contradict each other).** `scout-reconciliation.md` §4 says delete the
whole 302-line file, *"no migration needed (no table exists to drop)"*, and inventories it as only
`Base`/`Entity`/`relationships`/`events`/`memory_versions`. It **omits** that the same file declares
`ParentDocument` (`memory_schema.py:154`) and `Clause` (`:190`) — and
`scout-persistence-docling.md` §1 shows `parent_documents` has a real `create_table` at
`9f4a1b7c6d2e:29` and `clauses` is read by the clause/graphiti tools and written by
`ingestion_kb/nodes.py`. `Clause` (`:190-249`) is the **only declarative description of `clauses`
anywhere in the repo**, and Step 8 needs exactly that to write the missing `CREATE TABLE`.
Therefore: harvest/move first (Step 9), delete the remnant second (Step 10).

**OC3 — Merge before any autogenerate, and before the `--sql` proof works at all.** Confirmed today:
`uv run alembic upgrade head --sql` exits **255** with *"Multiple head revisions are present"*. Until
Step 7 lands, the cheapest proof in this change does not run. `alembic heads`/`branches`/`merge` all
work without a database; `upgrade --sql` runs `env.py` offline and also works without one.

**OC4 — Deletion and its coupled edit are ONE commit, never two.** Four couplings, and **two of them
are missing from the manifest**:
| deletion | coupled edit | in manifest? |
|---|---|---|
| `features/knowledge_base/`, `features/web_scraping/` | `src/app/features/__init__.py:3,8,9` | yes |
| `reconciliation/`, `memory_decay_reconciliation_tasks.py` | `src/tasks/__init__.py:6-9,18-20` | yes |
| `shared/vectorstore/` (package) | `src/app/shared/__init__.py:3,10` — `from . import crawler, rag, vectorstore` + `"vectorstore"` in `__all__` | **NO — SCOUT GAP** |
| `reconciliation/nodes.py` | `pyproject.toml:538` per-file-ignores key | **NO — SCOUT GAP** |

**OC5 — Nothing in change 0 touches `features/search/` behaviour.** D5.1 puts search in scope for
changes 1/2 only. The single exception is D5.2, which is explicit that `UserIdDep` is broken
**repo-wide** — so Step 11 edits `search/dependencies.py:44` too. That is a locked decision, not
scope creep.

**OC6 — The health probe (Step 14) is the acceptance signal for the lifespan's silent degradation, so
it lands after the `app.state` naming fix (Step 11)** — otherwise you are adding probes to a state
surface whose read/write contract is still wrong.

## Steps

Notation: **Dep** = inbound dependency (which step must be committed first, and why). **Proof** = the
exact command and the exact expected output. **Coverage** = whether a test exercises this step today.

Two reusable proofs are referenced by name:

```bash
# PROOF-BOOT  (the six-module import probe — all six print OK today)
uv run python -c "
import importlib
for m in ['app.main','app.api.v1','app.features','tasks',
          'app.shared.rag.graphiti.registry','app.shared.langchain_layer.agents.tools']:
    importlib.import_module(m); print('OK  ', m)"

# PROOF-STATE  (every *.state.<attr> READ has a lifespan WRITE).
# MUST be AST-based, not regex: lifespan.py:171 writes db_engine via TUPLE UNPACKING
# (`app.state.db_engine, app.state.db_session_local = pg_task.result()`) and lines 241-305
# are COMMENTED-OUT writes. A regex reports db_engine as a break (false positive) and
# ingestion_graph/langgraph_checkpointer as satisfied (false negatives). Verified both ways.
uv run python - <<'PY'
import ast, pathlib
w = set()
for n in ast.walk(ast.parse(pathlib.Path("src/app/lifecycle/lifespan.py").read_text())):
    if isinstance(n, (ast.Assign, ast.AnnAssign)):
        stack = list(n.targets) if isinstance(n, ast.Assign) else [n.target]
        while stack:
            t = stack.pop()
            if isinstance(t, ast.Tuple): stack.extend(t.elts)
            elif isinstance(t, ast.Attribute) and isinstance(t.value, ast.Attribute) \
                 and t.value.attr == "state": w.add(t.attr)
r = {}
for p in sorted(pathlib.Path("src/app").rglob("*.py")):
    if "lifecycle/" in str(p): continue
    try: t = ast.parse(p.read_text())
    except SyntaxError: continue
    for n in ast.walk(t):
        if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Attribute) \
           and n.value.attr == "state": r.setdefault(n.attr, []).append(f"{p}:{n.lineno}")
for k in sorted(set(r) - w): print("UNWRITTEN:", k, "->", ", ".join(r[k][:5]))
PY
```

`PROOF-STATE` today prints exactly seven `UNWRITTEN:` lines:

| attr | sites | owner |
|---|---|---|
| `storage` | `profile/router.py:29` | **Step 11** (B1) |
| `mongodb` | `profile/router.py:30` | **Step 11** (B2) |
| `user_id` | `agent_saul/dependencies.py:63`, `crawler/router.py:28`, `documents/dependencies.py:62`, `ingestion/dependencies.py:12`, `search/dependencies.py:45` | **Step 12** (D5.2) |
| `ingestion_graph` | `ingestion/dependencies.py:8` | change 1 (commented wiring `lifespan.py:241-248`) |
| `saul_graph` | `agent_saul/dependencies.py:41` | change 3 (B3) |
| `langgraph_checkpointer` | `agent_saul/dependencies.py:45` | change 3 (B4, commented `lifespan.py:294-305`) |
| `ws_rate_limit_id` | `auth/websocket_security.py:310,312` | **FALSE POSITIVE** — those two *are* the writes (`websocket.state`, not `app.state`); read is guarded by `getattr(..., "ws:anonymous")` at `:84`. Do not touch. |

**NEW FINDING, in neither scout report:** `user_id` has **five** read sites, not the two D5.2 names.
`crawler/router.py:28` is the fifth and it is *correctly guarded* (`if hasattr(request.state,
"user_id")`) and its router is **mounted nowhere** (`rg crawler src/app/api/*.py src/app/main.py` →
exit 1), so it is not a break — but after Step 11 nothing in the repo ever sets
`request.state.user_id`, making that branch provably dead. Simplify it in Step 12 or record it.

After Step 11 the `storage`/`mongodb` rows must be gone; after Step 12 the `user_id` row must be gone.
Exactly three `UNWRITTEN:` lines remain at the end of change 0, all owned by later changes.

---

### Step 0 — Freeze the baseline to `/tmp`
**Dep:** none. **Coverage:** n/a.

```bash
uv run pytest -q --no-header -p no:cacheprovider 2>&1 | tee /tmp/base-pytest.txt \
  | grep -E '^(FAILED|ERROR)' | sort > /tmp/base-failures.txt
uv run ruff check src/       2>&1 | tail -3 > /tmp/base-ruff.txt
uv run ty check src/         2>&1 | tail -3 > /tmp/base-ty.txt
ast-grep scan src/           2>&1 | tail -3 > /tmp/base-astgrep.txt
uv run alembic heads         2>&1 | grep '(head)' > /tmp/base-heads.txt
/home/harmeet/.bun/bin/openspec validate --all 2>&1 | tail -1 > /tmp/base-openspec.txt
```

**Proof:** `/tmp/base-failures.txt` has **35 lines**; `grep -c . /tmp/base-failures.txt` → `35`.
`/tmp/base-pytest.txt` last summary line reads `22 failed, 55 passed, 11 warnings, 13 errors`.
`/tmp/base-ruff.txt` contains `Found 125 errors.`; `/tmp/base-ty.txt` contains
`Found 46 diagnostics`; `/tmp/base-astgrep.txt` contains `Error: 4 error(s) found in code.`;
`/tmp/base-heads.txt` is exactly two lines (`0004 (head)`, `a71f0d7d9c12 (head)`);
`/tmp/base-openspec.txt` reads `Totals: 16 passed, 6 failed (22 items)` — all four re-confirmed today.

---

### Step 1 — Author `openspec/changes/cleanup-foundation/`
**Dep:** Step 0 (the openspec totals baseline). **Coverage:** n/a.

Bare-slug ID `cleanup-foundation` (D12). `.openspec.yaml` = `schema: spec-gated`, `created: 2026-08-17`,
**no `skip_specs`** — change 0 carries real deltas (see Openspec mapping). Class **L**; `design.md`
mandatory. `review.md` written by a fresh subagent, not the author; `tasks.md` only after its
`VERDICT:` is not `CHANGES-REQUESTED`. Scenario headers take **exactly four hashtags** — three fail
silently.

**Proof:** `/home/harmeet/.bun/bin/openspec validate cleanup-foundation --type change` → passes; and
`openspec validate --all | tail -1` → `Totals: 17 passed, 6 failed (23 items)`. The failed count
**must stay at 6** — it is never "validate --all passes" (D12).

---

### Step 2 — Delete `src/app/shared/rag/document_processing/todo_temp.py` (783 lines)
**Dep:** Step 0. Independent of everything else; first because it is the only step whose proof is a
*drop* in the lint baseline, which validates that the baseline files are trustworthy.
**Coverage: ZERO.** Non-test evidence: the file **does not parse** (`ast.parse` → `IndentationError`
at `:406`), so a live importer is physically impossible (D11), and `rg todo_temp` outside the file is
empty — re-confirmed today across the whole repo excluding `graphify-out/` and `docs/`.

**Proof:** `uv run ruff check src/ 2>&1 | tail -1` → `Found 123 errors.` (was 125 — the two
`invalid-syntax` errors at `:406:1` and `:773:1` were both this file);
`uv run ruff check src/ 2>&1 | grep -c invalid-syntax` → `0`; and `src/` fully parses for the first
time. Also `uv run pytest -q --no-header -p no:cacheprovider 2>&1 | grep -c couldnt-parse` → `0`.
**From this step onward the ruff GREEN threshold is 123, not 125.**

---

### Step 3 — Delete `src/app/utils/toon_parser.py` (36 lines)
**Dep:** Step 2 (commit hygiene only — technically independent). **Coverage: ZERO.**
Non-test evidence: `rg -n "toon_parser" .` excluding the file itself and `graphify-out/`/`docs/` →
**exit 1, zero hits** (re-confirmed today). `graphify affected "toon_parser"` empty. Note it also
removes an import-time side effect (`:17` builds a `ChatPromptTemplate` at module import) and an
inverted `parse()` (`:13` returns `toons.dumps(text)`) — do **not** "fix" it, D-nothing sanctions it
and `serialize_to_toon` already owns this job with 16 call sites.

**Proof:** `test ! -e src/app/utils/toon_parser.py`; `rg -n "toon_parser" src/ tests/` → exit 1;
`PROOF-BOOT` → 6× OK; `uv run ruff check src/ 2>&1 | tail -1` → still `Found 123 errors.`

---

### Step 4 — Delete `src/app/shared/vectorstore/` **and edit `src/app/shared/__init__.py:3,10`**
**Dep:** Step 0. **Coverage: ZERO.** **SCOUT GAP — this coupling is not in the manifest.**
The manifest lists only the three 0-byte modules (`vector_store.py`, `insert_vectors.py`,
`similarity_search.py`) and not the package, so following it literally would leave a package whose
`__init__.py` is a 4-line comment with `__all__ = []`. Delete the **whole package** and remove
`vectorstore` from `shared/__init__.py:3` (`from . import crawler, rag, vectorstore`) and from
`__all__` at `:10`. Both edits are in this same commit — deleting the directory alone is an
`ImportError` on `import app.shared`, i.e. on every module in the app.

**Proof:** `PROOF-BOOT` → 6× OK (this is the proof that matters — `app.shared` is on every import
path); `rg -n "vectorstore" src/` → exit 1; `test ! -d src/app/shared/vectorstore`;
pytest summary line still `22 failed, 55 passed, 11 warnings, 13 errors`.

---

### Step 5 — Delete `src/app/shared/langchain_layer/agents/orchestration_type/` (5 files, all 0 bytes)
**Dep:** Step 0. **Coverage: ZERO.** Non-test evidence: all five files are 0 bytes including
`__init__.py`, and `rg -n "orchestration_type" src/ tests/ openspec/ pyproject.toml` → exit 1
(re-confirmed today). No coupled edit: nothing imports the package.

**Proof:** `test ! -d src/app/shared/langchain_layer/agents/orchestration_type`;
`rg -n "orchestration_type" src/ tests/ pyproject.toml` → exit 1; `PROOF-BOOT` → 6× OK.

---

### Step 6 — Delete `features/knowledge_base/` + `features/web_scraping/` **and edit `features/__init__.py:3,8,9`**
**Dep:** Step 0. **Coverage: ZERO.** 15 modules, all 0 bytes; both `__init__.py` files carry
`__all__ = []` and import none of their siblings. Coupled edit: `features/__init__.py:3` is
`from . import documents, health, knowledge_base, web_scraping` → becomes
`from . import documents, health`, and `:8,9` drop out of `__all__`. Same commit.
Manifest Fog #4 (whether a migration/test/spec names them) is now **closed**: repo-wide search over
`src/ tests/ openspec/ pyproject.toml Makefile alembic.ini` finds them only in
`features/__init__.py` and in one *archived* change's prose
(`archive/2026-06-21-…/design.md:25`, a historical scan list — archived changes are immutable
records and must not be edited).

**Proof:** `uv run python -c "import app.features; print(app.features.__all__)"` →
`['documents', 'health']`; `PROOF-BOOT` → 6× OK; `rg -n "knowledge_base|web_scraping" src/` → exit 1;
`/home/harmeet/.bun/bin/openspec validate --all | tail -1` → still `17 passed, 6 failed (23 items)`.

---

### Step 7 — Merge the two alembic heads
**Dep:** Step 0. Independent of every deletion — but it must precede Steps 8 and 9 because
`alembic upgrade head --sql` (the only DB-free migration proof) refuses to run with two heads, and
`alembic revision --autogenerate` refuses too.
**Coverage: ZERO** — no test imports alembic. Non-test evidence: the `--sql` dry run below.

Both branches declare `down_revision = "2bc7726317f6"` (`8a7d9b1c2e3f:19`, `a71f0d7d9c12:17`) and
touch **disjoint tables**, so the merge body is empty. Both independently
`CREATE EXTENSION IF NOT EXISTS vector / pg_textsearch` — idempotent, no reconciliation needed.

```bash
uv run alembic merge -m "merge_search_and_unified_document_heads" 0004 a71f0d7d9c12
```

**Proof:** `uv run alembic heads 2>/dev/null | grep -c '(head)'` → `1` (today: `2`, and
`/tmp/base-heads.txt` proves it). `uv run alembic branches` still shows `2bc7726317f6 (branchpoint)`
— that is expected and correct; a merge revision does not remove the historical branch point.
`uv run alembic history | head -3` shows the new revision with `Revises: 0004, a71f0d7d9c12`.

---

### Step 8 — Give `clauses` a `CREATE TABLE` so `9f4a1b7c6d2e` runs on a clean DB
**Dep:** Step 7 (`--sql` proof needs one head). **Must precede Step 10** (it harvests from
`memory_schema.py`, which Step 10 deletes). **Coverage: ZERO.**

The defect, re-verified line by line: `9f4a1b7c6d2e` opens `batch_alter_table("clauses")` at `:63`,
`op.execute("UPDATE clauses SET chunk_id = id …")` at `:101-102`, `alter_column` at `:103-105`,
`create_foreign_key` at `:107`, four `create_index` at `:115-129`, and a raw
`CREATE INDEX clauses_bm25_idx … USING bm25(search_text)` at `:131`. **No revision anywhere creates
`clauses`** — `rg "create_table" src/alembic/versions/*.py | rg -i clause` → exit 1. So
`9f4a1b7c6d2e` aborts on a fresh database, which blocks `0001→0004`: **outbox and billing cannot be
migrated from scratch today.** A merge revision does not touch this; it is a separate defect.

**Recommended fix — a raw `CREATE TABLE IF NOT EXISTS` at the top of `9f4a1b7c6d2e.upgrade()`**, not
a new revision and not an inspector-based guard. Three reasons, each load-bearing:

1. **`op.execute("CREATE TABLE IF NOT EXISTS clauses (…)")` is offline-safe.** An
   `inspect(op.get_bind()).has_table("clauses")` guard raises in offline mode (no bind), which would
   destroy the `--sql` proof. Raw DDL emits verbatim in both modes.
2. **Editing this revision body is safe for the deployed DB.** `alembic_version` holds only the head
   ID; a database already at `0004` never re-executes an ancestor. `IF NOT EXISTS` makes it a no-op
   even if it were re-run.
3. **It must create the PRE-ALTER shape, not the final shape.** `:64-99` `add_column`s eight columns;
   creating them up front makes `add_column` fail with *"column already exists"*. Harvest from
   `memory_schema.py:190-249` (`Clause`) and **subtract** the eight added at `:64-99`:

| create now (pre-ALTER) | added later by `9f4a1b7c6d2e:64-99` — do NOT create |
|---|---|
| `id` (uuid PK), `contract_id`, `doc_id`, `user_id`, `clause_id`, `text`, `embedding vector(768)` NULL, `clause_type`, `risk_score`, `decay_score`, `access_count`, `created_at`, `last_accessed_at` | `chunk_id`, `parent_doc_id`, `chunk_index`, `preamble`, `chunk_text`, `metadata_`, `custom_metadata`, `search_text` (generated) |

Indexes: create only `idx_clauses_doc_id`, `idx_clauses_type`, `idx_clauses_risk_score`,
`idx_clauses_user_id` (the four from `Clause.__table_args__` that `9f4a1b7c6d2e` does **not** create).
Do **not** create `idx_clauses_parent_doc_id`, `idx_clauses_parent_chunk_index`,
`idx_clauses_metadata_gin`, `idx_clauses_chunk_id` — `:115-129` creates all four.
`embedding` must be `vector(768)` nullable so `:105`'s `alter_column(type_=Vector(768))` is a no-op.
Extensions are already created at `:24-26`, above the insertion point — keep the new DDL below them.
`downgrade()` gains a matching `op.execute("DROP TABLE IF EXISTS clauses")` **after** the existing
`drop_column` batch (`:150-158`), so downgrade stays symmetric.

**Proof (DB-free):**
```bash
uv run alembic upgrade head --sql > /tmp/head.sql 2>/tmp/head.err; echo "exit=$?"   # → exit=0 (today: 255)
grep -n 'CREATE TABLE IF NOT EXISTS clauses' /tmp/head.sql          # line N
grep -n 'ALTER TABLE clauses ADD COLUMN chunk_id' /tmp/head.sql     # line M, and M > N
grep -c 'CREATE TABLE' /tmp/head.sql                                # ≥ 20 tables emitted
```
The ordering assertion (`N < M`) is the real proof; a nonzero exit or a missing `CREATE TABLE
… clauses` is a fail. **Proof (real DB) is NOT available locally — see Fog F1.**

---

### Step 9 — Move `ParentDocument` + `Clause` onto `database.Base`, then register every model module in `env.py`
**Dep:** Step 8 (the migration's DDL is the reference shape) and Step 7 (single head).
**Must precede Step 10.** **Coverage: ZERO.**

Two halves, one commit, because half one without half two makes autogenerate *worse*, not better.

*Half one — the move.* Create `src/database/schemas/contract_kb.py` holding `ParentDocument`
(from `memory_schema.py:154-189`) and `Clause` (`:190-249`), re-parented from the module-local
`declarative_base()` at `memory_schema.py:51` onto the shared `Base`. Verified today that
`app.shared.Base is database.Base` → `True` (both are `app.shared.base.Base`), so `from app.shared
import Base` and `from database import Base` are interchangeable; use whichever the neighbouring
schema modules use. Export from `src/database/schemas/__init__.py` and `src/database/__init__.py`
alongside `ChatMessage, ChatSession, DocumentVector`.

*Half two — the registration.* `env.py:23-24` registers only `app.features.billing.models` and
`app.shared.outbox.model`. Add `app.features.documents.model`, `app.features.search.model`, and the
new `database.schemas.contract_kb`, each with the same `# noqa: F401, E402` comment the existing two
carry.

**Why the move is not optional scope creep:** `env.py:27` sets `target_metadata = Base.metadata`.
Registering documents/search *without* also putting `clauses`/`parent_documents` on `Base` means the
next `--autogenerate` emits `DROP TABLE clauses` and `DROP TABLE parent_documents` — a *new*
data-loss hazard created by this step. The alternative is an `include_object` allow-list in `env.py`,
which requires hand-maintaining a table-name list forever; the move is smaller and self-maintaining.

**Proof:**
```bash
uv run python -c "
import app.features.billing.models, app.shared.outbox.model
import app.features.documents.model, app.features.search.model, database.schemas.contract_kb
from database import Base; print(sorted(Base.metadata.tables))"
```
→ `['chat_messages', 'chat_sessions', 'chunks', 'clauses', 'document_vectors', 'documents',
'parent_documents', 'search_chunks', 'search_documents', …billing…, …outbox…]`.
Today the same command (without the last two imports) yields **7** tables and no `clauses` /
`parent_documents`. Also `rg -c "^import (app|database)\." src/alembic/env.py` → `5`;
`uv run alembic upgrade head --sql > /dev/null; echo $?` → `0` (env.py still imports cleanly);
`uv run ruff check src/ 2>&1 | tail -1` → still `Found 123 errors.`

---

### Step 10 — Delete the reconciliation subsystem (1129 lines) with its **two** coupled config edits
**Dep:** Step 9 (`ParentDocument`/`Clause` already rehomed — OC2). **Coverage: ZERO**, and the
baseline is explicit that green here proves nothing: *"Deleting reconciliation will produce no test
signal at all."*

Delete: `src/app/shared/langgraph_layer/reconciliation/` (5 files, 618 lines),
`src/tasks/memory_decay_reconciliation_tasks.py` (209), and the **remnant** of
`src/database/schemas/memory_schema.py` (the module-local `Base` at `:51` plus `Entity`,
`relationships`, `events`, `memory_versions` — the four tables no migration ever created).

Coupled edits, same commit:
- `src/tasks/__init__.py:6-9` (the `from .memory_decay_reconciliation_tasks import (…)` block) and
  `:18-20` (three `__all__` entries). Without this, **every celery worker fails at import**.
- `pyproject.toml:538` — the per-file-ignores key
  `"src/app/shared/langgraph_layer/reconciliation/nodes.py"`. **SCOUT GAP:** neither scout report
  mentions it. Ruff does not error on a stale key by default, so this leaves no signal if missed.

D10 is the record that memory decay is dropped deliberately: `_compute_decay`
(`memory_decay_reconciliation_tasks.py:51`) is the repo's only decay formula and dies here. Change 4's
`design.md` Non-Goals and `adrs.md` must carry it. Do **not** preserve it "just in case" — that is
exactly the dead-code pattern this change removes.
Naming trap: `billing.reconciliation` (`tasks/billing_tasks.py:253,346-348`, beat entry
`connections/celery.py:272-275`) is a **live, scheduled, unrelated** subsystem. Do not touch it.

**Proof:**
```bash
uv run python -c "import tasks; print(sorted(tasks.__all__))"
# → ['add','ingest_document','ingest_search_document','process_document',
#    'send_password_reset_email','send_verification_email']   — 6 names, was 9
uv run python -c "from app.connections.celery import celery_app as c; print(len(c.conf.beat_schedule), sorted(c.conf.include))"
# → 4 ['tasks.auth_email_tasks','tasks.billing_tasks','tasks.example','tasks.search_tasks']
#   — BOTH lists unchanged by this step (verified today); the deleted module was in neither.
#   Note in passing: tasks.document_tasks is ALSO absent from include — that is item 198.4, change 1.
rg -n "reconciliation" src/ | rg -v "billing"        # → exit 1, zero hits
rg -n "memory_schema|ReconciliationGraph" src/       # → exit 1
```
plus `PROOF-BOOT` → 6× OK and the pytest summary line **unchanged** at
`22 failed, 55 passed, 11 warnings, 13 errors`. Per `baseline-tests.md`: *if any number moves here,
something imported reconciliation that you did not expect — that is a finding, not noise.*

---

### Step 11 — Fix the `app.state` name mismatches in `profile/router.py` (B1/B2)
**Dep:** Step 0. Independent of every other step. **Coverage: ZERO** — no test touches
`features/profile/`. Non-test evidence: `PROOF-STATE`.

`_get_profile_service` (`profile/router.py:27-32`) reads `request.app.state.storage` (`:29`) and
`request.app.state.mongodb` (`:30`). Lifespan writes **`object_store`** (`lifespan.py:108,112,270`)
and **`db`** (`:180,183,186`) — never those two names. `profile_router` is mounted at `api/v1.py:15`,
so **every** profile endpoint raises `AttributeError` on its first request today.

Rename the *reads*, not the writes: `object_store` and `db`. Do not rename `app.state.redis` at `:31`
— that one is correct. Note `object_store` is set to `None` on failure (`lifespan.py:112,270`) while
the annotation at `:29` is a non-optional `StorageService`, so the honest fix reads through
`getattr(request.app.state, "object_store", None)` and raises `ServiceUnavailableException` when it
is `None` — the same shape `agent_saul/dependencies.py:46-48` already uses. A bare rename converts an
`AttributeError` into a `None` that fails later and further away.

**Proof:** `PROOF-STATE` no longer prints `storage` or `mongodb` (five `UNWRITTEN:` lines remain, not
seven); `rg -n "app\.state\.(storage|mongodb)" src/` → exit 1;
`uv run ty check src/ 2>&1 | tail -1` → no worse than `Found 46 diagnostics`; `PROOF-BOOT` → 6× OK.

---

### Step 12 — Fix `UserIdDep` repo-wide (D5.2)
**Dep:** Step 0. **Coverage: ZERO**, and worse — the 13 `client`-fixture collection errors mean there
is **no TestClient harness at all**, so the 401-instead-of-500 behaviour cannot be asserted by a test
today (Fog F3). Non-test evidence: `PROOF-STATE` plus OpenAPI generation.

Five sites read `request.state.user_id`; nothing in `src/` ever assigns it and there is no auth
middleware (`main.py:77-94`):

| site | mounted? | status |
|---|---|---|
| `features/documents/dependencies.py:61-62` | **YES** (`api/v1.py:16`) | **live `AttributeError` on 6 endpoints** (`documents/router.py:31,55,67,79,91,105`) |
| `features/search/dependencies.py:44-45` | no | latent |
| `features/ingestion/dependencies.py:11-12` | no | latent |
| `features/agent_saul/dependencies.py:56` | **YES** (`api/v1.py:17`) | self-documented `Stub — replace with your project's JWT/session auth dependency` |
| `features/crawler/router.py:28` | no (mounted nowhere) | guarded by `hasattr`, **not a break** |

The repo already has the right seam: `auth/dependencies.py:88` `get_token_claims` returns
`TokenClaims` (`auth/security.py:74-84`) whose `sub` is the user id, with **zero DB round trips**.
Rewrite all four unguarded sites to:

```python
async def get_current_user_id(claims: Annotated[TokenClaims, Depends(get_token_claims)]) -> str:
    return claims.sub
```

This is a **behaviour change, deliberately**: unauthenticated calls to the six mounted documents
endpoints go from `500 AttributeError` to `401` from `UnauthorizedException`. That is the openspec
delta this step carries. In `crawler/router.py:28` the `hasattr(request.state, "user_id")` branch
becomes provably dead once nothing sets it — either delete the branch or leave it and record it; do
not add a `request.state.user_id` writer to keep it alive.

**Proof:** `rg -n "request\.state\.user_id" src/` → exit 1 (today: 5 hits);
`PROOF-STATE` no longer prints a `user_id` row (three `UNWRITTEN:` lines remain: `ingestion_graph`,
`saul_graph`, `langgraph_checkpointer` — all owned by later changes);
`uv run python -c "from app.main import app; s=app.openapi(); print(len(s['paths']))"` succeeds and
prints the same path count as before the change (the deps are not query/body params, so the schema
must not grow — a changed count means a signature leaked into the public API);
`uv run ty check src/ 2>&1 | tail -1` → no worse than 46; `PROOF-BOOT` → 6× OK.

---

### Step 13 — Item 199 annotation residue (**SCOUT CORRECTION — the named defect is already fixed**)
**Dep:** Step 0. **Coverage: ZERO.**

`dispositions.md` row 199 says *"`DocumentQueryService.__init__` uses `object | None` for
redis/graphiti"*. **It does not, today.** `features/documents/service.py:232-242` reads
`redis: Redis | None` and `graphiti: Graphiti | None`, and `tests/performance/todo.md:190` marks item
199 `DONE`. The dispositions row was written from the todo text, not from the code.

The genuine `object`-annotation residue inside the change-0 blast radius is two sites:
`features/documents/service.py:815` `embedding_fn: object` and
`features/documents/legal_metadata.py:40` `llm: object`. `graphiti_verifier.py:88`
`_extract_search_blob(item: object)` is **correct as written** — it accepts genuinely unknown input —
and must not be changed. Give `embedding_fn` a `Callable`/`Protocol` type and `llm` a
`BaseChatModel`, both under `TYPE_CHECKING` per the house import rules.

**Proof:** `rg -n ":\s*object\b" src/app/features/documents/` → exactly one hit,
`graphiti_verifier.py:88` (today: three); `uv run ty check src/ 2>&1 | tail -1` → no worse than
`Found 46 diagnostics` — and it should *drop*, since `documents/service.py` carries 3 of the 46 and
`legal_metadata.py` carries 1; `uv run ruff check src/ 2>&1 | tail -1` → still `Found 123 errors.`

---

### Step 14 — Health probe: graphiti + cognee (item 198.2)
**Dep:** Step 11 (OC6 — probe a state surface whose read/write contract is already correct).
**Coverage: ZERO usable** — `tests/integration/test_health.py` is 6 of the 13 `client`-fixture
collection errors, so the health endpoint has no working test. Non-test evidence: direct service
instantiation (below), which needs no FastAPI and no services.

Today `HealthService` (`features/health/service.py:24-38`) takes five clients and
`get_health` (`:56-98`) reports seven checks; `_check_neo4j` (`:159`) exists but **nothing probes
graphiti or cognee**, while `lifespan.py:218-223` sets `app.state.graphiti = None` and continues on
failure and `:207` sets `app.state.cognee_config` unconditionally. The probe is the only observable
signal that the degradation happened.

Four coordinated edits: two constructor params + two `_check_*` methods; two new
`HealthChecksDTO` fields (`dto.py:24-30` — the model is `extra="forbid"`, so the DTO edit is
**mandatory**, not cosmetic); two `getattr(request.app.state, …, None)` providers in
`health/dependencies.py` mirroring `get_health_neo4j_driver` (`:29-30`); and inclusion in
`_compute_overall_status` (`service.py:238-…`).

**The semantic decision this step must make explicit:** `_not_configured()` must **not** flip overall
status to `unhealthy`, or every dev box without Neo4j starts returning 503 from a mounted endpoint in
both `api/v1.py:13` and `api/v2.py:9`. Mirror the existing neo4j treatment exactly and say so in
`design.md`; graphiti-absent is *degraded*, not *down*.

**Proof:**
```bash
uv run python -c "
import asyncio
from app.features.health.service import HealthService
s = HealthService(None, None, None, None, None, graphiti=None, cognee_config=None)
r = asyncio.run(s.get_health())
print(r.data.checks.graphiti['status'], r.data.checks.cognee['status'], r.data.status, r.status_code)"
# → not_configured not_configured  <same overall status+code as before this step>
rg -n "graphiti|cognee" src/app/features/health/dto.py   # → 2 hits (the two new fields)
uv run python -c "
from app.features.health.dto import HealthChecksDTO
print(len(HealthChecksDTO.model_fields), sorted(HealthChecksDTO.model_fields))"
# → 9 ['celery','cognee','database','disk','graphiti','memory','neo4j','postgres','redis']  (was 7)
```
The overall-status half of the assertion is the one that matters: run it **before** the step, record
the value, and require it unchanged after.

---

### Step 15 — RECOMMENDED ADDITION (outside the brief's enumerated scope): `src/app/utils/embedding.py:5`
**Dep:** Step 0. **Coverage: THE ONLY STEP IN CHANGE 0 WITH REAL TEST COVERAGE.**

Not in the scope list I was given, and no locked decision covers it — so this is a proposal, not a
plan item, and the orchestrator/user should accept or reject it explicitly. The case for putting it
here: it is a one-line fix, it is a *confirmed runtime break on the live ingestion path* (the exact
change-0 category), and it is the only change-0-eligible step that any test exercises.

`embedding.py:5` is `from app.utils import logger`, which binds the **submodule**
`app.utils.logger`, not the loguru object re-exported at `app/utils/__init__.py:59`. So
`logger.warning(...)` at `:22` raises `AttributeError: module 'app.utils.logger' has no attribute
'warning'` on **every** dimension mismatch. `normalize_embedding` has 15 callers including
`ingestion_kb/nodes.py`, `retrieval_kb/nodes.py`, and `features/documents/service.py`. Fix:
`from app.utils.logger import logger`.

**Proof:** `uv run pytest -q --no-header -p no:cacheprovider 2>&1 | tail -1` →
`16 failed, 61 passed, 11 warnings, 13 errors` (6 failures in
`tests/unit/documents/test_normalize_embedding.py` convert to passes; **passed rises 55 → 61**, the
cheapest verifiable win in the baseline). Exit code stays 1 because of `--cov-fail-under=80` — compare
the line, never `$?`.

---

### Step 16 — Final gate
**Dep:** all of 1-14 (15 if accepted).

**Proof — every rung compared to `/tmp`, none by exit code:**
```bash
uv run ruff format --check src/                                   # see Fog F4 — baseline UNMEASURED
uv run ruff check src/        2>&1 | tail -1   # Found 123 errors.        (was 125; -2 from Step 2)
uv run ty check src/          2>&1 | tail -1   # ≤ Found 46 diagnostics   (should drop, Step 13)
ast-grep scan src/            2>&1 | tail -2   # Error: 4 error(s) found in code.   (exit 0 — compare the COUNT)
uv run pytest -q --no-header -p no:cacheprovider 2>&1 | grep -E '^(FAILED|ERROR)' | sort > /tmp/after.txt
diff /tmp/base-failures.txt /tmp/after.txt     # empty, OR exactly the 6 test_normalize_embedding lines removed (Step 15)
uv run alembic heads 2>/dev/null | grep -c '(head)'              # 1
uv run alembic upgrade head --sql > /dev/null; echo $?           # 0
/home/harmeet/.bun/bin/openspec validate --all 2>&1 | tail -1    # Totals: 17 passed, 6 failed (23 items)
```
plus `PROOF-BOOT` → 6× OK and `PROOF-STATE` → exactly three `UNWRITTEN:` lines
(`ingestion_graph`, `saul_graph`, `langgraph_checkpointer`).

**GREEN for change 0 means:** pytest failing-set identical to baseline (or 6 smaller), ruff ≤ 123,
ty ≤ 46, ast-grep = 4, openspec failed = 6, alembic heads = 1. It does **not** mean any exit code is 0
— three of the six rungs are red before this refactor starts and stay red after it.

## Openspec mapping

`openspec/specs/` was enumerated today: **20 capabilities**, not 21/22 as earlier notes said —
`cognee-v1-api`, `datetime-utc-cleanup`, `llm-injection`, `mcp-context-di`,
`mcp-directory-restructure`, `mcp-server-codemode`, `mcp-server-composition`,
`mcp-server-pagination`, `mcp-server-prompts`, `mcp-server-resources`, `mcp-telemetry`,
`mcp-testing`, `noqa-documentation`, `outbox-helper-extraction`, `pattern-matching-standard`,
`session-required`, `settings-validation`, `test-mock-isolation`, `transactional-outbox`,
`typed-exception-handling`. **None covers migrations, health probing, `app.state` wiring, or request
identity.** So change 0 needs new capabilities — which matches house style: every existing capability
is a narrow, change-shaped slug created by archiving one change (most still carry
`## Purpose — TBD - created by archiving change …`).

**Change ID:** `cleanup-foundation` (bare slug; the archive adds the `YYYY-MM-DD-` prefix — D12).
**No `skip_specs`.** Three of change 0's steps change externally visible behaviour, so a zero-delta
opt-out would be wrong (`schema.yaml:49-59`).

### New capabilities (4)

| capability | covers | steps |
|---|---|---|
| `migration-chain-integrity` | The migration chain SHALL have exactly one head, and every revision SHALL be runnable against an empty database (no ALTER without a CREATE). | 7, 8 |
| `orm-metadata-registration` | Every live SQLAlchemy model module SHALL be registered on the single shared `Base.metadata` that alembic autogenerate reads, so autogenerate never proposes dropping a live table. | 9 |
| `request-identity-from-token` | The authenticated user id SHALL be derived from validated access-token claims, not from unset request state. Unauthenticated requests SHALL receive 401, never 500. | 12 |
| `dependency-health-probe` | The health endpoint SHALL report the state of every client the lifespan initialises, including those it degrades silently; an unconfigured optional client SHALL report `not_configured` without failing the overall probe. | 14 |

Naming rationale: kebab-case, scoped to *observable behaviour* rather than to the files touched.
`migration-chain-integrity` is deliberately broader than "merge two heads" so changes 1 and 2 (which
both add migrations) can add requirements to it instead of minting more slugs.

### Modified capability (1)

`typed-exception-handling` — **`## MODIFIED Requirements`** on `### Requirement: Database operations
SHALL catch asyncpg.exceptions.PostgresError`. Its `#### Scenario: Reconciliation fetch failure
catches PostgresError` (`openspec/specs/typed-exception-handling/spec.md:148-150`) becomes
unsatisfiable the moment Step 10 lands — the code it describes no longer exists. Measured today: that
requirement block is **30 lines with 6 scenarios** (the spec has 11 requirements total), so a
`MODIFIED` delta copying the whole block is tractable. Replace the reconciliation scenario with an
equivalent live asyncpg site; the sibling `Outbox publish failure` scenario already covers the
requirement's substance, so the requirement itself does not weaken.

**Two traps here, both real:**
1. `MODIFIED` must copy the **entire** requirement block, `### Requirement:` through every scenario,
   with the header matching whitespace-insensitively. Partial content silently loses detail at archive.
2. `spec/typed-exception-handling` is **already one of the 6 baseline failures**. Diagnose it first
   (`openspec validate typed-exception-handling --type spec`) so you can tell your delta's errors from
   the pre-existing ones. If the pre-existing failure is in this very requirement, drop the `MODIFIED`
   delta and record the orphaned scenario in `design.md` instead — do not fight an unrelated red.

### What gets NO delta

The deletions (Steps 2-6, 10) and the annotation fix (Step 13) are pure refactor: no externally
visible behaviour changes. Per `schema.yaml:49-59`, **do not invent a requirement to cover them.**
They appear in `tasks.md` and in `design.md`'s Migration Plan, nowhere in `specs/`.

### Non-Goals that `design.md` must carry (recorded gaps, not omissions)

- Memory decay / curation / dedup are dropped with reconciliation (D10). Change 4 owns the record.
- `shared/agents/**` shadow deletion deferred to change 3 (OC1).
- `entity_extractor.py:78`'s phantom `graphiti_graph` import (B5) and
  `rag_agent_advanced.py:119,198,267,373`'s phantom `ingestion.embedder` (manifest §6) are **not**
  fixed here — both are unsanctioned by the carve-outs and sit in change 1's blast radius.
- `app.state.saul_graph` / `langgraph_checkpointer` / `ingestion_graph` stay unwired (B3/B4 → changes
  1 and 3). Step 14's probe will report the degradation rather than hide it.
- `mcp_core/__init__.py:2-19`'s 18 `undefined-export` errors are **18 of the 123 remaining ruff
  errors** and are untouched here.

## Risks

| # | Risk | Mitigation |
|---|---|---|
| R1 | **The clean-DB migration fix cannot be proven locally.** `9f4a1b7c6d2e` needs `pg_textsearch` (`USING bm25`) and `diskann`; `docker-compose.yml:23` pins `timescale/timescaledb-ha:pg18`, which ships pgvector/pgvectorscale but **not** VectorChord's `pg_textsearch`. The `--sql` proof in Step 8 checks *ordering*, not *executability*. | Treat Step 8's real-DB verification as an explicit Open Question in `design.md`, with Fog F1's command as the resolution path. Do not claim the chain is clean-DB-runnable until F1 is answered. |
| R2 | **`docker-compose.yml:34` bind-mounts `./scripts/init-db.sql`, which does not exist** (`ls scripts/` → 8 files, no `.sql`). Docker will create a *directory* at that path inside the container's init dir, which can make `docker compose up -d timescale` behave unpredictably — i.e. the one command anyone would reach for to test R1 is itself suspect. | Fix or remove the mount before using compose as evidence. Out of change 0's scope; record it. |
| R3 | **Deleting 1129 lines with zero test coverage produces zero signal.** `baseline-tests.md` is blunt: green after the reconciliation deletion *"means nothing was checked, not that nothing broke."* | Every deletion step's proof is an **import probe + `rg` emptiness + unchanged pytest summary**, never "tests pass". `PROOF-BOOT` covers all six boot entry points including the celery `tasks` package. |
| R4 | **Editing a historical alembic revision (Step 8) is normally forbidden.** | It is safe *here* for three verified reasons (ancestors are never re-run; `IF NOT EXISTS` is idempotent; the revision has never successfully run on any clean DB). State all three in `design.md` Decisions with "new revision inserted before `9f4a1b7c6d2e`" as the alternative considered and rejected. |
| R5 | **Step 9's model move could resurrect dead schema.** `clauses`/`parent_documents` come back onto `Base.metadata` — someone may read that as "these are blessed", contradicting change 2's consolidation onto `UnifiedDocument`/`UnifiedChunk`. | The move is for *autogenerate safety*, not endorsement. Say so in `contract_kb.py`'s module docstring and in change 2's `design.md` Context. Change 2 decides their fate (dispositions row 184 → "A+: retarget, don't leave stale"). |
| R6 | **Step 12 changes a mounted API's failure mode** (500 → 401 on six documents endpoints). Any client currently treating 500 as "retry" will now see a terminal 401. | That is the point, and it is the delta. Flag BREAKING explicitly in `proposal.md` per `config.yaml:39-43`. |
| R7 | **`HealthChecksDTO` is `extra="forbid"`** and the health router is mounted on **both** `api/v1.py:13` and `api/v2.py:9`. Adding fields changes two public response shapes at once. | Additive only, never rename; assert the property count (7 → 9) and assert overall status/`status_code` unchanged for the all-`None` case. |
| R8 | **Ruff's `blanket-type-ignore` count (76 of 123) can move for unrelated reasons** as files are deleted, making "≤ 123" a soft gate. | Compare `tail -1` counts *and* keep `/tmp/base-ruff.txt`; if the count drops, confirm the drop maps to deleted files (`git diff --stat`) rather than to a suppressed real error. |
| R9 | **`tests/unit/billing/` and `tests/unit/search/` lack `__init__.py`** while siblings have one, and two files share the basename `test_circuit_breaker.py`. Latent pytest "import file mismatch". | No step here adds a test file. If one is added, give it a unique basename. Recorded, not fixed. |

## Fog

Each item is what I could **not** establish, and the single command that would establish it.

**F1 — Can the repaired chain actually run on an empty database?** The `--sql` proof is static; it
never executes `CREATE EXTENSION pg_textsearch` or `USING bm25`. Everything about Step 8's
executability rests on extension availability.
```bash
docker compose up -d timescale && sleep 15 && docker exec timescale_db \
  psql -U postgres -c "SELECT name FROM pg_available_extensions WHERE name IN ('vector','pg_textsearch','vchord_bm25','vectorscale');"
```

**F2 — Are `parent_documents`, `clauses`, `entities`, `relationships`, `events`, `memory_versions`
present in the deployed database?** Inherited open Fog from both scouts, assigned to this plan as a
precondition and still unanswered — I have no DB credentials and no container is running
(`docker ps` → empty). It decides whether Step 10's deletion of the four orphan models is pure
subtraction or leaves live tables un-modelled.
```bash
docker exec timescale_db psql -U postgres -d langchain_db -c "\dt"
```

**F3 — Does an unauthenticated request to a documents endpoint currently 500, and does it 401 after
Step 12?** FastAPI resolves *all* dependencies for a route; whether `get_token_claims` raises before
`get_document_query_service` needs Postgres is not something I could determine statically, and there
is no `client` fixture to test with (13 collection errors).
```bash
uv run python -c "
from fastapi.testclient import TestClient
from app.main import app
print(TestClient(app, raise_server_exceptions=False).get('/api/v1/documents/').status_code)"
```

**F4 — `uv run ruff format --check src/` has never been run.** Inherited Fog from
`baseline-tests.md`, still open. Two `invalid-syntax` errors mean it may not even parse today, and
Step 2 removes both — so this rung's baseline may *change* as a side effect of Step 2 and there is no
"before" to compare against.
```bash
uv run ruff format --check src/ 2>&1 | tail -3   # run BEFORE Step 2, and again after
```

**F5 — What exactly makes `spec/typed-exception-handling` fail today?** It is 1 of the 6 baseline
failures and it is the one spec change 0 wants to modify (Step 10's orphaned scenario). Without the
diagnosis I cannot say whether a `MODIFIED` delta is safe or will collide with the existing red.
```bash
/home/harmeet/.bun/bin/openspec validate typed-exception-handling --type spec
```

**F6 — The pre-ALTER shape of `clauses` is reconstructed, not recorded.** I derived it as
`Clause` (`memory_schema.py:190-249`) *minus* the eight columns `9f4a1b7c6d2e:64-99` adds. If the
table was actually created out-of-band with a different shape (which R1/F2 suggest is likely — some
tables were created by hand), the reconstruction is a guess about history and `IF NOT EXISTS` will
silently keep the real one. That is the safe failure mode, but it means the two shapes may diverge.
```bash
docker exec timescale_db psql -U postgres -d langchain_db -c "\d+ clauses"
```

**F7 — Whether `9f4a1b7c6d2e`'s `batch_alter_table` emits usable offline SQL at all.** I could never
reach it: the `--sql` run died at "Multiple head revisions" before executing any revision. If batch
mode needs `recreate=` semantics offline, Step 8's proof may need `--sql 8a7d9b1c2e3f:9f4a1b7c6d2e`
scoped to that one revision instead of the whole chain.
```bash
uv run alembic upgrade 8a7d9b1c2e3f:9f4a1b7c6d2e --sql 2>&1 | tail -20
```

---

# ADDENDUM — live-database supersession (2026-08-17, after `findings-database.md`)

`docs/relay/findings-database.md` is authoritative over every DB statement made above it and over every
scout guess. It was produced by connecting to the **actual** database. This addendum is **binding where it
conflicts with anything earlier in this file.**

## A0 — What above is superseded, and what survives

| Earlier content | Status |
|---|---|
| Step 7 (merge the two heads) | **SURVIVES unchanged.** Still necessary, still first. |
| Step 8 (`CREATE TABLE clauses` prepend to `9f4a1b7c6d2e`) | **SURVIVES, but its justification changes** — see A3. It is now a *replayability* fix only; it will never execute against the live DB. |
| Step 9 (harvest `ParentDocument`/`Clause`, then register every model in `env.py`) | **HALVED.** The `env.py` registration half was **done under me by concurrent work** — see A2. The harvest half survives, retargeted. |
| R1, R2 (compose image / extension availability) | **RETIRED.** Wrong target; the DB is Timescale Cloud. See A1. |
| F1 (`pg_textsearch` availability) | **CLOSED — favourable.** Available 1.3.0, not yet installed. `CREATE EXTENSION IF NOT EXISTS pg_textsearch` will succeed. D5.1 stands on solid ground. |
| F2 (live/orphan status of `parent_documents`, `events`, `memory_versions`) | **CLOSED.** None of them exist. |
| F6, F7 (`alembic upgrade head --sql` baseline captures) | **INVALIDATED TWICE** — once by the live findings, once by the working tree changing underneath the plan. See A2. |
| Everything else (Steps 0–6, 10–16; OC1's cycle resolution; the openspec mapping) | **Unaffected.** |

## A1 — The deployment target is Timescale Cloud, not the compose service

`POSTGRES_URL` resolves to `qbid1qrc75.nnro3dh8tf.tsdb.cloud.timescale.com:39662/tsdb`, PostgreSQL 18.0.4,
`sslmode=require`. Nothing listens on `localhost:5432`; the compose `timescale` service has never been up in
this working copy. Consequences for this plan:

- **R1 and R2 are retired.** "Check what the image ships" answers the wrong question — we do not control the
  managed instance's extension set, and the set that matters is the cloud one.
- `docker-compose.yml:34`'s bind-mount of the nonexistent `./scripts/init-db.sql` is confirmed as a fiction and
  is **not** worth fixing in change 0; it is dead configuration on a service nothing uses. Record it as a
  Non-Goal, do not delete it here — deleting a compose service is a deployment change and change 0 must stay
  independently committable without touching deploy.
- Installed: `vector` 0.8.2, `vectorscale` 0.9.0, `timescaledb` 2.29.1. Available-not-installed:
  `pg_textsearch` 1.3.0, `pg_trgm`, `unaccent`, `uuid-ossp`, `vchord`. **Not available at all:**
  `vchord_bm25`, `pg_search`. Access methods present: `diskann`, `hnsw`, `ivfflat` — **no `bm25` yet**,
  because `pg_textsearch` is not installed.

## A2 — The working tree changed underneath this plan. Step 0 is now mandatory, not hygienic.

While this plan was being written, concurrent work **split `features/billing/` into six feature packages**
(`audit`, `dunning`, `invoices`, `payments`, `plans`, `subscriptions`, `webhooks`) via `git mv`. Evidence:
26 `R`/`RM` entries in `git status`, `src/app/api/v2.py` rewritten to assemble `billing_router` from six
routers, `tests/unit/billing/` split into `tests/unit/{invoices,payments}/`.

Two consequences the implementer must absorb before touching anything:

1. **Every baseline captured earlier in this file is stale.** The `ruff 125` / `ty 46` / `55 passed` /
   `openspec 16-passed-6-failed` / `alembic --sql exit 255` numbers were taken against the pre-split tree.
   Mid-write, the same commands began failing with `ModuleNotFoundError: No module named
   'app.features.billing.models.audit'` and then `ImportError: cannot import name 'RefundRequestDTO' from
   'app.features.subscriptions.dto'` — i.e. the split was in progress and transiently broken.
   **Step 0 must be re-run from scratch and its numbers used in place of every number above.** Do not trust a
   single figure in this file as a gate value; trust only the freshly captured `/tmp/baseline-*` files.

2. **`src/alembic/env.py`'s model registration was already fixed — for billing only.** `env.py` now imports
   14 model modules at `:23-36` (`app.features.audit.model`, five `invoices.*`, two `payments.*`,
   `plans.model`, two `subscriptions.*`, two `webhooks.*`, `app.shared.outbox.model`). So the half of Step 9
   that said "register every model module" is **partly done by someone else**.

   What is still missing from `env.py`, and is still change 0's:

   | Not registered | Table(s) | Effect on `--autogenerate` |
   |---|---|---|
   | `app.features.documents.model` | `documents`, `chunks` | would emit `DROP TABLE` for both, once they exist |
   | `app.features.search.model` | `search_documents`, `search_chunks` | same |
   | `database.schemas.memory_schema` | `parent_documents`, `clauses`, + 4 orphans | **cannot be fixed by an import** — `memory_schema.py:51` declares its own `declarative_base()`, so importing it registers nothing on `database.Base.metadata`. This is exactly why Step 9's harvest half survives. |

   Also still true and still worth one line of the diff: `env.py:38-42` wraps `target_metadata = Base.metadata`
   in `try/except ImportError`, but every import sits at module scope **above** it, so the `except` branch is
   unreachable and the `logger.warning` is dead. Fold that deletion into Step 9.

   Proof (post-split, run after Step 9): `uv run python -c "from alembic import context" ` is not the check —
   instead
   `cd src && uv run python -c "import env" 2>/dev/null || uv run python -c "
   import database, app.features.documents.model, app.features.search.model
   print(sorted(database.Base.metadata.tables))"` →
   expect the printed list to contain `chunks`, `documents`, `search_chunks`, `search_documents` alongside the
   15 billing tables and the 2 outbox tables. It must **not** contain `clauses`/`parent_documents` unless the
   Step 9 harvest also ran.

**Ordering note added by this addendum:** Step 0 now has a hard precondition of its own — the concurrent
billing split must be **committed and importable** before change 0 begins. Proof:
`uv run python -c "import app.main"` exits 0 and `uv run python -c "import app.api.v2"` exits 0.
If either fails, change 0 has not started yet; the tree is mid-refactor and every gate value is noise.

## A3 — THE HEADLINE: the database was stamped, not migrated — and it is worse than `findings-database.md` §4 states

`alembic_version` holds one row: **`0004`**. Alembic therefore believes
`c0c17c6eb1cc → 2bc7726317f6 → 8a7d9b1c2e3f → 9f4a1b7c6d2e → 0001 → 0002 → 0003 → 0004` are all applied.

The live inventory is 16 tables and `findings-database.md` §4 states it is **complete**:

```
alembic_version, audit_logs, currencies, email_templates, fx_rates, invoice_batches,
invoice_line_items, invoice_voids, invoices, payment_receipts, payments, plans,
reports, subscriptions, trial_extensions, webhook_events
```

`findings-database.md` §4 checked 8 tables. **Cross-referencing that complete inventory against the `create_table`
calls in each stamped revision escalates the count to 11, and reveals one live production break the findings
report did not name.** Derivation (each row = a `create_table` in a revision alembic marks applied):

| Revision (marked applied) | Creates | Present in the 16? |
|---|---|---|
| `c0c17c6eb1cc` | `chat_messages` (`:26`), `chat_sessions` (`:47`), `document_vectors` (`:64`) | **NO, NO, NO** — *not checked by §4* |
| `2bc7726317f6` | (rename `document_vectors.metadata`→`meta_data`) | n/a — never ran, target table absent |
| `8a7d9b1c2e3f` | `search_documents` (`:32`), `search_chunks` (`:45`) | NO, NO |
| `9f4a1b7c6d2e` | `parent_documents` (`:28`); **ALTERs** `clauses` (`:63`) | NO; `clauses` NO |
| **`0001`** | **`outbox_events` (`:21`), `dead_letter_events` (`:41`)** | **NO, NO** — *not checked by §4* |
| `0002` | the 15 billing tables (`:39`–`:355`) | **YES, all 15** |
| `0003`, `0004` | ALTERs on billing tables | consistent |

So **only `0002`–`0004` genuinely ran.** The most likely history is `alembic stamp 0001` (or `stamp
9f4a1b7c6d2e`) followed by `alembic upgrade head`, which applied `0002`–`0004` and nothing before.

### A3.1 — NEW CONFIRMED RUNTIME BREAK: the transactional outbox writes to a table that does not exist

This is the addendum's own finding and it ranks with `UserIdDep` (D5.2) as a live break on mounted surface.

`src/app/shared/outbox/helper.py:31` executes a raw `INSERT INTO outbox_events (...)`. Its callers:

| Caller | Router mounted? | Status |
|---|---|---|
| `features/documents/service.py:184` (upload → `event_type="tasks.documents_ingest"`) | **YES** (`api/v1.py:16`) | live 500 |
| `features/auth/service.py:494` and `:516` (`_publish_outbox_event`, `:481`, called from the session path at `:435`) | **YES** (auth is mounted) | live 500 |
| `features/search/service.py:106` | no (unmounted) | latent |

And the **reader runs unconditionally at boot**: `lifespan.py:285` calls `_init_outbox_relay` (`:116`), which
constructs `OutboxRelay` (`:125`) and launches `asyncio.create_task(relay.run_listener())` (`:131`).
`relay.py:46,101` `SELECT … FROM outbox_events`. Every boot starts a background task querying a nonexistent
relation — the invisible-failure register again, this time inside a fire-and-forget task where the exception is
not awaited by anyone.

**Ordering consequence, and it is load-bearing.** Today the documents upload path fails at `UserIdDep`
(`AttributeError`) *before* it reaches `with_outbox`. **Step 12 unmasks this break**: fix `UserIdDep` and the
same endpoint starts failing one layer deeper with `asyncpg.UndefinedTableError`. Therefore the outbox tables
must exist **before or in the same commit as Step 12**, or change 0 trades one 500 for another and the
acceptance test for Step 12 cannot pass. This is the "step that temporarily breaks a path, paired with its
restoring step" case the brief asked to be called out.

## A4 — Migration shape: the recommendation is **half adopted, half rejected**

`findings-database.md` §4.5 recommends: *"merge the two heads, then add one new migration that creates the
target schema outright (unified `documents`/`chunks` plus whatever `search/` capability survives D5.1)."*

**Adopted:** merge the two heads. **Rejected:** creating the target schema in change 0.

Two reasons the second half fails, and neither is a preference:

1. **It smuggles changes 1 and 2 into change 0.** The target schema is not known yet. `Vector(768)` must become
   `Vector(settings.EMBEDDING_DIMENSION)` — that is item 198.3, **change 1**, and it is blocked on resolving the
   768-vs-1536 conflict at `document_processing/embedder.py:26-29`. Collapsing `search_*` into
   `documents`/`chunks` is **change 2**, and it still has four unresolved cells (`chunks.user_id` NOT NULL with
   no source value; `documents.object_uri` NOT NULL with no source value; `UnifiedChunk` has no `updated_at`;
   the hardcoded constraint name at `search/repository.py:157`). A change-0 migration that picks answers to
   those pre-empts two later changes and violates D8's "ordering is load-bearing" and D3's work order.
2. **It would create tables for change 2 to immediately drop.** Of the 11 phantom tables, `chat_messages`,
   `chat_sessions`, `document_vectors` have zero live readers (`strategies.py` is entirely commented out,
   `scout-persistence-docling.md` §5), and `search_documents`/`search_chunks`/`parent_documents`/`clauses` are
   deleted or collapsed by D5.1 + change 2. Creating them now is net-negative work and re-establishes the
   original disease in mirror image: DDL with no reader instead of a reader with no DDL.

### Options considered

| Option | Mechanic | Verdict |
|---|---|---|
| **A** `alembic stamp base` + `upgrade head` | Re-runs `0002`–`0004` against the 15 tables that genuinely exist → `DuplicateTableError` on `plans` | **Rejected** (findings §4.5 rejects it too, and for the right reason) |
| **B** One post-merge migration that (re)creates **all 11** stamped-but-absent tables | History and DB converge; alembic stops lying | **Rejected** — creates 9 tables nobody reads, 7 of which change 2 deletes |
| **C** One post-merge migration that creates the **target** schema (the recommendation) | — | **Rejected** — reasons 1 and 2 above |
| **D** One post-merge migration that creates **only what live code reads or writes today** — `outbox_events`, `dead_letter_events` — and records the remaining 9 phantoms as knowingly-phantom | Smallest honest change 0; every later change owns its own DDL | **ADOPTED** |
| **E** Operator runbook: `alembic stamp 9f4a1b7c6d2e` → `alembic upgrade 0001` → `alembic stamp 0004` | Uses `0001`'s real DDL, no copy-paste; makes history genuinely honest for `0001` | **Rejected as the default, offered as the fallback** — see below |

**Why D beats E**, even though E avoids duplicating DDL: E is a three-command manual runbook, not a committable
artifact. It does not self-apply in CI, on a teammate's machine, or on a fresh environment, and it requires an
operator to judge whether the local DB matches the assumed state. Change 0 must be independently committable and
reproducible; a markdown runbook is neither. E's one genuine advantage — no duplicated DDL — is worth recording
as the reason a **history squash to a single baseline revision** is the right long-term fix, and that squash is
a **change 2 candidate** (it can only happen once change 2 has settled the target schema), noted below as a
Non-Goal here.

**Why D is safe on replay.** The new revision duplicates `0001`'s DDL, so a future `alembic upgrade head` from
base would run `0001` (creating the tables) and then the new revision (creating them again) →
`DuplicateTableError`. The new revision therefore **must** be written as raw
`op.execute("CREATE TABLE IF NOT EXISTS outbox_events (…)")` rather than `op.create_table`, plus
`CREATE INDEX IF NOT EXISTS idx_outbox_unpublished … WHERE published_at IS NULL`. This is the same
offline-safe idempotence technique Step 8 uses for `clauses`, and for the same reason: an
`sa.inspect(op.get_bind())` guard is unavailable in `--sql` mode. `downgrade()` must be a no-op with a comment
explaining why (dropping tables `0001` also claims to own would corrupt the other revision's contract).

**What D leaves dishonest, on the record.** After change 0, alembic still claims `c0c17c6eb1cc`,
`2bc7726317f6`, `8a7d9b1c2e3f` and `9f4a1b7c6d2e` are applied when they are not. Two mitigations, both required:

- The merge revision's docstring must state this explicitly, naming the 9 remaining phantom tables, so the next
  person to read the history is not misled.
- **`alembic downgrade` past the merge point is forbidden** and must be documented as such: the downgrade bodies
  of `0001` (`:57-61` drops `dead_letter_events`, `idx_outbox_unpublished`, `outbox_events`), `8a7d9b1c2e3f` and
  `9f4a1b7c6d2e` drop tables that do not exist, so downgrade raises `UndefinedTableError`. This is already true
  today; change 0 does not create the hazard, it documents it. `design.md` carries it as an operational
  constraint and `adrs.md` as a known consequence.

**Zero rows anywhere** (findings §4.2) means D needs no backfill, no data migration and no `DROP TABLE`.
D5.1's "`DROP TABLE` + retarget" shape is refined to **retarget only** — there is nothing to drop.

## A5 — Revised and new steps (these replace the same-numbered steps above)

### Step 7 (REVISED) — Merge the two alembic heads

Unchanged in mechanic, changed in *what its proof may claim*. Inbound dependency: Step 0 (re-captured).

```bash
uv run alembic merge -m "merge documents branch into billing lineage" a71f0d7d9c12 0004
```

Docstring must name the 9 knowingly-phantom tables (A4) and state that downgrade past this point raises.

**Proof:** `uv run alembic heads` prints **exactly one** line, and that line's revision hash matches the
`revision = "…"` in the new merge file. Before: two lines (`a71f0d7d9c12`, `0004`).
Corroborating proof: `uv run alembic history | head -3` shows the merge revision with **two** parents in its
`down_revision` tuple.

**Proof this step does NOT claim:** it must *not* claim that `alembic upgrade head` now produces the schema.
It does not — the phantom revisions stay stamped. A merge revision has an empty `upgrade()`.

**Test coverage: ZERO.** No test imports alembic. Non-test evidence standing in: `alembic heads` output above,
plus `uv run alembic upgrade head --sql` exiting 0 instead of 255 with `Multiple head revisions are present`.

### Step 8 (REVISED JUSTIFICATION) — `CREATE TABLE IF NOT EXISTS clauses` prepended to `9f4a1b7c6d2e.upgrade()`

The mechanic from Step 8 above stands verbatim: raw `op.execute` (offline-safe), creating the **pre-ALTER**
shape — `Clause` (`memory_schema.py:190-249`) **minus** the 8 columns `9f4a1b7c6d2e:64-99` adds — or the
subsequent `add_column` calls fail with `DuplicateColumnError`.

What changes is *why*, and the change matters for how the implementer reasons about risk:

- **It will never execute against the live database.** The DB is stamped at `0004`, past `9f4a1b7c6d2e`. So this
  step has **zero runtime effect today**. It is purely a *replayability* fix.
- **It is still required**, because without it `alembic upgrade head` from an empty database raises at
  `batch_alter_table("clauses")` — which means no step in changes 1–4 can ever use a from-scratch database as
  its proof, and CI can never build a schema. That is a permanent tooling blocker for the rest of the refactor,
  which is why it belongs in the foundation change and not in the change that eventually deletes `clauses`.

**Proof (unchanged, and DB-free):**
`uv run alembic upgrade head --sql 2>/dev/null | grep -n -E "CREATE TABLE (IF NOT EXISTS )?clauses|ALTER TABLE clauses"`
→ the first `CREATE TABLE … clauses` line number must be **strictly less than** the first
`ALTER TABLE clauses` line number. Today the `CREATE` line is absent entirely.

**Test coverage: ZERO.** Non-test evidence: the `--sql` ordering check above, which is a genuine assertion about
generated DDL and does not need a database.

### Step 8b (NEW) — Create `outbox_events` and `dead_letter_events` (A3.1)

Inbound dependency: Step 7 (there must be a single head to add a revision onto). **Must land before or with
Step 12** (A3.1's ordering consequence).

New revision, `down_revision` = the Step 7 merge revision. Body is raw SQL only:

```python
op.execute("CREATE TABLE IF NOT EXISTS outbox_events (…)")   # mirror 0001:21-33 exactly
op.execute("CREATE INDEX IF NOT EXISTS idx_outbox_unpublished ON outbox_events (created_at) WHERE published_at IS NULL")
op.execute("CREATE TABLE IF NOT EXISTS dead_letter_events (…)")  # mirror 0001:41-53 exactly
```

`downgrade()` is `pass` with a comment: `0001` already claims ownership of these tables' lifecycle.

Column shapes must be copied from `0001` verbatim — `id String(36)` PK, `aggregate_type String(64)`,
`aggregate_id String(128)`, `event_type String(64)`, `payload JSONB`, `created_at timestamptz`,
`published_at timestamptz NULL`, `publish_attempts integer NOT NULL DEFAULT 0`, `last_error text NULL`; and for
`dead_letter_events` additionally `original_event_id String(36)`, `dead_letter_at timestamptz NOT NULL`,
`last_error text NOT NULL`. A drift here is silent: `helper.py:31`'s INSERT names only 5 columns, so a wrong
`NOT NULL` on any other column fails only at the first real write.

**Proof (requires the live DB, and it is the one step that does):**
```bash
uv run alembic upgrade head
uv run python -c "
import asyncio, asyncpg, os
from app.connections.postgres import get_database_url
async def main():
    c = await asyncpg.connect(get_database_url().replace('postgresql+asyncpg://','postgresql://'))
    for t in ('outbox_events','dead_letter_events'):
        print(t, await c.fetchval('select to_regclass($1)', t))
    await c.close()
asyncio.run(main())"
```
→ both print a non-`None` OID. Today both print `None`.

**Secondary proof, DB-free:** `uv run alembic upgrade head --sql | grep -c "CREATE TABLE IF NOT EXISTS outbox_events"`
→ `1`.

**Test coverage: ZERO** for the table's existence. There *is* an outbox spec
(`openspec/specs/transactional-outbox/`) — and note it is **one of the 6 pre-existing openspec failures** (D12),
so touching it must not be mistaken for having fixed it.

### Step 9 (REVISED — halved) — Harvest `ParentDocument` + `Clause` off the orphan `Base`

The `env.py` billing registration half is **already done** (A2). What remains:

1. Move `ParentDocument` (`memory_schema.py:154`) and `Clause` (`:190-249`) onto `database.Base`, because
   `memory_schema.py:51` declares its own `declarative_base()` and no import can register them otherwise.
   `scout-reconciliation.md` §4 says delete `memory_schema.py` whole and claims "no migration needed (no table
   exists to drop)" — the second half is now **confirmed true** (neither table exists), but the first half is
   still wrong: §4 inventories only `Base`/`Entity`/`relationships`/`events`/`memory_versions` and **omits**
   `ParentDocument` and `Clause`, which `9f4a1b7c6d2e` references. Harvest before deleting.
2. Add `app.features.documents.model` and `app.features.search.model` to `env.py`'s import block.
3. Delete the unreachable `try/except ImportError` at `env.py:38-42` (A2).

**Proof:** `cd src && uv run python -c "
import database, app.features.documents.model, app.features.search.model, app.shared.outbox.model
t = sorted(database.Base.metadata.tables)
print(len(t)); print([x for x in t if x in ('documents','chunks','search_documents','search_chunks','clauses','parent_documents','outbox_events','dead_letter_events')])"`
→ the second line lists all 8 names. Before this step it lists `outbox_events`, `dead_letter_events` only.

**Trap:** do **not** then run `alembic revision --autogenerate`. With the models newly visible and the tables
absent from the DB, autogenerate would emit `create_table` for all 8 — which is option C from A4, rejected.
Registration here exists so that *future* autogenerate runs (in changes 1 and 2) are correct, not to generate
DDL now.

**Test coverage: ZERO.** Non-test evidence: the metadata-table listing above.

## A6 — The `POSTGRES_URL` question, closed

The orchestrator assigned "does `create_async_engine` accept this URL" as an open question and then withdrew it.
**Independently verified here by reading `src/app/connections/postgres.py:30-71`, and the withdrawal is correct.**
`get_database_url()` already does all three things:

| Concern | Handled at | Behaviour |
|---|---|---|
| `postgres://` dialect alias removed in SQLAlchemy 2 | `:42-47` | rewrites `postgres://` → `postgresql+asyncpg://` (and `:36-41` handles `postgresql://`) |
| `create_async_engine` needs an explicit async driver | same | the rewrite supplies `+asyncpg` |
| URL carries no password | `:56-70` | `urlparse`, and if `not parsed.password`, re-injects `settings.POSTGRES_PASSWORD` via `urlunparse` |
| asyncpg rejects libpq query params | `:51-54` | strips `?sslmode=require`, `&sslmode=require`, `?channel_binding=require`, `&channel_binding=require` |

**So there is no scheme defect and no auth-by-env-side-channel defect on the `init_db` path.** The earlier
`InvalidPasswordError` was an artifact of probing with the raw DSN — precisely what this helper exists to prevent.

### Step 17 (NEW) — "No caller can obtain an unusable URL"

The real defect is narrower and is a *bypass* problem, not a URL problem: **two consumers read the raw,
passwordless `settings.POSTGRES_URL` instead of calling `get_database_url()`.** Full consumer census (`rg POSTGRES_URL src/`):

| Site | Reads | Status |
|---|---|---|
| `connections/postgres.py:32` (inside `get_database_url`) | raw | **correct — this is the repair point** |
| `connections/postgres.py:80` | `get_database_url()` | correct |
| `connections/postgres.py:133,138` | raw | **benign** — logging/display only (`urlparse(...).hostname`, `.split("/")[-1]`). Leave, or switch for consistency; not a defect. |
| `shared/langgraph_layer/checkpointer.py:9` | raw, in a **docstring** | documentation defect — see below |
| `shared/langchain_layer/agents/memory/cognee_client.py:111` | raw | **real** — hands Cognee a credential-less URL, and Cognee cannot recover it |
| `lifecycle/lifespan.py:297` | raw, **commented out** | becomes real the moment change 1 uncomments the checkpointer block |
| `features/auth/service.py:512` | `create_async_engine(get_database_url())` | right URL source, **wrong lifecycle** — builds a second engine outside lifespan, with its own pool, per call, and disposes it in a `finally` (`:523`). A connection-pool-per-event defect, not a credentials defect. |

Frame the fix as *"no caller can obtain an unusable URL"* — push the repair behind the single accessor rather
than patching three call sites:

1. `cognee_client.py:111` → `get_database_url()`.
2. `auth/service.py:502-524` → take the session/engine from `app.state.db_session_local` like every other caller,
   deleting the ad-hoc engine. If that is not reachable from `_publish_outbox_event`'s call site, it is a
   dependency-plumbing task and may be deferred to change 1 **on the record** — but the duplicated pool must not
   pass unremarked.
3. `checkpointer.py:9`'s docstring is **wrong in both directions** and must be corrected as part of this step,
   because change 1 will follow it: `AsyncPostgresSaver` is psycopg-based and **cannot accept a
   `postgresql+asyncpg://` URL**, while the raw `POSTGRES_URL` it currently names has no password. The
   checkpointer needs its **own psycopg-flavoured accessor** (`postgresql://` with the password injected, and
   `sslmode=require` **retained** — psycopg wants it, asyncpg does not). Change 0 writes the accessor and fixes
   the docstring; change 1 consumes it. Recording this here is what stops change 1 from adopting
   `get_database_url()` and failing at the driver.

**Two minor defects in the same helper**, both latent, both one-line, both worth taking while the file is open:

- `:57` compares `settings.POSTGRES_PASSWORD.get_secret_value() != "pass"` — a literal coupled to
  `settings.py:140`'s placeholder default `postgresql://user:pass@host/db`. If the real password were ever the
  string `pass`, injection would silently skip. Compare against the field default, or drop the guard and let a
  missing password fail loudly.
- `:61-67` appends `:{port}` after a ternary whose `else parsed.netloc` branch **already contains the port**, so
  a URL with no username yields `host:39662:39662`. Latent only because the live URL *does* have a username
  (`tsdbadmin`), which takes the truthy branch where the append is correct.
- **Third, found here and not previously reported:** the password is interpolated into the netloc as
  `f"{parsed.username}:{password}@{parsed.hostname}"` with **no percent-encoding**. Any `@ : / ? # %` in the
  password produces a silently malformed URL — `urlunparse` does not escape. Use `urllib.parse.quote(password,
  safe="")`. This is a correctness landmine on the next credential rotation, not a bug today.

**Proof for Step 17:**
```bash
rg -n "settings\.POSTGRES_URL" src/ | rg -v "connections/postgres.py:(32|133|138)|settings.py:140"
```
→ **no output** (today: 3 lines — `checkpointer.py:9`, `cognee_client.py:111`, `lifespan.py:297`).
Plus: `uv run python -c "
from urllib.parse import urlparse
from app.connections.postgres import get_database_url
u = urlparse(get_database_url())
print(u.scheme, bool(u.password), u.hostname, u.port)"`
→ `postgresql+asyncpg True <cloud host> 39662`.

**Test coverage: ZERO.** `rg -l "get_database_url" tests/` → no files. Non-test evidence: the two commands above.
The second is a genuine assertion (scheme rewritten, password present) and needs no database.

## A7 — F7 CLOSED with hard evidence, and Step 8's reconstruction is now exactly specified

Re-run today against the post-split tree (`env.py` now imports cleanly, exit 0):

```
$ uv run alembic heads
0004 (head)
a71f0d7d9c12 (head)
```

Two heads confirmed post-split — **Step 7's premise survives the concurrent refactor unchanged.**

`uv run alembic upgrade 8a7d9b1c2e3f:9f4a1b7c6d2e --sql` now **runs to completion and emits usable offline SQL**,
which closes F7 favourably on two counts:

1. **`batch_alter_table` needs no `recreate=` handling offline.** It emits plain
   `ALTER TABLE clauses ADD COLUMN …` (8 of them, output lines 35-49), then the two `UPDATE`s, then
   `ALTER COLUMN … SET NOT NULL` ×2 and `ALTER COLUMN embedding TYPE VECTOR(768)`. No temp-table dance.
2. **The scoped `--sql` invocation works *before* the merge**, so Step 8's proof does not depend on Step 7.
   OC relaxation: **Step 8 may land before or in parallel with Step 7.**

The emitted SQL also *proves* the defect rather than inferring it: exactly **one** `CREATE TABLE` appears in the
whole revision, and it is `parent_documents` (line 11). The first `ALTER TABLE clauses` is line 35, with no
`CREATE TABLE clauses` anywhere above it.

### The pre-ALTER `clauses` shape, now derived from the generated SQL rather than guessed

The emitted `search_text` definition pins down what the pre-existing table *must* contain:

```sql
ALTER TABLE clauses ADD COLUMN search_text TEXT GENERATED ALWAYS AS
  (COALESCE(clause_type, '') || ' ' || COALESCE(preamble, '') || ' ' || COALESCE(chunk_text, text, '')) STORED;
```

`preamble` and `chunk_text` are added by this same revision, but **`clause_type` and `text` are not** — so the
`CREATE TABLE IF NOT EXISTS clauses` that Step 8 prepends **must** declare `clause_type` and `text`, or the
generated-column DDL fails with `column "text" does not exist`. That is the single constraint that makes the
reconstruction verifiable instead of a guess.

Required columns for the prepended CREATE (= `Clause` at `memory_schema.py:190-249` **minus** the 8 added by
`:64-99`): `id` (PK, uuid), `contract_id`, `doc_id`, `user_id`, `clause_id`, `text`, `clause_type`, `embedding`,
`risk_score`, `decay_score`, `access_count`, `created_at`, `last_accessed_at`.

**Declare `embedding` as `vector(768)`.** The revision later runs `ALTER COLUMN embedding TYPE VECTOR(768)`,
which is then a harmless no-op. Declaring it as bare `vector` would also work but leaves a real type change in
the path; declaring any other dimension would fail on the ALTER because pgvector's typmod is not widenable in
place once rows exist (and would need every index dropped first).

**F6 is CLOSED as a consequence.** F6 worried that the reconstruction might diverge from a real out-of-band
table. `clauses` **does not exist** (A3), so there is nothing to diverge from — whatever Step 8 writes *is* the
shape, and `IF NOT EXISTS` has no pre-existing table to defer to. The reconstruction is authoritative by default.

**One residual, and it is genuinely unresolvable statically:** `CREATE INDEX clauses_bm25_idx ON clauses USING
bm25(search_text)` is emitted (line ~68) and there is **no `bm25` access method on the server yet** (A1) —
`pg_textsearch` is available at 1.3.0 but not installed. The revision's own
`CREATE EXTENSION IF NOT EXISTS pg_textsearch` (line 9) is what installs it, and it precedes the index, so the
ordering is correct. Whether `pg_textsearch` 1.3.0 actually registers an access method **named `bm25`** (as
opposed to a different name, which would make this index DDL wrong) is the one thing neither the `--sql` output
nor `pg_available_extensions` can answer. New Fog F8.

## A8 — Revised ordering constraints (supersede OC0-OC6 where they conflict)

| # | Constraint | Why |
|---|---|---|
| **OC-A** | The concurrent billing split must be **committed and importable** before change 0 begins. Proof: `uv run python -c "import app.main"` and `import app.api.v2` both exit 0. | A2. Every gate value is noise until the tree settles. |
| **OC-B** | Step 0 re-runs from scratch; its numbers replace every baseline figure elsewhere in this file. | A2.1 |
| **OC-C** | Step 8 (clauses CREATE) is **independent of Step 7** — the scoped `--sql` proof works with two heads present (A7). Either order. | A7 |
| **OC-D** | Step 7 (merge) → Step 8b (outbox revision). A new revision needs a single head to attach to. | alembic topology |
| **OC-E** | **Step 8b → Step 12** (or same commit). Fixing `UserIdDep` unmasks `UndefinedTableError` on `outbox_events` on the same endpoint. | A3.1 — this is the plan's only "one fix exposes another break" pair |
| **OC-F** | Step 8 → Step 9. Step 8 reads `Clause` from `memory_schema.py` to build its CREATE; Step 9 moves that class. | A5/A7 |
| **OC-G** | Step 9 → Step 10. Step 10's deletion sweep touches `memory_schema.py`; the harvest must precede it. | OC2, unchanged |
| **OC-H** | Step 17 is fully independent — no inbound, no outbound. Land it first or last. | A6 |
| **OC-I** | OC1 (the `shared/agents/**` cycle) is **unchanged**: that deletion is change 3's, not change 0's. Nothing in the live-DB findings touches it. | OC1 |

**Suggested commit seams** (each independently committable, repo bootable after each):

1. Step 0 (no code) · 2. Step 17 (URL accessor) · 3. Steps 2-6 (deletions) · 4. Step 8 (clauses replay fix) ·
5. Step 7 + Step 8b (merge + outbox tables) · 6. Step 9 (harvest + env.py) · 7. Step 10 (reconciliation) ·
8. Step 11 (`app.state` names) · 9. Step 12 (`UserIdDep`) · 10. Steps 13-15 · 11. Step 16 (gate).

Seam 5 must precede seam 9 (OC-E). No seam leaves the repo unbootable: the only step that changes boot
behaviour is Step 8b, and it *restores* a table rather than removing one.

## A9 — Openspec mapping delta

Two additions to the four new capabilities:

| capability | change | covers | steps |
|---|---|---|---|
| `migration-chain-integrity` | **broaden** | Add a requirement: *the set of tables the migration history claims to have created SHALL match the set that exists, or the divergence SHALL be recorded in the migration history itself.* This is the honest way to spec option D from A4 without pretending the phantoms are fixed. | 7, 8, 8b |
| `database-url-single-source` | **NEW (5th)** | *Every consumer of the database connection string SHALL obtain it from a single accessor that guarantees a usable driver, scheme and credential; no consumer SHALL read the raw configured URL.* Explicitly notes the psycopg-vs-asyncpg split so change 1 inherits the constraint. | 17 |

`transactional-outbox` (an existing capability, and **one of the 6 baseline failures** — D12) is the natural home
for Step 8b, but **do not add a delta to it.** Two reasons: the requirement Step 8b satisfies ("the outbox table
exists") is infrastructure, not behaviour, and adding a delta to an already-red spec makes your own errors
indistinguishable from the pre-existing ones — the same trap already recorded for `typed-exception-handling`.
Step 8b's requirement goes under `migration-chain-integrity` instead. Record the reasoning in `design.md`.

Change 0 therefore ships **5 new capabilities + 1 MODIFIED**, not 4 + 1.

## A10 — Risk register delta

| # | Change |
|---|---|
| **R1** | **RETIRED.** Wrong target (compose vs Timescale Cloud) and the underlying question is answered: `pg_textsearch` 1.3.0 is available. |
| **R2** | **RETIRED as a risk, kept as a Non-Goal.** The compose `timescale` service and its missing `./scripts/init-db.sql` mount are dead config on a service nothing uses. |
| **R4** | **STRENGTHENED.** Editing `9f4a1b7c6d2e` is now provably safe on a *fourth* ground beyond the three recorded: the live DB is stamped past it, so the edit **cannot execute** against production. Its only effect is on replay from base. |
| **R5** | **WEAKENED but kept.** "Step 9 could resurrect dead schema" — the tables do not exist, so nothing is resurrected; only `Base.metadata` gains two entries. The misreading risk ("these are blessed") survives; the DDL risk does not. |
| **R6** | **AMPLIFIED — read with OC-E.** Step 12 does not turn 500 into 401 on its own; without Step 8b it turns `AttributeError` 500 into `UndefinedTableError` 500. The BREAKING flag in `proposal.md` must describe the *pair*. |
| **R10** | **NEW.** *The outbox has been silently dropping every event since the DB was provisioned.* Three writers and a boot-time reader all target a nonexistent table. Once Step 8b lands, the relay's `run_listener` starts working for the first time and `documents`/`auth` events begin actually dispatching to Celery — **into a queue with no worker** (`findings-deployment.md` §1-§3: no worker or beat service in compose, and `Makefile:52` names a nonexistent `celery_config`). Mitigation: Step 8b's `design.md` note must state that enabling the outbox makes a *second* latent defect observable, and that consuming the queue is change 1's (disposition 198.4). Do not let "the outbox now works" be read as "events are now processed". |
| **R11** | **NEW.** *The plan was written against a moving tree.* Any line/file reference in this document may have shifted under the billing split. Mitigation: treat every `file:line` here as needing re-confirmation via `rg` before editing, and prefer symbol names over line numbers in `tasks.md`. |

## A11 — Fog delta

### Closed by this addendum

| # | Was | Now |
|---|---|---|
| **F1** | Can the repaired chain run on an empty DB (does the image ship `pg_textsearch`)? | **CLOSED — favourable.** `pg_textsearch` available 1.3.0 on the actual server; the image was the wrong target. |
| **F2** | Are `parent_documents`/`clauses`/`entities`/`relationships`/`events`/`memory_versions` in the deployed DB? | **CLOSED.** None of them. Nor `documents`, `chunks`, `search_documents`, `search_chunks`, `chat_messages`, `chat_sessions`, `document_vectors`, `outbox_events`, `dead_letter_events`. Only the 15 billing tables + `alembic_version`. |
| **F6** | Is the reconstructed pre-ALTER `clauses` shape right? | **CLOSED.** No table exists to diverge from, and the emitted `search_text` generated-column DDL pins the two required pre-existing columns (`clause_type`, `text`). Authoritative by default (A7). |
| **F7** | Does `batch_alter_table` emit usable offline SQL? | **CLOSED — favourable.** Plain `ADD COLUMN` statements, no `recreate=` needed, and the scoped `--sql` invocation works before the merge (A7). |
| — | Does `get_database_url()` rewrite `postgres://`? (assigned open question) | **CLOSED.** Yes — `:42-47`, plus password injection at `:56-70` and libpq-param stripping at `:51-54`. No scheme defect (A6). |

### Still open

**F3, F4, F5 — unchanged.** None is touched by the DB findings. F5 (`spec/typed-exception-handling`'s
existing failure) is now *more* load-bearing, since A9 adds a second reason not to touch red specs.

**F8 — NEW. Does `pg_textsearch` 1.3.0 register an access method literally named `bm25`?**
`9f4a1b7c6d2e` emits `CREATE INDEX clauses_bm25_idx ON clauses USING bm25(search_text) WITH (text_config=…, k1=…, b=…)`,
and `search/repository.py:415-419` calls `to_bm25query(:query, 'search_chunks_bm25_idx')`. The server currently has
**no `bm25` access method and 0 `pg_proc` rows for `to_bm25query`** — expected, since the extension is not
installed. If 1.3.0's access method or function names differ from what this code assumes, D5.1's entire "keep the
existing BM25 implementation" premise breaks and change 1 inherits a rewrite. This is the last unverified
precondition under D5.1.
```bash
# against the cloud DB; the extension install is idempotent and 9f4a1b7c6d2e does it anyway
psql "$PG_ADMIN_DSN" -c "CREATE EXTENSION IF NOT EXISTS pg_textsearch;" \
  -c "SELECT amname FROM pg_am WHERE amname='bm25';" \
  -c "SELECT proname FROM pg_proc WHERE proname='to_bm25query';"
```

**F9 — NEW. Are `0003` and `0004`'s ALTERs actually present on the 15 billing tables, or were they stamped too?**
A3 establishes that `0002` genuinely ran (all 15 tables exist) and that everything before `0001` did not. It does
**not** establish that `0003`/`0004` ran — the tables would exist either way. If they were stamped, change 0's
merge sits on top of a second, subtler divergence, and the billing feature has columns its code expects and the
DB lacks. This matters because the concurrent billing split (A2) is actively shipping against those tables.
```bash
uv run python -c "
import asyncio, asyncpg
from app.connections.postgres import get_database_url
async def main():
    c = await asyncpg.connect(get_database_url().replace('postgresql+asyncpg://','postgresql://'))
    # pick one column each that 0003 and 0004 add, then:
    print(await c.fetch(\"select table_name, column_name from information_schema.columns where table_name in ('plans','subscriptions') order by 1,2\"))
    await c.close()
asyncio.run(main())"
# then diff against the op.add_column calls in 0003_*.py and 0004_*.py
```

**F10 — NEW. Does `OutboxRelay.run_listener()` surface the `UndefinedTableError`, or swallow it?**
`lifespan.py:131` launches it with `asyncio.create_task` and nobody awaits the result, so the exception may be
lost entirely (Python logs "Task exception was never retrieved" only at GC). This decides whether the outbox
break is currently *invisible* or merely *ignored*, which changes how Step 8b's before/after evidence is framed.
```bash
uv run python -c "import app.main" 2>&1 | rg -i "outbox|relation|does not exist|Task exception"
# and: rg -n "except|logger" src/app/shared/outbox/relay.py | sed -n '1,40p'
```

**F11 — NEW. Was `alembic stamp` run deliberately, and by what?** No script under `src/` or `scripts/` performs a
stamp, yet the DB is stamped at `0004` with `0001` unapplied. If a deploy pipeline stamps on every release, change
0's merge revision will be stamped rather than applied on the next deploy and Step 8b's tables will never be
created in production.
```bash
rg -n "alembic (stamp|upgrade)" --glob '!**/.venv/**' . ; ls -la .github/workflows/ Makefile
```

## A12 — Steps with ZERO test coverage (revised full list)

`rg -l "alembic|get_database_url|outbox_events|app\.state\.storage" tests/` → **no files.** So:

| Step | Coverage | Non-test evidence standing in |
|---|---|---|
| 7 (merge) | none | `alembic heads` → 1 line; `--sql` exit 0 not 255 |
| 8 (clauses) | none | scoped `--sql` line-ordering assertion (A7) — a real assertion on generated DDL, no DB needed |
| 8b (outbox) | none | `to_regclass` → non-`None` for both tables; `--sql | grep -c` → 1 |
| 9 (harvest + env.py) | none | `Base.metadata.tables` membership listing |
| 11 (`app.state` names) | none | AST `PROOF-STATE` walk (see the Steps section's trap note: regex gives both false positives and false negatives here) |
| 17 (URL accessor) | none | `rg` emptiness on raw `settings.POSTGRES_URL`; `urlparse(get_database_url())` scheme + password assertion |

Only Steps 2-6, 10, 13 have any adjacent test signal, and `baseline-tests.md`'s warning governs all of it:
green after a deletion *"means nothing was checked, not that nothing broke."* **Compare the pytest summary line,
never `$?`** — `--cov-fail-under=80` against 18.38% coverage makes a fully green suite exit 1.

**End of addendum.**
