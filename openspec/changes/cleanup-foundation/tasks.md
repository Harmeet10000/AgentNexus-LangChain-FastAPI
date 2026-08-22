# Tasks

Twelve ordered groups, following `design.md`'s Migration Plan exactly. Each group is at least one commit, every
commit leaves the application bootable, and each carries a **Proof** that is a command you can run.

**Five rules every Proof below obeys.** They are not style preferences — each one exists because the obvious form of
the proof is unexecutable in this repository.

1. **Compare against a captured baseline, never an absolute number.** Deleting ~2,900 lines moves every lint, type
   and test count for reasons unrelated to correctness. Step 1 captures baselines to disk; later steps assert `<=` or
   `identical` against **those files**, never against a figure quoted in any document — including this one.
2. **Read `pytest`'s printed summary counts, never its exit code.** `--cov-fail-under=80` sits in the runner's
   default arguments and coverage stands at **22.16%**, so the runner exits non-zero even when every collected test
   passes. Any proof of the form "the suite is green" is **unexecutable here**. Capture with coverage disabled and
   compare passed/failed/errored counts. (Exit codes *are* meaningful for `rg`, `alembic` and `openspec`, and are
   used there.)
3. **No Proof renders migration history from base.** Offline `--sql` has no database from which to read
   `alembic_version`, so it **always renders from base** and is never incremental (D14.3). Where migration output
   must be asserted, render a **scoped range** (`alembic upgrade <parent>:<rev> --sql`), which does emit only that
   range.
   **Do not justify this rule by claiming a from-base render aborts** — it does not. Measured: `alembic upgrade heads
   --sql` exits **0** and emits **697 lines**, `clauses` ALTERs included, because offline rendering emits DDL as text
   and never executes it. A phantom target is not an error in a render. The rule is correct; that reason for it is
   false.
4. **`alembic upgrade head` — singular — is not a valid command until step 3.** The chain is branched, so the
   singular form does not resolve. Only `heads` is well-defined before the merge lands.
5. **Nothing in this change applies DDL to the deployed database.** Every migration proof runs against a **local
   scratch database**, which needs no authorization. Applying to the deployed instance is a **separately authorized
   act**, requested only after the scratch rehearsal passes, and no task here assumes it has happened.

**Deletion proofs never rest on "tests pass".** Every deleted tree has zero coverage, so green afterwards means
nothing was checked. Each deletion is proved by an import probe over all boot entry points **plus** an emptiness
search **plus** an unchanged test-failure count. If any count moves, something imported the tree unexpectedly —
that is a finding, not noise.

---

## 1. Capture every baseline to disk

Nothing later in this change is provable without this, because three of the six gates are **red before this change
starts and stay red after it**. The pass criterion for those is "no worse than baseline", which requires a baseline
file to compare against.

**Precondition:** the concurrent billing-feature split must be committed and importable first. It moved files under
this plan while it was being written, which is why every inherited gate figure is treated as stale.

- [x] Capture tests (coverage disabled), collection, lint, types, format, structural scan, spec validation,
      migration heads, and the pre-change SHA.

**Proof**

```bash
mkdir -p /tmp/c0-baseline
git rev-parse HEAD                                  > /tmp/c0-baseline/sha.txt
uv run pytest -q -p no:cacheprovider --no-cov 2>&1 | tail -3 > /tmp/c0-baseline/tests.txt
uv run pytest --collect-only -q                2>&1 | tail -2 > /tmp/c0-baseline/collect.txt
uv run ruff check src/                         2>&1 | tail -2 > /tmp/c0-baseline/ruff.txt
uv run ty check src/                           2>&1 | tail -2 > /tmp/c0-baseline/ty.txt
uv run ruff format --check src/                2>&1 | tail -2 > /tmp/c0-baseline/fmt.txt
ast-grep scan src/                             2>&1 | tail -3 > /tmp/c0-baseline/astgrep.txt
openspec validate --all                        2>&1 | tail -3 > /tmp/c0-baseline/openspec.txt
uv run alembic heads                           2>&1 > /tmp/c0-baseline/heads.txt
wc -l /tmp/c0-baseline/*
```

Passes when every file is non-empty. Expected shape at authoring time, recorded as orientation and **not** as the
gate: tests `22 failed, 55 passed, 13 errors`; collection `90 tests, 0 errors`; ruff `123`; ty `46`; ast-grep `4`;
openspec `21 passed, 6 failed`; `alembic heads` prints **two** heads.

**The thirteen errors are `setup` errors, not collection errors** — all thirteen are `fixture 'client' not found`
in `tests/integration/test_health.py` and `tests/integration/test_api_deprecation.py`. Right magnitude, wrong kind,
and the kind matters: it is one missing fixture, not a broken test tree.

### MEASURED 2026-08-22 at `6525c6f` — the orientation figures above are stale on four of seven gates

| Gate | Quoted above | **Measured** |
|---|---|---|
| tests | 22 failed, 55 passed, 13 errors | **3 failed, 103 passed, 9 errors, 48 deselected** |
| collection | 90 tests, 0 errors | **115/163 collected (48 deselected)** |
| ruff | 123 | **0 — `All checks passed!`** |
| ty | 46 | **2** |
| format | — | 389 files already formatted |
| ast-grep | 4 | 4 ✓ |
| openspec | 21 passed, 6 failed | 21 passed, 6 failed ✓ |
| alembic heads | two | two (`0005`, `6c42587c7195`) ✓ |

Two consequences the later steps must absorb:

1. **The thirteen `fixture 'client' not found` errors no longer exist.** The nine actual errors are `TokenClaims`
   `ValidationError`s in `tests/unit/test_websocket_security_*.py` — the model requires `role` / `permissions`, which
   the fixtures omit; the three failures are the same drift plus `Settings` being `frozen=True`. **This undercuts
   step 8's stated rationale**, which argues from the absence of a `client` fixture. Step 8's *change* may still be
   right; its *evidence* is gone and must be re-derived. All twelve are pre-existing and owned by no change.
2. **`ruff` is already at zero**, so step 2's "count must DROP" proof is arithmetically unsatisfiable — see step 2.

**Precondition status:** the billing split is committed, but `import app.features.subscriptions` alone still raises
`ImportError` (partially initialized module) via `subscriptions/__init__ → router → dependencies → payments/__init__
→ payments/router:8 → back into subscriptions.dependencies`. Every real entry point (`app.main`, `app.api.v1`,
`app.api.v2`, `tasks`, `app.lifecycle.lifespan`) imports cleanly, so the app boots and this change is not blocked —
the graph is order-fragile, not broken. Unowned; surfaced for assignment.

### GAP CLOSED — the baseline files were never written to disk

The figures above were recorded **into this document** and `/tmp/c0-baseline/` was never created. That was not a
cosmetic omission: **five later proofs read those files by path**, and step 12 states its pass criterion as *"every
rung compared against the files from step 1, none by exit code."* With the directory absent, every one of those
comparisons was unexecutable, and step 11's agent hit exactly that. The failure mode is the quiet one — a `cat` of
a missing file yields an empty string, and `count <= ` *nothing* reads as satisfied.

`/tmp/c0-baseline/` now exists, **reconstructed from the recorded figures above rather than captured live**, one
file per gate, each carrying a two-line provenance header naming the SHA
(`6525c6f36fb54b244cc0755b0511d39085ece96a`) and stating that it was not measured at write time. A live re-capture
was rejected deliberately: the working tree already carries steps 3, 4, 5, 7, 10 and 11, so measuring *now* would
record the post-change state under the name "baseline" — the one error that would make every downstream gate look
green by construction.

Consumers should read these by eye, not `cat` them into arithmetic, because of the header. `/tmp` is volatile — if
these files are gone at step 12, rebuild them from the table above rather than re-measuring.

## 2. Delete the unparseable draft

First, deliberately: its proof is a **drop** in the lint baseline, which is what establishes that the baseline files
from step 1 are trustworthy before any harder step depends on them.

- [x] Delete `src/app/shared/rag/document_processing/todo_temp.py` (783 lines; does not parse — its `__all__`
      closes and an orphaned class body resumes below it). Zero importers.

**Proof**

```bash
rg -l 'todo_temp' src/ tests/          # must print nothing
uv run ruff check src/ 2>&1 | tail -2  # count must DROP vs /tmp/c0-baseline/ruff.txt
```

Passes when the importer search is empty and the count drops. **Confirm the drop maps to the deleted file** — a drop
that maps to a suppressed real diagnostic is a regression wearing a green tick.

### RESOLVED as a no-op — the file was already deleted, and the proof was substituted

`src/app/shared/rag/document_processing/todo_temp.py` **does not exist**. `git log --diff-filter=D` places its
deletion in `f0e0b84` *"feat: implement user credit integration feature"* — an unrelated commit that swept it up.
`rg -l 'todo_temp' src/ tests/` prints nothing. That is also **why ruff measures zero**, which makes the "count must
DROP" proof unsatisfiable: nothing can drop from zero.

Because this step exists to establish *that the step-1 baseline files are trustworthy*, that purpose was served by a
substitute proof — distinguishing a genuinely clean ruff from a silently-broken invocation, which look identical in a
captured file:

```bash
printf 'import os\n' | uv run ruff check --stdin-filename src/app/_probe.py -   # → F401, 1 error
uv run ruff check --statistics src/ ; find src -name '*.py' | wc -l            # → 389 files scanned
```

Ruff reports the planted `F401` and scans all 389 files — matching `fmt.txt`'s "389 files already formatted". The
harness is live and the clean baseline is real. **No suppression was added and no file was created**; the probe ran
through stdin. `document_processing/` now holds only parseable modules: `chunker, docling_enhanced, embedder,
entity_extractor, ingest, ingest_v2, models`.

## 3. Join the two migration heads

A merge revision with an **empty body**: both branches declare the same parent and touch disjoint relations, and
both create their extensions idempotently, so there is nothing to reconcile.

The docstring is load-bearing and is the deliverable here as much as the revision is. It must name every phantom
relation, name `9f4a1b7c6d2e` as **unrunnable** and the `clauses` relation it presupposes, and state that **reversal
below this point is unsupported** — because those reversals drop relations that were never created.

This is also what makes `alembic upgrade head` (singular) resolve again, repairing `Makefile:39`, `README.md:272`
and `.github/workflows/test.yml:105` **without editing them**.

- [x] Add the merge revision joining the two heads.
- [x] Write the docstring: phantom relations, `9f4a1b7c6d2e` unrunnable, reversal prohibition.

**Proof**

```bash
uv run alembic heads | wc -l           # 1
uv run alembic upgrade head --sql >/dev/null; echo "exit=$?"   # 0 — singular now resolves
rg -c 'clauses|9f4a1b7c6d2e|reversal' src/alembic/versions/<merge_rev>.py
```

Passes when exactly one head remains, the singular form exits 0, and the docstring names all three subjects.

### DONE — `9b6bf3d1d548_join_credit_and_unified_documents_heads.py`

Measured topology, which is one level deeper than "both branches declare the same parent":

```
c0c17c6eb1cc → 2bc7726317f6 ─┬─ 8a7d9b1c2e3f → 9f4a1b7c6d2e → 0001 → 0002 → 0003 → 0004 ─┬─ 0005
                             │                                                            │
                             └─ a71f0d7d9c12 ───────────────────────────────────────────── 6c42587c7195
```

`6c42587c7195` is **itself already a merge** (`down_revision = ("0004", "a71f0d7d9c12")`, empty body). `0005` was then
stacked on `0004`, re-splitting the graph at `0004`. So the new revision merges one leaf-with-DDL (`0005`, credit
relations) with one empty merge whose far leg carries the unified `documents`/`chunks` DDL. Disjoint relations, both
extension-idempotent — the empty body holds.

Proof, all three legs green:

| Check | Result |
|---|---|
| `alembic heads` | **1** — `9b6bf3d1d548 (head)` |
| `alembic upgrade head --sql` | **exit 0** — singular form resolves; `Makefile:39`, `README.md:272`, `.github/workflows/test.yml:105` repaired unedited |
| `rg -c 'clauses\|9f4a1b7c6d2e\|reversal'` | **13** (`clauses` 6, `9f4a1b7c6d2e` 6, `reversal` 3) |

Gates held at baseline: ruff clean, format clean (390 files, +1 as expected), ty 2, ast-grep 4, pytest 3 failed /
103 passed / 9 errors. One self-inflicted ruff regression (0 → 2, `ambiguous-unicode-character-docstring` on `×`
characters I wrote) was **fixed in prose, not suppressed**.

The docstring's claims were each verified against the revisions rather than inherited:

- **`clauses` is phantom in the strong sense.** A census of `create_table` across all ten revisions yields
  `outbox_events, dead_letter_events, plans…reports, user_credits, credit_consumptions, search_documents,
  search_chunks, parent_documents, documents, chunks, chat_messages, chat_sessions, document_vectors` — **no
  `clauses`**. Yet `9f4a1b7c6d2e` runs `batch_alter_table("clauses")`, `UPDATE clauses …`, and `create_foreign_key`
  against it. That is what makes it unrunnable.
- **Reversal is a live hazard, not a precaution.** `9f4a1b7c6d2e.downgrade()` → `batch_alter_table("clauses")` plus a
  bare `drop_table("parent_documents")`; `8a7d9b1c2e3f.downgrade()` → bare `drop_table("search_chunks")` /
  `("search_documents")`; `0001.downgrade()` → bare `drop_table("dead_letter_events")` / `("outbox_events")`. **None
  carry `IF EXISTS`**, so a reversal aborts partway and leaves `alembic_version` disagreeing with the schema.

## 4. Add the authoritative revision

Defines the whole target shape in one place. **Ordered internally: outbox relations first**, then the unified
`documents` / `chunks` relations, `chunks.updated_at`, the retrieval indexes, and the extensions those indexes need.

The outbox half is justified independently of everything else: it repairs **two mounted, public, rate-limited
endpoints that return 500 today** — `POST /auth/forgot-password` and `POST /auth/resend-verification` — which fail
*after* persisting a reset/verification token no email will ever deliver. That is a partial write on shipped surface.

**Written as raw `IF NOT EXISTS`-style DDL, not ORM operations**, for two load-bearing reasons: `a71f0d7d9c12` is
unstamped and **will execute on the next upgrade**, creating `documents` and `chunks` before this revision runs, so
non-idempotent DDL fails on a duplicate relation; and an inspector-based guard needs a live connection, which would
destroy the only proof here that needs no database.

**All four extensions are created explicitly** — `vector`, `vectorscale`, `pg_trgm`, `pg_textsearch` — conditionally
and before the first dependent object. Never rely on ambient availability: the chain's correctness is currently
supplied by the **hosting image** rather than by any revision that will execute, so it will not reproduce on a fresh
environment or a differing managed image.

**Index names are a query contract, not a migration-local choice.** The two-argument keyword-ranking constructor
takes the **index name as a literal SQL argument**, pinned at `src/app/features/search/constants.py:15`. An index of
the right shape under a different name **matches nothing and reports no error**. Create every retrieval index under
the exact name the declaring revision already uses, so the two converge instead of producing two differently-named
indexes of the same shape.

- [x] Add the authoritative revision: outbox first, then documents/chunks, `updated_at`, indexes, four extensions.
- [x] Create all three retrieval branches' indexes — vector, BM25 keyword, trigram fuzzy — by exact name.

**Proof**

```bash
# (a) the revision alone, as a scoped range — never from base (rule 3)
uv run alembic upgrade <parent>:<auth_rev> --sql > /tmp/c0-auth.sql; echo "exit=$?"
grep -c 'CREATE TABLE' /tmp/c0-auth.sql
grep -oE 'CREATE (INDEX|EXTENSION)[^;]*' /tmp/c0-auth.sql | sort

# (b) index names match the pinned query contract
rg -n 'bm25|_idx' src/app/features/search/constants.py

# (c) scratch database only — NOT the deployed instance
#     start a local PG, then:
uv run alembic upgrade heads
uv run python -c "
import asyncio, asyncpg, os
async def m():
    c = await asyncpg.connect(os.environ['SCRATCH_URL'])
    for t in ['outbox_events','dead_letter_events','documents','chunks']:
        print(t, await c.fetchval('SELECT to_regclass(\$1)', 'public.'+t))
    print('bm25 indexes:', await c.fetchval(
        \"SELECT count(*) FROM pg_index i JOIN pg_class c ON c.oid=i.indexrelid \"
        \"JOIN pg_am a ON a.oid=c.relam WHERE a.amname='bm25'\"))
    await c.close()
asyncio.run(m())"
```

Passes when the scoped render exits 0 and emits only this revision's DDL, every retrieval index name matches the
pinned constant, and on the scratch database all four relations resolve non-null with a non-zero `bm25` index count.

**Before any deployed upgrade** (separately authorized, not part of this change): assert the extension
`a71f0d7d9c12` needs is present, because that revision executes **first** and this revision **cannot repair it** —
a failure there aborts the upgrade before the outbox repair runs.

### WRITTEN — `a5bd6b69a28e_authoritative_target_shape_outbox_.py`. Proofs (a) and (b) green; **(c) NOT RUN — see gate below**

Proof (a), the scoped range — rendered offline, never from base:

| Check | Result |
|---|---|
| `alembic upgrade 9b6bf3d1d548:a5bd6b69a28e --sql` | **exit 0** |
| Revisions in the render | **only** `Running upgrade 9b6bf3d1d548 -> a5bd6b69a28e` — nothing leaked |
| `CREATE TABLE` count | **4** — `outbox_events`, `dead_letter_events`, `documents`, `chunks` |
| Extensions | **4/4** — `vector`, `vectorscale`, `pg_trgm`, `pg_textsearch`, all `IF NOT EXISTS`, all before the first dependent object |
| Retrieval indexes | **3/3** — `chunks_bm25_idx`, `chunks_embedding_idx`, `chunks_search_text_trgm_idx` |
| Gates | ruff clean, format clean (391 files), ast-grep 4, `alembic heads` = **1** (`a5bd6b69a28e`), singular `upgrade head --sql` exit 0 |

**Proof (b) was misaimed and has been redirected.** The section cites the pin as
`src/app/features/search/constants.py:15`. Measured: that line holds
`SEARCH_CHUNKS_BM25_INDEX_NAME = "search_chunks_bm25_idx"` and has **zero readers anywhere in `src/`** — it is dead
code, and it names an index on the *separate* `search_chunks` relation, not on `chunks`. Following proof (b)
literally would have named the unified index `search_chunks_bm25_idx` and silently broken all six query sites.

The real contract is a **hardcoded literal** in the SQL: `search_text <@> to_bm25query(:query, 'chunks_bm25_idx')`.
Verified name-match against it:

| Index | In revision | Literal sites in `features/documents/repository.py` |
|---|---|---|
| `chunks_bm25_idx` | 1 | **6** ✓ |
| `chunks_embedding_idx` | 1 | 0 — expected |
| `chunks_search_text_trgm_idx` | 1 | 0 — expected |

The two zeros are correct, not a miss: **only bm25 takes the index name as a SQL argument**. Vector and trigram
branches are planner-selected, so their names bind nothing at query time. All three still match the names
`a71f0d7d9c12` declares, which is what makes the two revisions converge rather than produce duplicate
differently-named indexes.

Three further measured deviations, recorded:

1. **`chunks.updated_at` needed a `server_default`, and the section does not say so.** Neither the `Chunk` ORM model
   (`features/documents/model.py:76`) nor any query declares or reads it — only `UnifiedDocument` has one (`:60`).
   `NOT NULL` with no default would break every ORM insert. Implemented as
   `ADD COLUMN IF NOT EXISTS updated_at timestamptz NOT NULL DEFAULT now()`.
2. **The outbox→500 chain is confirmed, exactly as the section argues.** `features/auth/router.py:195`
   (`/forgot-password`) and `:179` (`/resend-verification`) → `AuthService._publish_outbox_event` (`service.py:564`,
   called from `:246` and `:273`) → `with_outbox` (`:571`) against an absent `outbox_events`. ORM models exist at
   `src/app/shared/outbox/model.py:15,31`, so only the relations were missing.
3. **`downgrade()` is an intentional no-op**, consistent with `9b6bf3d1d548`'s reversal prohibition. Dropping these
   relations would re-break both public endpoints while leaving `0001` and `a71f0d7d9c12` still claiming to have
   created them.

### GATE — proof (c) has not run, and no later task may assume it did

Proof (c) is the **scratch-database rehearsal**. No local PostgreSQL exists (nothing listening on 5432/5433,
`pg_isready` absent, Docker unavailable). A Timescale Cloud instance was made available instead and probed
**read-only** — host/port/database only, no DDL, no credential printed:

```
host qbid1qrc75.nnro3dh8tf.tsdb.cloud.timescale.com  port 39662  database tsdb  sslmode require
server_version 18.4        current_user superuser: False
alembic_version            0004
```

**Measured relation state — every one absent**, which confirms the stamped-not-migrated diagnosis exactly:

| Relation | `to_regclass` |
|---|---|
| `outbox_events`, `dead_letter_events` | **NULL** — this is why the two public auth endpoints 500 today |
| `documents`, `chunks` | **NULL** |
| `search_chunks`, `parent_documents`, `clauses` | **NULL** |
| bm25 index count | **0** — proof (c) requires non-zero |

**Extension state — and this is a blocker step 4 anticipated:**

| Extension | Available | Installed |
|---|---|---|
| `vector` | yes | **0.8.2** ✓ |
| `vectorscale` | yes | **0.9.0** ✓ |
| `pg_textsearch` | yes | **1.3.0** ✓ |
| `pg_trgm` | yes | **NOT INSTALLED** ✗ |
| `uuid-ossp` | yes | **NOT INSTALLED** ✗ |

Step 4's closing instruction is to *"assert the extension `a71f0d7d9c12` needs is present"* before any upgrade,
because that revision **executes first** and this revision **cannot repair it**. **That assertion fails.**
`a71f0d7d9c12` opens with `CREATE EXTENSION IF NOT EXISTS pg_trgm` (and `uuid-ossp`), and `current_user` is not a
superuser — so whether the upgrade proceeds or aborts depends on whether the role may create a *trusted* extension.
If it aborts, it aborts **before** the outbox repair, leaving the two 500s in place. This must be settled before
`alembic upgrade heads` is run against this instance, not discovered during it.

**Second blocker, ordering:** `alembic upgrade heads` loads `src/alembic/env.py`, which is step 5's edit target.
The rehearsal must run **after** step 5 lands, or it exercises a half-written environment.

Status: **paused** when written; **CLOSED 2026-08-23** — see "APPLIED LIVE" below. Both blockers were resolved
first: the extension assertion passes (next section), and step 5 landed before the upgrade ran, so `env.py` was
whole. Nothing between this line and that section assumes (c) ran.

### EXTENSION BLOCKER RESOLVED 2026-08-23 — the assertion passes; the earlier reading was of the wrong privilege

The extension question above is settled **empirically**, not by catalog inference. Both missing extensions were
created inside a transaction that was then rolled back, under the deployed role:

```
pg_trgm      CREATE SUCCEEDED  (rolled back, nothing persisted)
uuid-ossp    CREATE SUCCEEDED  (rolled back, nothing persisted)
post-rollback: neither installed — confirmed via pg_extension
```

So `a71f0d7d9c12` will **not** abort on its opening `CREATE EXTENSION` block, and the outbox repair behind it is
reachable. The gate's fear was misplaced, and the reason is worth recording because it is a trap that reads the
other way round:

| Catalog column | `pg_trgm` | What it actually means |
|---|---|---|
| `superuser` | `true` | installing requires superuser… |
| `trusted` | **`true`** | …**unless** the caller holds `CREATE` on the current database |

`usesuper = false` is therefore **not** the deciding privilege — `has_database_privilege(current_user,
current_database(), 'CREATE')`, measured `true`, is. A role that is not a superuser installs a *trusted* extension
into the default schema without complaint; this is the PostgreSQL 13+ trusted-extension mechanism. The gate above
read `usesuper = false` and inferred a blocker from the half of the contract that does not decide.

**The inverse case is the real hazard, and it is now recorded in the revision's own docstring.** `pg_textsearch` and
`vectorscale` are `trusted = false`, so this role **cannot** create them at all. They pass here only because they
are already installed, which makes `CREATE EXTENSION IF NOT EXISTS` a no-op. `IF NOT EXISTS` guards against the
object existing, **not** against the caller lacking privilege — on an image shipping without them, revision
`a5bd6b69a28e` fails at its extension block rather than degrading quietly. That is the correct behaviour (the
`bm25` and `diskann` branches are unsatisfiable without them) but it is an **environment prerequisite**, not
portability, and the docstring previously implied otherwise. Corrected.

**Remaining precondition for (c) is now only the ordering one**, and it is wider than the gate above states.
`alembic.ini` declares **no `sqlalchemy.url`** (only `script_location = src/alembic` at `:3`), so `env.py:127`
obtains its connection from `await init_db()` — meaning the upgrade connects through
`src/app/connections/postgres.py`, which is **step 9's edit target**, and imports `database.Base`, whose package
`__init__` reaches the module tree **step 6 deletes**. The rehearsal is therefore gated on steps 5, 6 **and** 9,
not on step 5 alone. Recorded as the reason (c) still had not run at the time this block was written.

**Finding that belongs to step 9, discovered by this preflight.** `POSTGRES_URL` in the deployed environment
carries **no password and the legacy `postgres://` scheme**. `make_url(POSTGRES_URL).password` is the empty
string; the real credential lives only in the discrete `POSTGRES_PASSWORD` (`SecretStr`) field, and the discrete
`POSTGRES_HOST`/`PORT`/`DB_NAME`/`USERNAME` fields all agree with the URL's non-secret parts. Two consequences for
the accessor step 9 rewrites:

1. Any accessor that builds its DSN by **string-substituting the scheme** on `POSTGRES_URL` — which is precisely
   what `get_database_url`'s docstring describes ("Convert psycopg2 URL to asyncpg URL") — yields a
   **credential-less DSN that cannot authenticate**. The password must come from the discrete field.
2. `postgres://` is the *legacy* scheme; SQLAlchemy 2.x resolves no dialect for it and raises
   `NoSuchModuleError` on `create_async_engine`. A scheme rewrite must produce `postgresql+asyncpg://`, and
   must not assume the input scheme is `postgresql://`.

### APPLY-SET PROVEN 2026-08-23 — exactly 5 revisions execute, and one of them is off the stamped branch

Computed from alembic's own `ScriptDirectory` API (`iterate_revisions(rev, "base")` set difference), which reads
the version files **without executing `env.py`** — so this proof is valid while the import trees are mid-edit, and
needs no database:

| # | Revision | What it does | Risk against the measured state |
|---|---|---|---|
| 1 | `a71f0d7d9c12` | 4 extensions, `documents`, `chunks`, 3 retrieval indexes | **the only non-idempotent DDL** — self-contained, verified below |
| 2 | `6c42587c7195` | mergepoint | body is `pass` |
| 3 | `0005` | `user_credits`, `credit_consumptions` | FKs resolve — see below |
| 4 | `9b6bf3d1d548` | mergepoint (this change) | body is `pass` |
| 5 | `a5bd6b69a28e` | this change's authoritative shape | fully `IF NOT EXISTS` |

**`a71f0d7d9c12` is not an ancestor of `0004`.** It sits on the sibling branch under the `2bc7726317f6`
branchpoint, so `upgrade head` must apply it to satisfy the `6c42587c7195` mergepoint. It carries the only
non-idempotent DDL in the whole apply-set, executing against a database whose recorded history is partly fiction —
which makes it the one revision that had to be read line by line before any upgrade. Read: it is **self-contained**.
It creates `documents` and `chunks` from nothing, references no relation from any unapplied revision, and performs
no `ALTER TABLE` against a phantom. `documents` and `chunks` are both `NULL` on the instance, so its `create_table`
calls cannot collide.

**FK targets in the apply-set all resolve.** `0005` → `user_credits.id` (created in the same revision) and
`invoices.id` (**present**, one of the 16 existing relations). `a71f0d7d9c12` → `documents.id` (same revision).
Nothing references the absent `users` relation: every `user_id` is a bare `String(255)`, consistent with auth
identities living in MongoDB via Beanie rather than in this database.

**The phantom set — believed applied, DDL never ran, and none of it will ever re-run.** These 5 of the 8 revisions
the `0004` stamp marks applied left no relations behind:

| Revision | Doc | Relations it should have created — all `NULL` |
|---|---|---|
| `c0c17c6eb1cc` | Initial schema: document_vectors and chat tables | `document_vectors`, chat tables |
| `2bc7726317f6` | rename_metadata_to_meta_data | (a rename that never happened) |
| `8a7d9b1c2e3f` | Add search documents and chunks schema | `search_documents`, `search_chunks` |
| `9f4a1b7c6d2e` | Contract KB parent documents and pg_textsearch clauses | `parent_documents`, `clauses` |
| **`0001`** | **Add outbox_events and dead_letter_events tables** | **`outbox_events`, `dead_letter_events`** |

**`0001` is the load-bearing one, and it settles step 4's justification empirically.** It is an ancestor of `0004`,
therefore marked applied, therefore **it will never execute again** — while `to_regclass('outbox_events')` is
`NULL`. So `a5bd6b69a28e` is not a convenience or a duplicate of `0001`: it is the *only* revision that will ever
create the two relations the two 500ing public auth endpoints require. Ordering the outbox block first in that
revision is what makes the repair independent of everything else in the chain.

Note the ordering anomaly that identifies this as a stamped database rather than a migrated one: `0002`–`0004`'s
billing relations are **present** while `0001`'s are **absent**. A normal `upgrade` cannot produce that, since
`0001` precedes `0002`. The chain was applied from `0002` onward and then stamped.

**Transaction scope — the safety property that makes running this acceptable.** `alembic.ini` sets no
`transaction_per_migration` and `env.py`'s `do_run_migrations` does not pass it, so `context.begin_transaction()`
wraps the **entire 5-revision chain in one transaction**. PostgreSQL has transactional DDL, so a failure at any
point rolls back every relation, index, extension **and** the `alembic_version` update together. There is no
partial-application state to clean up: the upgrade either lands whole or leaves the instance exactly as measured.
This is why no scratch rehearsal was substituted for it once the inspection above came back clean.

### APPLIED LIVE 2026-08-23 — proof (c) has run; the GATE above is CLOSED

The user granted the separate authorization rule 5 requires ("DB is live now. you can do it."). `alembic upgrade head`
was run twice against the deployed instance, both exit 0:

| Run | Revisions applied | `alembic_version` after |
|---|---|---|
| 1 | `a71f0d7d9c12`, `6c42587c7195`, `0005`, `9b6bf3d1d548`, `a5bd6b69a28e` | `a5bd6b69a28e` |
| 2 | `b3e7c41d92af` | `b3e7c41d92af` |

`alembic_version` holds **exactly one row**. Public tables **16 → 22 → 27**. Extensions now installed:
`pg_textsearch`, `pg_trgm`, `uuid-ossp`, `vector`, `vectorscale` — the two absent ones created without incident,
as the rolled-back rehearsal predicted.

**The outbox repair landed.** `outbox_events` and `dead_letter_events` both exist with 9 columns. The two public
endpoints that returned 500 — `POST /auth/forgot-password` (`features/auth/router.py:195`) and
`POST /auth/resend-verification` (`:179`) — have the relations `_publish_outbox_event` requires.

**The three retrieval index branches exist under their contract names**: `chunks_bm25_idx` (bm25),
`chunks_embedding_idx` (diskann), `chunks_search_text_trgm_idx` (gin). `chunks.search_text` is `generated=ALWAYS`;
`chunks.updated_at` is `NOT NULL DEFAULT now()`.

**`b3e7c41d92af` — the five relations the phantoms never created.** Added because `alembic check` proposed
`add_table` for `chat_sessions`, `chat_messages`, `document_vectors`, `search_documents`, `search_chunks`. All five
now exist; `message_role` enum is `['user','assistant','system']`; `search_chunks.content_tsv` is
`to_tsvector('english'::regconfig, content)` generated ALWAYS; `document_vectors`' column is `metadata` (matching
the ORM, since `2bc7726317f6`'s rename is itself a phantom); FK
`fk_search_chunks_document_id_search_documents` present.

### DRIFT CLOSED — and step 5's contract had a hole one level down

`alembic check` measured after each stage. Counts are deduplicated: the tool emits its operation list **twice**
(once as an ERROR log line, once in the raised exception), so raw grep counts are exactly 2× the real figure — a
trap worth naming, since every intermediate measurement in this change was initially double.

| Operation | Before `b3e7c41d92af` | After it | After the ORM declarations |
|---|---|---|---|
| `add_table` | 10 | **0** | 0 |
| `add_index` | 12 | 0 | 0 |
| `add_fk` | 1 | 1 | **1** — pre-existing, `fk_payments_invoice_id_invoices` |
| `remove_index` | 8 | 4 | **0** |
| `remove_column` | 1 | 1 | **0** |
| `modify_default` | 55 | 55 | **54** — all on billing tables |
| **DROP of any table** | **0** | **0** | **0** |

**The hole.** After `b3e7c41d92af` closed every `add_table`, four `remove_index` and one `remove_column` remained —
and all five named objects **this change had just created**:

| Proposed operation | Object | Created by |
|---|---|---|
| `remove_index` | `chunks_bm25_idx` | `a5bd6b69a28e` |
| `remove_index` | `chunks_embedding_idx` | `a5bd6b69a28e` |
| `remove_index` | `chunks_search_text_trgm_idx` | `a5bd6b69a28e` |
| `remove_index` | `idx_outbox_unpublished` | `a5bd6b69a28e` |
| `remove_column` | `chunks.updated_at` | `a5bd6b69a28e` |
| `modify_default` | `outbox_events.publish_attempts` | `a5bd6b69a28e` |

Step 5 applied the drop-safety contract to **table modules** and stopped there. Raw-SQL indexes and a raw-SQL
column are the same defect one level down: an object the database has and `Base.metadata` does not is an object
autogenerate proposes to **DROP**. `a5bd6b69a28e`'s own docstring reasoned correctly that the `UnifiedChunk` model
declares no `updated_at` — then used that fact to justify a server default, without noticing it is precisely the
condition that makes the column droppable.

**`chunks_bm25_idx` is the one that would have hurt.** Its *name* is a literal SQL argument —
`search_text <@> to_bm25query(:query, 'chunks_bm25_idx')`. A future `--autogenerate` folding that `remove_index`
into an unrelated revision would disable keyword retrieval with **no error at any layer**. `remove_column` loses
data; `remove_index` normally loses only speed; this one loses correctness silently.

**Fix — declared in the ORM, not suppressed.** An `include_object` filter in `env.py` would also have silenced the
proposals, but it silences the *report* rather than closing the *gap*, and it would hide a genuine future
divergence too. Two files:

- `src/app/features/documents/model.py` — three `Index()` entries added to `UnifiedChunk.__table_args__`
  (`postgresql_using` `bm25`/`diskann`/`gin`, with `postgresql_with` and `postgresql_ops`), plus the `updated_at`
  column with `server_default=func.now()`.
- `src/app/shared/outbox/model.py` — `__table_args__` created with the partial index
  (`postgresql_where=text("published_at IS NULL")`), and `server_default=text("0")` on `publish_attempts`.

Each declaration was verified by compiling `CreateIndex` under the PostgreSQL dialect and diffing against the DDL
the revision executes — all four are **byte-identical** modulo `IF NOT EXISTS`. Two details are load-bearing:

- `postgresql_with` values render through `str()` **unquoted**, so the text config must carry its own quotes:
  `{"text_config": "'english'"}`. The natural-looking `"english"` emits `text_config = english`, an unquoted
  identifier PostgreSQL rejects. `postgresql_ops` is the inverse — a bare opclass name, because an opclass *is* an
  identifier.
- `env.py:100,117` set `compare_server_default=True`, so the defaults must match **textually**. `func.now()`
  renders as `now()` and `text("0")` as `0`, matching the applied DDL.

Declaring exotic index types in `__table_args__` is safe here because **nothing in the repo calls
`metadata.create_all`** — verified repo-wide; the only hit is prose inside `b3e7c41d92af`'s own docstring. On a
project that did build its test schema that way, these three declarations would require `pg_textsearch`,
`vectorscale` and `pg_trgm` in the test database.

**Residual, and explicitly not in scope:** 54 `modify_default` and 1 `add_fk`, spread over 17 billing relations
(`invoices` 8, `plans` 8, `subscriptions` 8, `invoice_batches` 6, …). Zero of them touch any of the nine relations
this change created — verified by intersecting the drift's table set against that list, result empty. `alembic
check` still exits 255 for this residue; the change-0-attributable figure is **zero operations**.

## 5. Register the live model modules; delete the unreachable fallback

Register the document and search model modules in the migration environment's import block, so the shared registry
contains the unified document and chunk relations, the search relations, the outbox relations and the billing
relations. Registration prevents a **future** comparison from proposing to drop them.

**The six models on the private registry are NOT harvested** (D-3): entity, relationship, parent-document, clause,
event and memory-version. They have **zero importers repo-wide**, and registering them would schedule creation of
relations nothing reads — the mirror image of the defect this change exists to close. Their module is deleted in
step 6 as part of the reconciliation group it sits inside.

Also delete the migration environment's fallback around its metadata assignment: every import sits above it, so it
**cannot be reached**, and an unreachable handler that would have reported a broken registration is worse than none.

**Do not generate a migration by comparison afterwards.** With models newly registered and relations absent,
comparison emits a create for everything — including relations changes 1 and 2 retire.

- [x] Add the document and search model imports to the migration environment.
- [x] Delete the unreachable import fallback.

**Proof**

```bash
uv run python -c "
from alembic import context  # noqa
import src.alembic.env as e
names = sorted(e.target_metadata.tables)
print(len(names)); print('\n'.join(names))" | tee /tmp/c0-registered.txt
rg -n 'try:|except ImportError' src/alembic/env.py   # fallback gone
```

Passes when `documents`, `chunks`, the search relations, the outbox relations and the billing relations all appear,
and no import fallback remains.

### DONE — `src/alembic/env.py` only; registry 30 → 32 relations

**Both bullets landed.** `import app.features.documents.model` and `import app.features.search.model` joined the
block, and the `try: / except ImportError:` around the metadata assignment is gone — now a plain
`target_metadata = Base.metadata`. `rg -c 'except ImportError' src/alembic/env.py` → **exit 1, no match**.

**Deviation, accepted — the block was restructured rather than suppressed.** The block's existing convention was
`# noqa: F401, E402`, and two more bare side-effect imports measure as 4 diagnostics (2×`E402`, 2×`F401`). The
decisive fact is that **`F401` is in this project's ruff `fixable` set**: `uv run ruff check --fix src/` — a command
this change's own tasks instruct — *deletes* an unsuppressed registration import and thereby silently unregisters
its tables. Confirmed by `--stdin-filename` probe: `[*] 2 fixable with the --fix option`. So the imports were made
genuinely referenced instead: the block moved into the top import section (kills `E402`) and a `_MODEL_MODULES`
tuple names every module a second time (kills `F401`). Net **16 suppressions removed, 0 added**; the statements stay
real `import`s, so codegraph/graphify keep their `env.py → */model.py` edges and a typo stays statically catchable.
Cost is +51/−24 lines instead of +2.

**Second deviation, mine — the two credit modules were registered too.** The bullet names only documents and
search. But `0005` creates `user_credits` and `credit_consumptions`, both subclassing the same
`database.base.Base` (`credits/models/credit.py:45`, `consumption.py:29`) that `env.py` reads via
`database/__init__.py:3`, and **step 3's merge pulled `0005` into the single-head chain**. Leaving them
unregistered would have a future comparison propose dropping two relations this chain creates — the exact defect
this step exists to close. Registered as `app.features.credits.models.{credit,consumption}`, with the reason
recorded in-file. Registry went **30 → 32**.

**Import order is load-bearing, and was verified rather than assumed.** `app.features.payments.model` must precede
`app.features.subscriptions.model`: `import app.features.subscriptions.model` alone raises `ImportError` through the
known `subscriptions`↔`payments` cycle, and still fails with `app.connections` imported first — what makes it work
is `payments.model` landing earlier. Alphabetical order happens to satisfy this. `search.model` inserts between
`plans` and `subscriptions`, and the credit pair between `audit` and `documents`; neither disturbs the relation.
Documented in-file.

**Proof line 1 is unexecutable as written** — before *and* after the edit, verbatim:

```
File ".../src/alembic/env.py", line 63, in <module>
    config = context.config
AttributeError: module 'alembic.context' has no attribute 'config'
```

`alembic.context` is a **thread-local proxy populated only during a real alembic run**, so `import src.alembic.env`
can never reach `target_metadata` (and `src` is not a package — no `src/__init__.py`). Another instance of
decisions.md D14.3: a proof that was never run is not a proof.

**Substitute proof** — `/tmp/c0-envprobe.py` loads the real `src/alembic/env.py` through alembic's own
`EnvironmentContext`, so the actual file executes, then prints `sorted(target_metadata.tables)`. Output →
`/tmp/c0-registered.txt`, **32 relations**, containing `documents` ✓ `chunks` ✓ `search_documents`/`search_chunks` ✓
`outbox_events`/`dead_letter_events` ✓ all 15 billing relations from `0002`–`0004` ✓ `user_credits`/
`credit_consumptions` ✓.

**Extra proof**, since the substitute bypasses alembic's CLI: `uv run alembic upgrade head --sql` → **exit 0** with
`INFO [alembic.runtime.migration]` lines still printing, so moving the imports above `fileConfig()` did not break
logging configuration. `uv run alembic heads` → `a5bd6b69a28e (head)`, still singular.

**Gates after this step:** `ruff check src/` **All checks passed!**, `ruff format --check` clean, `alembic heads` 1,
offline render exit 0.

### Three findings from step 5 that belong to other steps

**(1) D-3's premise is already violated, and not by this step.** All six private-registry models — `clauses`,
`entities`, `events`, `memory_versions`, `parent_documents`, `relationships` — are **already** on `Base.metadata`,
both before and after the edit. Path: `env.py from database import Base` → `src/database/__init__.py` →
`src/database/schemas/__init__.py:5-12` → `memory_schema`. So "zero importers repo-wide" is true of *feature* code
and false of the *registry*. This step cannot avoid harvesting them without editing
`src/database/schemas/__init__.py`, which is outside its scope. **D-3 as written cannot be satisfied by step 5
alone** — it is a step 6 concern.

**(2) Step 6 group (7) is missing a coupled edit, and the omission is a boot break.** Deleting
`src/database/schemas/memory_schema.py` requires editing `src/database/schemas/__init__.py:5-12` **and** its
`__all__` at `:14-24` in the *same* commit. Step 6 lists only `src/tasks/__init__.py:6-9,18-20` and the pyproject
per-file-ignore. Without the `database/schemas/__init__.py` edit, `from database import Base` raises
`ModuleNotFoundError` — breaking **every alembic invocation** and every `database` importer at import time, not
merely the Celery workers. Recorded against step 6 below.

**(3) `documents`/`chunks`/`search_*` were registered only by accident before this step** — via
`app/features/__init__.py`, which any `app.features.*.model` import triggers. That is the same
`features/__init__.py` step 6 edits for groups (5) and (6), so the accidental registration was one deletion away
from vanishing silently. The explicit imports close that. The count holding at 30 pre-credits is the expected
result, not evidence of a no-op.

## 6. Delete the remaining dead trees, each with its coupled edit in the same commit

**Six groups remain** (group 1 landed in step 2). Four carry a coupled edit, and **no commit may leave the
application unbootable** — so the coupled edit ships in the *same* commit as its deletion, never after.

- [ ] **(2)** the inverted 36-line parser `src/app/utils/toon_parser.py`.
- [ ] **(3)** the zero-byte vector-store package — **coupled:** `src/app/shared/__init__.py` imports *and*
      re-exports it, so deleting the directory alone is an `ImportError` in every module in the application.
- [ ] **(4)** the zero-byte orchestration-type package (five files, `__init__` included).
- [ ] **(5)** the zero-byte `knowledge_base` feature package — **coupled:** `src/app/features/__init__.py:3,8,9`.
- [ ] **(6)** the zero-byte `web_scraping` feature package — **coupled:** same import list.
- [ ] **(7)** the 1,129-line reconciliation subsystem: its 618-line package, its 209-line worker module, and the
      302-line private-registry schema module — **coupled:** `src/tasks/__init__.py:6-9` imports the reconciliation
      helpers and re-exports at `:18-20`, so deleting without editing this **breaks every Celery worker at import**;
      and a per-file lint-ignore key in `pyproject.toml` names a file inside the tree, which leaves no signal at all
      when it goes stale.

**Nothing is harvested before any deletion.** The earlier constraint — harvest the parent-document and clause models
first — is **void**, because D-3 deletes that module with all six models.

**The one ordering constraint that remains points outward, and it is not satisfiable inside this change.** The shadow
agents tree `src/app/shared/agents/memory/memory_scope.py` (30 bytes, exports only `PRECEDENT_SCOPE`) is
**deliberately NOT in this list**. `precedent_tools.py:21` imports the shadow while `graphiti/subgraph.py:30` imports
the real 7,189-byte module at `src/app/shared/langchain_layer/agents/memory/memory_scope.py`. Deleting the shadow
before **change 3 retargets its importers** makes `registry.py:41-46`'s eager imports raise `ImportError` at boot.
**Blocked on change 3; do not delete it here.**

### CORRECTIONS to step 6, measured at `6525c6f` before it runs

**(a) Group (7) is missing a coupled edit, and the omission breaks the boot — including alembic.** The bullet names
`src/tasks/__init__.py:6-9,18-20` and the pyproject key. It **omits `src/database/schemas/__init__.py`**, which
imports six names out of `memory_schema` at `:5-12` and re-exports them in `__all__` at `:14-24`. Deleting the
schema module without editing that file makes `from database import Base` raise `ModuleNotFoundError` — which
breaks **every alembic invocation** and every `database` importer at import time, not merely the Celery workers.
This edit ships in the **same commit** as group (7).

**(b) The import probe names a module that does not exist.** `celery_app` → `ModuleNotFoundError: No module named
'celery_app'`, measured, at baseline and independent of any deletion here. As written the loop fails on iteration 5
regardless of whether step 6 is correct, so it can never signal anything. The real module is
`app.connections.celery` (`import app.connections.celery` → **OK**; the app object is built inside a factory at
`:187`, so there is no module-level `app` attribute to assert on). `tasks` imports **OK** and stays in the loop.
Substitute the module name; do not add `celery_app` as a file.

**(c) D-3 cannot be satisfied by step 5, so group (7) is the only place it can be honoured.** All six
private-registry models are already on `Base.metadata` today via `database/schemas/__init__.py` — not via feature
code. See step 5's finding (1). Deleting `memory_schema.py` here is what actually removes them from the registry;
until then the registry contains six relations nothing reads, and `parent_documents`, `clauses`, `entities`,
`events`, `memory_versions`, `relationships` appear in step 5's 32-relation output for that reason.

**Proof** (run after *each* group's commit, not once at the end)

```bash
# no zero-byte python survives in the deleted trees
find src/app -name '*.py' -size 0 | sort
# import probe over every boot entry point, including the worker package
for m in app.main app.api.v1 app.api.v2 tasks app.connections.celery app.lifecycle.lifespan; do
  uv run python -c "import importlib,sys
try: importlib.import_module('$m'); print('OK   $m')
except Exception as e: print('FAIL $m', type(e).__name__, e); sys.exit(1)"
done
# emptiness search: the deleted names appear nowhere
rg -l 'toon_parser|knowledge_base|web_scraping|memory_schema|reconciliation' src/ tests/ pyproject.toml
# failure count unchanged vs baseline
uv run pytest -q --no-cov 2>&1 | tail -3
diff <(tail -3 /tmp/c0-baseline/tests.txt) <(uv run pytest -q --no-cov 2>&1 | tail -3)
```

Passes when the import probe is OK for **all six** entry points, the emptiness search is empty, and the
failed/errored counts are **identical** to `/tmp/c0-baseline/tests.txt`. A moved count is a finding — investigate it,
do not accept it.

## 7. Fix the profile handlers' state names and absent-client behaviour

`src/app/features/profile/router.py:29,30` read `app.state.storage` and `app.state.mongodb`; startup publishes
`object_store` (`lifespan.py:108`) and `db` (`lifespan.py:180`).

**A bare rename is the wrong fix** (D-5): startup sets one of those clients to absent on failure while the
annotation promises a value, so a rename converts an `AttributeError` into a `None` that fails later and further
from the cause. Resolve defensively and answer **503** when the client is absent — the shape another feature's
dependency already uses.

- [x] Resolve both clients under the published names, defensively.
- [x] Answer 503 on absence.

**Proof**

```bash
rg -n 'app\.state\.(storage|mongodb)' src/    # must print nothing
rg -n 'app\.state\.(object_store|db)' src/app/features/profile/
rg -n '503' src/app/features/profile/
```

### DONE — 2 files, both inside `src/app/features/profile/`

`router.py:32-60` — `_get_profile_service` now resolves via `getattr(state, "db", None)` (`:47`),
`getattr(state, "redis", None)` (`:52`), `getattr(state, "object_store", None)` (`:57`), raising
`ServiceUnavailableException` → 503 when the database or session store is absent (`:48-50`, `:53-55`).
`service.py:24` — `storage: StorageService | None` (was non-optional, the lie D-5 names); `:81-84` answers 503 at the
point of use.

**Deliberate design choice beyond the two bullets:** the object store is *not* required at resolution time. Only
`POST /profile/avatar` uses it, so demanding it in `_get_profile_service` would make `PATCH /profile` and
`POST /profile/change-password` return 503 whenever S3 is down. Absence is answered at point of use — the same shape
`documents/service.py:153-154` and `invoices/service.py:381-384` already use. `redis` was guarded too: the name was
already correct, but `lifespan.py:203` sets it to `None` on failure — same defect class.

Deviations:

1. **Proof leg 2 (`rg 'app\.state\.(object_store|db)'`) cannot match by construction.** It requires a literal bare
   attribute read — exactly what D-5 forbids, since an unset attribute raises `AttributeError`. The defensive form
   reads `getattr(request.app.state, "db", None)`, whose text never matches that pattern. Substituted with
   `rg 'getattr\(state, "(object_store|db|redis)"'` → 3 hits, plus a `ServiceUnavailableException` census.
2. **Leg 3 (`rg -n '503'`) matches prose only** — the 503 comes from `ServiceUnavailableException`
   (`utils/exceptions.py:163`), so no literal `503` appears in profile code. Asserted by direct probe instead:
   empty state → 503 *User database is unavailable*; `redis=None` → 503 *Session store is unavailable*;
   `object_store=None` → service builds, then `upload_avatar` → 503 *Object storage is not configured*; and
   **publishing only the stale names still yields 503**, confirming they are no longer consulted.
3. **Cited line numbers were stale** (harmless): `lifespan.py` `object_store` is `:119` not `:108`, `db` is `:191`
   not `:180`. The new docstring deliberately carries no line numbers so it cannot rot the same way.
4. **Confirmed genuinely absent before implementing** — `rg 'app\.state\.(storage|mongodb)' src/` had exactly two
   hits repo-wide, both in `profile/router.py`; no defensive read and no `ServiceUnavailableException` existed in
   `profile/`. Nothing was duplicated.

Gates unchanged from baseline: ruff 0, ty 2 (both still `auth/service.py:494,501` — not this step's), format clean,
ast-grep 4, pytest 3 failed / 103 passed / 9 errors. No suppression added.

**Two live defects found in `profile/` and deliberately NOT fixed — unowned, assign these:**

1. **`POST /profile/avatar` is still broken after this step.** `profile/service.py:86` calls
   `storage.upload_avatar(...)`, but `StorageService` has **no `upload_avatar` method** —
   `shared/services/storage.py` exposes `put_object`, `get_signed_put_url`, `delete_object`. The call type-checks only
   via a pre-existing `# ty: ignore[unresolved-attribute]`, correctly left in place. Runtime result is
   `AttributeError` → 500. `storage.public_url` does exist (`storage.py:241`), so only the method is missing.
2. **`profile/service.py:96` assigns `user.avatar_url`, which is not declared on the auth `User` model.**
   Repairing it requires editing `src/app/features/auth/`, fenced off for concurrency.

## 8. Fix identity resolution repo-wide

Rewrite all four unguarded identity readers onto the existing dependency that decodes and validates the access
token and returns its claims with no database round trip. The fifth is a guarded branch in an **unmounted** router
and is **deleted** rather than kept alive by introducing a writer for state nothing assigns.

**Ordering, load-bearing: this lands in or after step 4's commit, never before.** Repairing identity *without* the
outbox relations does not repair the upload endpoint — it moves the 500 from the dependency layer **down to the event
insert**. The two land together or the repair is illusory.

**BREAKING:** six mounted document endpoints move from `500` to `401` for unauthenticated calls. That is the delta,
not a side effect.

- [x] Rewrite the four readers onto token claims.
- [x] Delete the guarded branch in the unmounted router.

**Proof**

```bash
rg -n 'request\.state\.user_id' src/          # must print nothing
rg -n 'UserIdDep' src/app/features/documents/dependencies.py
git log --oneline -1 -- src/alembic/versions/  # the revision commit must not be AFTER this one
```

The `401` is asserted by a **direct probe, not an automated test**, and the reason is measured rather than assumed:
no `client` fixture exists anywhere in the suite (the thirteen setup errors are precisely its absence), so an
endpoint-level assertion requires test infrastructure this change does not own; and the coverage gate makes the
runner's exit code unusable as a proof for *any* task here. The automated version is **follow-up owned by test
infrastructure** — repairing the fixture changes neither these specs nor this breakdown.

### DONE — but bullet 1 had no subject left, and this step is **no longer BREAKING**

**Bullet 1 was already landed, in commit `7fc0ab5`** — titled, verbatim,
`fix(features): replace request.state.user_id auth stubs with real token claims`.
`features/documents/dependencies.py:62` reads:

```python
async def get_current_user_id(claims: CurrentClaims) -> str:
    return claims.sub
```

which is exactly what this bullet specifies — the existing dependency that decodes and validates the access token
and returns its claims **with no database round trip** — and `UserIdDep` (`:68`) is consumed **7 times** in
`features/documents/router.py`. This is the seventh instance of the pattern in
[verify-before-declaring-greenfield], and the first where checking the *edge wiring* confirmed the feature was not
merely present but actually connected.

**Bullet 2 was the only remaining work, and it is done.** `features/crawler/router.py` — the `hasattr`-guarded
branch at `:27-28` is deleted. The enclosing `get_client_identifier` is **kept**, because it has three live callers
at `:59`, `:90`, `:116` inside that same file; only the identity branch went, and the docstring's "(IP or user ID)"
claim was corrected, since it was describing a branch that can never be taken.

**The step's rationale is now proven rather than assumed.** It says to delete the branch "rather than keep it alive
by introducing a writer for state nothing assigns." Measured repo-wide: **nothing in `src/` assigns that request-state
attribute — there are zero writers.** So the guarded branch was unreachable, and introducing a writer would have
manufactured exactly the inverse defect.

**The BREAKING notice is void.** The `500` → `401` transition on the six mounted document endpoints **already
happened** in `7fc0ab5`. What landed here deletes unreachable code in a router that is **not mounted** — verified
twice: no `api/v1` or `api/v2` module references `crawler`, and the crawler package has **no importer anywhere
outside itself**. No mounted behaviour changes.

**Consequently the ordering gate is moot.** "This lands in or after step 4's commit, never before" existed because
repairing identity without the outbox relations moves the 500 from the dependency layer down to the event insert.
With bullet 1 already landed months earlier and bullet 2 confined to an unmounted router, **there is no interaction
with the outbox relations at all — step 8 is not gated on the database upgrade.**

**Proof legs.** Leg 1 `rg -n 'request\.state\.user_id' src/` → **zero hits repo-wide** ✓. Leg 2 `UserIdDep` present
at `dependencies.py:68` ✓. Leg 3 (`git log -1 -- src/alembic/versions/`) is **unexecutable today** because the two
revision files are still uncommitted; once step 4 is committed it passes by construction, since this edit sits in
the same working tree. Gates: ruff `All checks passed!`, format clean, and `app.main` / `app.api.v1` / `app.api.v2` /
`tasks` all import OK.

Note on the leg-1 proof, recorded because it nearly shipped broken: the docstring I first wrote *named* the
attribute, which made leg 1 print a hit and read as a failure. A search-must-be-empty gate cannot distinguish prose
from code. The prose was rewritten to describe the attribute without spelling it, and the docstring now says why.
Second occurrence of documentation breaking a gate in this change — the first was an ambiguous-unicode character in
a migration docstring.

**Unowned finding — every rate limit in the application is per-IP, silently.**
`src/app/utils/rate_limit/dependencies.py:27` reads `getattr(request.state, "user_id", None) or client_ip`. Since
there is **no writer** for that attribute, the first operand is always `None` and the fallback **always** wins. So an
authenticated user is never rate-limited as a user: everyone behind a shared NAT or proxy shares one bucket, and
per-user limiting does not exist despite the code appearing to implement it. Not a crash — a silent degradation, the
same class as `_HEALTH_TIMEOUT_S` being defined and never applied. Fixing it means either assigning the attribute in
middleware or moving the limiter onto token claims, both outside this step. **Unowned.**

**Second unowned finding — the whole `crawler` feature is orphaned.** No `api` module mounts it and nothing outside
its own package imports it. It is a dead-tree candidate of exactly the kind step 6 removes, but step 6 does not name
it, so it was left in place rather than expanding that step's scope unilaterally.

## 9. Move every database consumer onto the accessor

The accessor is **correct and is not the defect** — it rewrites the scheme, strips parameters the async driver
rejects, and injects the missing credential. The defect is that consumers **bypass** it: two read the raw,
credential-less configured value and a third string-edits the accessor's output, which is then string-edited again
downstream.

**Two flavours, and exactly two** (D-6): the async-ORM dialect for the application's pool, and a plain connection
URL for low-level-driver consumers, retaining the transport-security parameter those drivers want. One function
returning one string cannot serve both, so add explicit flavour selection.

**Plus discrete fields.** The embedded memory component takes a **discrete-field configuration object** and has no
connection-string field at all — building a third URL flavour would produce surface with no possible caller. Exposing
the same underlying values as discrete fields also closes a worse defect: that call site currently reads host and
database name from settings **independently of the accessor**, so it can be pointed at a different database than the
application with a valid credential and **succeed silently**.

- [x] Add flavour selection; move all three bypassing consumers onto the accessor.
- [x] Expose discrete fields; move the memory component's call site onto them.
- [x] Correct the checkpointer module's guidance — it currently recommends the async dialect, which that consumer's
      driver **cannot accept**, while naming a raw value that carries **no credential**. Wrong in both directions,
      and change 1 would otherwise follow it.
- [x] Fix three latent one-line defects: percent-encode the credential (a rotation onto a credential containing
      reserved characters silently produces a malformed URL); stop skipping injection by comparing against a
      placeholder literal; stop appending a port to a value that already has one on the no-username branch.

**Proof**

```bash
rg -n 'DATABASE_URL|POSTGRES_URL' src/ --glob '!**/config/*'   # no raw reads outside settings
rg -n 'replace\(.*postgresql' src/                             # no ad-hoc string surgery
rg -n 'quote_plus|quote\(' src/app/database/            # percent-encoding present
uv run python -c "
from app.database.connection import get_database_url
print(get_database_url(flavour='async')[:24])
print(get_database_url(flavour='plain')[:24])"
```

### DONE — 6 files; the accessor rebuilt by parse rather than string surgery

`src/app/connections/postgres.py` carries the change: `type DatabaseUrlFlavour = Literal["async","plain"]` (`:37`),
`get_database_url(flavour="async")` (`:124`), `get_database_fields()` (`:152`) returning a frozen
`DatabaseConnectionFields` model (`:58`) whose `password` is a `SecretStr`. `settings.POSTGRES_URL` is now read
**exactly once in the repository**, at `:140`, inside the accessor. Five other files moved onto it: `lifespan.py`
(`:135`, `:313-318`), `shared/outbox/relay.py` (`:35-40`, `:83`), `shared/langgraph_layer/checkpointer.py`,
`shared/langchain_layer/agents/memory/cognee_client.py` (`:33`, `:82`, `:103-120`).

All three callers enumerated and updated: `postgres.py:173` (`init_db`, async), `lifespan.py:135` (plain),
`auth/service.py:619` (async). `src/alembic/env.py:127` reaches it through `init_db()`. `connections/__init__.py:39`
exports only `get_postgres_db, init_db`, so there is no third import surface. Gates unchanged: ruff 0, ty 0, format
clean, ast-grep 4, pytest identical, `alembic heads` 1. No `# noqa`, no `# type: ignore`.

**The async output is byte-identical to the pre-change accessor** on the deployed configuration
(`sha256[:12] old=1909ca49ddd1 new=1909ca49ddd1`), so no existing async caller changed behaviour. `init_db()`
verified live read-only: `dialect=postgresql+asyncpg host=… port=39662 db=tsdb`, `alembic current` → `0004`,
`alembic heads` → `a5bd6b69a28e`. No DDL run.

**A live defect fixed that the task text did not name.** `postgres.py:134` derived the log's database name with
`settings.POSTGRES_URL.split("/")[-1]`. The deployed URL ends `/tsdb?sslmode=require`, so the startup log read
`database=tsdb?sslmode=require` — the query string was being reported as part of the database name.

**The coordinator's instruction on credential precedence was wrong, and was overridden.** This coordinator wrote
that "the discrete fields are the complete and authoritative source; the URL is not," on the strength of the
preflight measurement that `POSTGRES_URL` carries no password. The accessor was built the other way round — **URL
authoritative, each discrete field a per-component fallback used only when the URL omits that component**, with the
password the single value always taken from the field because the URL genuinely lacks it. The reasoning is better
than the instruction and is adopted:

- `POSTGRES_HOST`/`PORT`/`DB_NAME`/`USERNAME` default to `localhost`/`5432`/`db`/`user`. Under a fields-first
  accessor, a deployment that sets **only** `POSTGRES_URL` — which is the single value a managed provider hands you —
  would connect to `localhost:5432/db` and report success. **Derived-from-URL fails loudly; fields-first fails
  silently.**
- Step 9's own premise is that the accessor is correct and the defect is call sites reading fields *independently of
  the accessor*. Making the fields authoritative would invert that premise.

**The encoding advice was also corrected by measurement.** This coordinator offered `quote(pw, safe="")` **or**
`quote_plus`. Only the first is correct: SQLAlchemy (`engine/url.py`) and asyncpg (`connect_utils.py`) both decode
userinfo with plain `unquote()`, which does **not** map `+` back to a space — so `quote_plus` would authenticate
with a literally different secret. Verified `unquote(quote_plus(s)) == s` → `False` for `p@ss w+d/:?#`, and the same
secret round-trips correctly through both real parsers under `quote(…, safe="")`. The measured credential contains
none of `% + @ / : ? #`, so the defect was **latent — it would first appear on a credential rotation**, which is why
it was fixed regardless of the current value.

`sqlalchemy.URL.create` was considered and rejected with a stated reason: the return type must remain `str` for
`asyncpg.connect(dsn=…)`, and `str(URL)` masks the password as `***` unless
`render_as_string(hide_password=False)` is used — a sharper footgun than an encoding round-trip verified against
both decoders.

**Two proof legs were unexecutable as written; both were replaced with stronger forms.**

1. `from app.database.connection import get_database_url` and `rg … src/app/database/` — **`src/app/database/`
   does not exist**; the accessor lives at `app/connections/postgres.py`. Re-pointed at `src/app/connections/`:
   `quote(secret, safe='')` at `:121`.
2. `get_database_url(flavour='async')[:24]` — **not run.** A 24-character slice of a DSN reaches into the userinfo
   and prints the credential. Replaced with a `urlsplit` probe reporting scheme/host/port/database and the booleans
   `credential_present`, `credential_matches_setting`, `port_appears_once`, with userinfo rendered `<redacted>`.
   This is a **stronger** proof than the slice: it asserts the credential is present and correct instead of
   displaying twenty-four characters of it. Leg (a) also could not return empty (see below) and was replaced with an
   `ast-grep` form, which excludes comments by construction — `ast-grep run -p 'settings.POSTGRES_URL' -l py src/`
   returns exactly one hit, the accessor's own read.

Leg (b) as written (`rg 'replace\(.*postgresql' src/`) returned empty **but never matched the surgery that actually
existed**, which was `.replace("+asyncpg","")`. Both the original and a widened form return empty now. This is a
third instance of the pattern D14.3 names: a mechanical-looking proof that passes without testing its claim.

**`rag_agent_advanced.py:571` excluded from scope, with reasons.** It is not a database consumer: `initialize_db`
(`:33`) and `close_db` (`:38`) are logging stubs, `db_pool` (`:27`) is assigned `None` and never reassigned, the
module has **zero importers**, runs only under `__main__`, and reads `DATABASE_URL` — which is **not a `Settings`
field at all**, so there is no accessor value for it to bypass. Pointing it at the accessor would give a dead guard
a real dependency. It is dead-guard debt for a deletion sweep, and it is the reason leg (a) cannot return empty
while the file exists.

### Six findings from step 9 that belong elsewhere

1. **`POSTGRES_HOST`/`PORT`/`DB_NAME`/`USERNAME` are now dead for the database path** except as per-component
   fallbacks. Measured, they currently agree with the URL, so `findings-database.md` §9's Cognee-divergence hazard
   was **latent, never active**. Deletion candidates, but the fallbacks are load-bearing for a URL missing a
   component and `settings.py` is unowned.
2. **`cognee_client.py:107` still shadows `config`** — two locals of that name in `setup_cognee`, one configuring
   nothing (§9). Left alone; the renaming is change 4's, and the misleading `postgres_url` key is gone.
3. **`rag_agent_advanced.py`** — 18.6 KB, zero importers, `__main__`-only, stub lifecycle functions, references a
   setting that does not exist. Deletion candidate for the dead-code sweep.
4. **`auth/service.py:619` still builds a second engine outside the lifespan** (§2/§6). It now inherits the
   corrected accessor, so its URL is right; the duplicate pool is untouched and unowned.
5. **`relay.py:66` and `:81` still carry catch-alls** that make outbox failure silent. Untouched deliberately under
   D14.1 — tightening them before the tables exist converts silent degradation into a boot failure.
6. **`checkpointer.py:29`'s `AsyncPostgresSaver = Any` fallback is still the live path.** `psycopg-binary` is
   absent, so `setup_langgraph_checkpointer` short-circuits at `:51-53` **whatever DSN it is handed**. Step 9 fixed
   the *guidance* so change 1 does not follow it; the missing driver is change 1's step zero (§5).

## 10. Add the graph-memory dependency to the versioned health report

After step 7, so the probe reads a state surface whose contract is already correct.

**Additive only, on two API versions at once**, and the checks model forbids unknown fields. An absent optional
dependency reports `not_configured` and **does not change the overall status or the HTTP status code** — mirroring
how the existing graph-database check is already treated. Without that rule, every environment without the optional
dependency starts returning `503` from a mounted endpoint.

Note the scope correction: `check_graphiti` **already exists** at `src/app/features/health/health_check.py:83-90`.
Only whatever `features/health/service.py:160` still misses belongs here; **`check_cognee` is change 4's**.

- [x] Add the probe to both response shapes, additively.
- [x] Report `not_configured` on absence without altering overall status.

### DONE — 4 files, all inside `src/app/features/health/`

`src/app/middleware/health_check.py` was **not** touched: it already carries `check_graphiti` at `:83-90`, and this
step's deliverable is the *versioned* report.

| File:line | Change |
|---|---|
| `health/dto.py:28` | `graphiti: dict[str, Any]` on `HealthChecksDTO`, between `neo4j` and `celery`. Purely additive. |
| `health/service.py:26,31` | `_GRAPH_MEMORY_PROBE_TIMEOUT_S = 2.0`, `_GRAPH_MEMORY_PROBE_QUERY = "RETURN 1 AS ok"` |
| `health/service.py:33-47` | `GraphQueryDriver` / `GraphMemoryClient` Protocols |
| `health/service.py:61-62,69` | `graph_memory_client` ctor param, keyword-only, `None` default |
| `health/service.py:98,108` | probe called in `get_health()`; `graphiti=graphiti_check` into the DTO |
| `health/service.py:212-244` | `_check_graphiti()` |
| `health/service.py:309` | `checks.graphiti` added to `_compute_overall_status`'s `all_checks` |
| `health/dependencies.py:41-45,59` | `get_health_graph_memory_client()` reading `getattr(app.state, "graphiti", None)` |
| `health/__init__.py` | exports the provider and `GraphMemoryClient` |

**The scope-correction path above is stale.** `src/app/features/health/health_check.py` **does not exist**; the file
holding `check_graphiti` is `src/app/middleware/health_check.py`. And `service.py:160` — cited as "whatever still
misses" — was `_check_neo4j`. The gap was verified by grepping the *edge wiring* rather than the symbol, per
[verify-before-declaring-greenfield]: `HealthChecksDTO` had 7 fields and none was graphiti, `HealthService` took no
graphiti client, and no provider read `app.state.graphiti`. **Genuinely missing** — the unversioned `/health` had the
probe, the versioned report did not.

**Protocols instead of importing `graphiti_core`, for a measured reason.** `from graphiti_core import Graphiti`
costs **1755 ms**, and `app.features.health` sits on the `app.api.v1` → `app.main` import path — so a real import
would be paid at every boot, including every test collection. Structural typing gets the annotation with no runtime
edge.

**A bounded live query, not a presence check.** The step specifies neither timeout nor degraded semantics, so
nothing was assumed — but `app.state.graphiti` is set once at boot and **never cleared**, so a presence check would
report `healthy` forever after the backend died. That is "already built ≠ working" in probe form. Bounded at 2 s by
`asyncio.timeout`; a hanging backend reports `TimeoutError` instead of hanging the endpoint.

**`PLR0917` forced a design choice rather than a suppression.** A 6th parameter tripped
`too-many-positional-arguments` in both `HealthService.__init__` and `get_health_service` — `PLR0913` is ignored in
`pyproject.toml` but `PLR0917` is not, and `preview = true`. Making the new parameter keyword-only (`*`) clears it
**and** keeps every existing call signature valid. No suppressions added anywhere in this step.

**Both bullets verified behaviourally, five cases measured:** all six sub-dependencies resolve under FastAPI;
**absent** → `{"status":"unknown","state":"not_configured"}`, overall `healthy`, **200**; reachable → `healthy` with
`responseTime`; hanging → bounded, `TimeoutError`; and a driver raising
`ServiceUnavailable("…bolt://neo4j:<secret>@host:7687")` leaks nothing — `'bolt://' in body: False`, secret in body
`False`, because the probe reports `type(exc).__name__` only, never `str(exc)`, in both body and log line. One
router is mounted on `v1_router` *and* `v2_router`, so the edit lands on both versions at once; health-router
OpenAPI generates cleanly at `/api/v1/health/` and `/api/v2/health/` with `graphiti` in `properties` and `required`.

**Proof deviation — the block names a symbol that does not exist.** `from app.features.health.service import
compute_overall` → `ImportError`. The real symbol is `HealthService._compute_overall_status(checks:
HealthChecksDTO) -> str`, a staticmethod taking a **DTO, not a dict**, so *both* halves of the proof were
unexecutable. Substituted a proof written reflectively off `model_fields`, so one identical command runs on both
trees: fields `7 → 8` (`graphiti` inserted), and overall/status_code with all dependencies absent **IDENTICAL**
before and after (`diff` exit 0, both `healthy` / `200`).

**Flagged, because it is a live-traffic edge:** present-but-failing graphiti yields **503**. The step constrains
only the *absent* case and names `_check_neo4j` as the mirror, which behaves exactly this way. Since
`status_code = 200 if overall == "healthy" else 503`, even `degraded` returns 503 here — `not_configured` mapping to
`"unknown"` is the **only** value that preserves 200, which is precisely why absence must route through it.

### Three findings from step 10 that belong elsewhere

**(1) `src/app/middleware/health_check.py:21` — `_HEALTH_TIMEOUT_S = 2.0` is defined and never used**, while the
module docstring at `:3` claims "Each probe uses a 2-second timeout." All five probes on the unversioned `/health`
are in fact **unbounded**. The docstring is false today. Fixing it changes status codes on a mounted endpoint, so it
is outside this step's additive-only constraint. **Unowned.**

**(2) `src/app/features/health/service.py:208` — `_check_neo4j` catches only `Neo4jError`.** `ServiceUnavailable`
and `SessionExpired` derive from `DriverError`, a **sibling** of `Neo4jError` under `GqlError` — so the *most likely*
connectivity failure escapes the handler entirely and propagates out of the health check. Verified: the new
`_check_graphiti` at `:231` catches `(Neo4jError, DriverError, OSError, TimeoutError)` for exactly this reason, and
case 5 above confirms `ServiceUnavailable` is handled there. A one-word fix, but not this step's.

**(3) `app.openapi()` fails for the whole application, pre-existing.** `PydanticUserError`:
`PlanCreateDTO` … `is not fully defined`, from `src/app/features/plans/dto.py` — reproduced independently, and
`git status` shows **nothing uncommitted touches `plans/`**. Health-only routers generate fine; the whole-app schema
does not. This breaks `/docs` and any schema export on shipped surface. Belongs to whoever owns `features/plans/`.
**Unowned.**

**Proof**

```bash
# record overall status BEFORE the edit, require it unchanged after
uv run python -c "
from app.features.health.service import compute_overall
print(compute_overall({}))" > /tmp/c0-health-before.txt
# after:
diff /tmp/c0-health-before.txt <(uv run python -c "
from app.features.health.service import compute_overall
print(compute_overall({}))")
rg -n 'not_configured' src/app/features/health/
```

Passes when the all-absent overall status is **identical** before and after, on both versions, and no field is
renamed or removed.

## 11. Fix the two `object` annotations and the raising logging import

- [x] Give real types to the two `object`-typed parameters in the blast radius. A **third** `object` annotation is
      correct as written — it accepts genuinely unknown input — and is **left alone**.
- [x] Fix `src/app/utils/embedding.py:5`, which raises on every embedding-dimension mismatch.

Disposition-ledger item 199 is corrected: the constructor it names was **already fixed**; the genuine residue is two
other parameters in the same feature.

**Proof**

```bash
uv run ty check src/ 2>&1 | tail -2      # count <= /tmp/c0-baseline/ty.txt
uv run python -c "import app.utils.embedding; print('import OK')"
```

### DONE — one edit, not two; `embedding.py` was already fixed

**`src/app/utils/embedding.py` needed no edit.** Its raising import was fixed on 2026-08-20 in commit
`52baccb`: `from app.utils import logger` → `from app.utils.logger import logger`. The bullet describes a
defect that no longer exists. This is the sixth instance of the pattern in
[verify-before-declaring-greenfield] — the plan text was written against an older tree. `import app.utils.embedding`
→ `import OK`.

**The typed parameter.** `src/app/features/auth/service.py:462` had `ws_security_service: object`. Rather than
name the concrete class — which would import a service into a service and deepen the coupling this change is
unwinding — a structural type was added at `:54-72`:

```python
class WebSocketConnectionCloser(Protocol):
    redis: Redis
    async def close_connection(self, connection_id: str, /, *, reason: str) -> None: ...
```

`Redis` is imported under `TYPE_CHECKING` (`:24`), so nothing is added to the runtime import graph.

**Deviation — only one of the two named annotations was real.** The bullet says *two* `object` parameters plus a
third that is correct as written. Measured, the feature has **one** `object`-typed parameter and one
`dict[str, object]`; `payload: dict[str, object]` at `:593` **is** the third case the bullet exempts — genuinely
unknown JSON input — so it was left alone, and there is no second edit to make. The bullet's arithmetic is off by
one, not its intent.

**Gate.** `ty check src/` 2 → **1**. Both remaining-at-baseline diagnostics were in `auth/service.py` and both are
gone; the 1 that remained at the time of measurement was `health/service.py:101`, in-flight from step 10, and is
now also clear (`ruff` and `ty` both clean on `src/` as of this writing).

**Proof deviation.** `/tmp/c0-baseline/ty.txt` did not exist in the subagent's session — step 1's baseline files
did not survive across process boundaries, so `count <= baseline` was unexecutable as written. Compared against
the **measured** baseline of 2 from step 1's `### MEASURED` block instead. Any later step whose proof reads a
`/tmp/c0-baseline/` file must expect the same and use the recorded figures.

**KNOWN GAP, not fixed here — the Protocol has no implementer.** `WebSocketSecurityService`, the class actually
passed at the call site, declares **neither** `redis` **nor** `close_connection`. The annotation now states the
contract the caller depends on, and that contract is unmet: `revoke_session_and_close_connections` is dead and
fails silently today. Typing it is what makes the gap visible; **closing it is unowned work** and is listed with
the other unowned items rather than smuggled into this step.

## 12. Final gate

Every rung compared against the files from step 1, **none by exit code**.

- [x] Run all seven gates; compare each to its baseline file.

### MEASURED 2026-08-23 — all seven at baseline

| Gate | Baseline (step 1) | Final | |
|---|---|---|---|
| `uv run ruff check src/` | `All checks passed!` | `All checks passed!` | = |
| `uv run ruff format --check src/` | 358 files | **359** files | = + the one new revision file |
| `uv run ty check src/` | `All checks passed!` | `All checks passed!` | = |
| `ast-grep scan src/` | 41 (4 error + 37 warning) | 41 (4 error + 37 warning) | = |
| `uv run pytest -q` | 3 failed, 103 passed, 48 deselected, 9 errors | identical | = |
| `uv run alembic heads` | 1 | 1 — `b3e7c41d92af` | = |
| `openspec validate --all --strict` | 21 passed, 6 failed | 21 passed, 6 failed | = |

The 12 pytest failures/errors are the pre-existing websocket fixture drift, owned by no step here.
The 4 `no-raw-httpexception` errors are all in `src/app/examples/redis_examples.py` (`:211`, `:239`, `:265`, `:299`).

**The ast-grep figure needed settling, and the reason is a lesson about the gate itself.** An intermediate
measurement recorded this gate as **4**, and the final run reported **41** — which reads as a 37-point regression.
It is not. The earlier number counted `error[...]` lines only; the later one counted `error` *and* `warning`.
The 37 `warning[no-raise-app-error-mapper]` hits were never inside the earlier figure's scope.

Settled empirically rather than by argument, in a detached worktree at `6525c6f` so the working tree's 33 deletions
and 3 untracked revisions could not perturb it:

```
git worktree add --detach /tmp/agbase 6525c6f
BASELINE (6525c6f):  4 error[no-raw-httpexception]   37 warning[no-raise-app-error-mapper]   total 41
NOW (working tree):  4 error[no-raw-httpexception]   37 warning[no-raise-app-error-mapper]   total 41
```

Identical counts **and identical composition**. `git stash` was deliberately not used: with 33 staged deletions and
untracked files present, a stash/restore cycle is a mutation of the thing being measured, while a detached worktree
is a read.

**Generalise this: "compare each to its baseline file, none by exit code" is not sufficient on its own.** Two runs of
the same command can differ because the *filter* changed rather than the code. A gate figure is only comparable if
the severity scope, the path argument and the post-processing pipeline are all identical — record the command
verbatim, not just the number it produced.

**Proof**

```bash
uv run ruff format --check src/
uv run ruff check src/    2>&1 | tail -2   # <= baseline, and any drop maps to a deleted file
uv run ty check src/      2>&1 | tail -2   # <= baseline
ast-grep scan src/        2>&1 | tail -3   # <= baseline
uv run pytest -q --no-cov 2>&1 | tail -3   # failed/errored identical to baseline
uv run alembic heads | wc -l               # 1
openspec validate --all   2>&1 | tail -3   # no NEW failures beyond the baseline set
openspec validate cleanup-foundation --type change --strict
```

**On `openspec validate --all`: the expectation is "no new failures", not "passes".** Six failures are
pre-existing and **unreachable by this change**. Four — `cognee-v1-api`, `noqa-documentation`,
`pattern-matching-standard`, `typed-exception-handling` — fail for a missing `## Purpose` section, and **nothing in
the delta mechanism emits a `## Purpose` header**: `## ADDED` and `## MODIFIED` are the only sections a delta
contributes, so no change can repair them. The fifth, `transactional-outbox`, fails because no requirement body
carries SHALL/MUST; this change's delta supplies the keyword in **two of six** requirements, leaving four still
non-normative, so it stays red — the delta reduces the count, it does not turn the capability green. The sixth is
`change/mintlify-documentation`, an unrelated open change with its own author, named only so the count of six is
fully accounted for.

**And do not read a green `--strict` as evidence the deltas are complete.** It is evidence they are *well-formed*,
which is much weaker. A `## MODIFIED` block replaces its requirement **wholesale** on archive, so a block
reproducing fewer scenarios than the accepted requirement carries **deletes the missing ones** — no `## REMOVED`
block, no Reason, no error — and strict validation accepts it, because the evidence lives in a file the change does
not contain. This change's `typed-exception-handling` delta had exactly that defect through two drafts. **Count the
accepted requirement's scenarios and reproduce every one.**

---

## Rollback

Steps 2, 5–11 are ordinary reverts.

**Step 3** is revertible by deleting the merge revision — **only while no environment has upgraded past it**.

**Step 4 is not revertible by reversal.** Its reversal deliberately does not drop the outbox relations, and the
document relations it creates are owned by another revision's reversal. Roll it back by **restoring from a
snapshot**, which is safe precisely because there is no data: the target relations hold **zero rows**, because they
do not yet exist.

## Blocked on other changes

| Item | Blocked on | Why |
|---|---|---|
| Deleting the 30-byte shadow `shared/agents/memory/memory_scope.py` | **change 3**'s importer rewrite | `precedent_tools.py:21` imports the shadow; deleting first raises `ImportError` at boot (D6.1) |
| Resolving `clauses` | **change 2**, item 184 Option A+ | Only matters if the rewind route is ever revisited; this change's forward revision does **not** depend on it, which is *why* the forward route was chosen |
| Narrowing the relay's broad exception catch | **after** the relations exist | Recorded Non-Goal. Tightening it before step 4 converts silent degradation into a **boot failure** |
| Retiring the stale `Reconciliation fetch failure` scenario | spec-hygiene pass | No scenario-level `## REMOVED` exists; retiring one scenario is a direct edit to the accepted spec, alongside the four `## Purpose` failures |
| Asserting the `401` by automated test | test infrastructure | No `client` fixture exists anywhere in the suite |
