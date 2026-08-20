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

- [ ] Capture tests (coverage disabled), collection, lint, types, format, structural scan, spec validation,
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

## 2. Delete the unparseable draft

First, deliberately: its proof is a **drop** in the lint baseline, which is what establishes that the baseline files
from step 1 are trustworthy before any harder step depends on them.

- [ ] Delete `src/app/shared/rag/document_processing/todo_temp.py` (783 lines; does not parse — its `__all__`
      closes and an orphaned class body resumes below it). Zero importers.

**Proof**

```bash
rg -l 'todo_temp' src/ tests/          # must print nothing
uv run ruff check src/ 2>&1 | tail -2  # count must DROP vs /tmp/c0-baseline/ruff.txt
```

Passes when the importer search is empty and the count drops. **Confirm the drop maps to the deleted file** — a drop
that maps to a suppressed real diagnostic is a regression wearing a green tick.

## 3. Join the two migration heads

A merge revision with an **empty body**: both branches declare the same parent and touch disjoint relations, and
both create their extensions idempotently, so there is nothing to reconcile.

The docstring is load-bearing and is the deliverable here as much as the revision is. It must name every phantom
relation, name `9f4a1b7c6d2e` as **unrunnable** and the `clauses` relation it presupposes, and state that **reversal
below this point is unsupported** — because those reversals drop relations that were never created.

This is also what makes `alembic upgrade head` (singular) resolve again, repairing `Makefile:39`, `README.md:272`
and `.github/workflows/test.yml:105` **without editing them**.

- [ ] Add the merge revision joining the two heads.
- [ ] Write the docstring: phantom relations, `9f4a1b7c6d2e` unrunnable, reversal prohibition.

**Proof**

```bash
uv run alembic heads | wc -l           # 1
uv run alembic upgrade head --sql >/dev/null; echo "exit=$?"   # 0 — singular now resolves
rg -c 'clauses|9f4a1b7c6d2e|reversal' src/alembic/versions/<merge_rev>.py
```

Passes when exactly one head remains, the singular form exits 0, and the docstring names all three subjects.

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

- [ ] Add the authoritative revision: outbox first, then documents/chunks, `updated_at`, indexes, four extensions.
- [ ] Create all three retrieval branches' indexes — vector, BM25 keyword, trigram fuzzy — by exact name.

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

- [ ] Add the document and search model imports to the migration environment.
- [ ] Delete the unreachable import fallback.

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

**Proof** (run after *each* group's commit, not once at the end)

```bash
# no zero-byte python survives in the deleted trees
find src/app -name '*.py' -size 0 | sort
# import probe over every boot entry point, including the worker package
for m in app.main app.api.v1 app.api.v2 tasks celery_app app.lifecycle.lifespan; do
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

- [ ] Resolve both clients under the published names, defensively.
- [ ] Answer 503 on absence.

**Proof**

```bash
rg -n 'app\.state\.(storage|mongodb)' src/    # must print nothing
rg -n 'app\.state\.(object_store|db)' src/app/features/profile/
rg -n '503' src/app/features/profile/
```

## 8. Fix identity resolution repo-wide

Rewrite all four unguarded identity readers onto the existing dependency that decodes and validates the access
token and returns its claims with no database round trip. The fifth is a guarded branch in an **unmounted** router
and is **deleted** rather than kept alive by introducing a writer for state nothing assigns.

**Ordering, load-bearing: this lands in or after step 4's commit, never before.** Repairing identity *without* the
outbox relations does not repair the upload endpoint — it moves the 500 from the dependency layer **down to the event
insert**. The two land together or the repair is illusory.

**BREAKING:** six mounted document endpoints move from `500` to `401` for unauthenticated calls. That is the delta,
not a side effect.

- [ ] Rewrite the four readers onto token claims.
- [ ] Delete the guarded branch in the unmounted router.

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

- [ ] Add flavour selection; move all three bypassing consumers onto the accessor.
- [ ] Expose discrete fields; move the memory component's call site onto them.
- [ ] Correct the checkpointer module's guidance — it currently recommends the async dialect, which that consumer's
      driver **cannot accept**, while naming a raw value that carries **no credential**. Wrong in both directions,
      and change 1 would otherwise follow it.
- [ ] Fix three latent one-line defects: percent-encode the credential (a rotation onto a credential containing
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

## 10. Add the graph-memory dependency to the versioned health report

After step 7, so the probe reads a state surface whose contract is already correct.

**Additive only, on two API versions at once**, and the checks model forbids unknown fields. An absent optional
dependency reports `not_configured` and **does not change the overall status or the HTTP status code** — mirroring
how the existing graph-database check is already treated. Without that rule, every environment without the optional
dependency starts returning `503` from a mounted endpoint.

Note the scope correction: `check_graphiti` **already exists** at `src/app/features/health/health_check.py:83-90`.
Only whatever `features/health/service.py:160` still misses belongs here; **`check_cognee` is change 4's**.

- [ ] Add the probe to both response shapes, additively.
- [ ] Report `not_configured` on absence without altering overall status.

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

- [ ] Give real types to the two `object`-typed parameters in the blast radius. A **third** `object` annotation is
      correct as written — it accepts genuinely unknown input — and is **left alone**.
- [ ] Fix `src/app/utils/embedding.py:5`, which raises on every embedding-dimension mismatch.

Disposition-ledger item 199 is corrected: the constructor it names was **already fixed**; the genuine residue is two
other parameters in the same feature.

**Proof**

```bash
uv run ty check src/ 2>&1 | tail -2      # count <= /tmp/c0-baseline/ty.txt
uv run python -c "import app.utils.embedding; print('import OK')"
```

## 12. Final gate

Every rung compared against the files from step 1, **none by exit code**.

- [ ] Run all seven gates; compare each to its baseline file.

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
