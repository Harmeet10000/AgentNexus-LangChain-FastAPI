# Review — cleanup-foundation (change 0)

**Reviewer:** fresh adversarial reviewer (did not author any artifact in this change).
**Date:** 2026-08-18.
**Artifacts reviewed:** `proposal.md` (108), `design.md` (460), `adrs.md` (145), `specs/**` (8 capabilities).
**Authoritative context read first:** `docs/relay/decisions.md` (D5.1, D5.2, D6.1, D11, D12, D14, D14.1–D14.4, D15),
`docs/relay/findings-database.md` §4 + §8, `docs/relay/findings-openspec-baseline.md`, `docs/relay/dispositions.md`
(change-0 section + "Fog still open and assigned").

**Verdict: CHANGES REQUESTED.** `tasks.md` must not be authored until F1–F5 are resolved.

> Status of this file: written incrementally. Sections are appended in the order the checks were run.

---

## 0. Validation, measured

```
$ openspec validate cleanup-foundation --type change --strict
Change 'cleanup-foundation' is valid
EXIT=0
```

```
$ openspec validate --all
✓ change/agent-tools-unification
✓ change/cleanup-foundation
✓ change/cognee-agent-memory
✓ change/cognee-saul-memory-migration
✗ spec/cognee-v1-api
✓ spec/datetime-utc-cleanup
✓ change/documents-unified-schema
✓ change/ingestion-pipeline-unification
✓ spec/llm-injection
✓ spec/mcp-context-di
✓ spec/mcp-directory-restructure
✓ spec/mcp-server-codemode
✓ spec/mcp-server-composition
✓ spec/mcp-server-pagination
✓ spec/mcp-server-prompts
✓ spec/mcp-server-resources
✓ spec/mcp-telemetry
✓ spec/mcp-testing
✗ change/mintlify-documentation
✗ spec/noqa-documentation
✓ spec/outbox-helper-extraction
✗ spec/pattern-matching-standard
✓ spec/session-required
✓ spec/settings-validation
✓ spec/test-mock-isolation
✗ spec/transactional-outbox
✗ spec/typed-exception-handling
Totals: 21 passed, 6 failed (27 items)
```

**HOLDS.** 6 failures, not 7. `cleanup-foundation` passes `--strict`. The failing set is byte-identical to the
recorded baseline (`mintlify-documentation` + the five specs). No 7th failure introduced.

---

## 1. Blocking findings, most severe first

### F1 — `outbox-helper-extraction`'s MODIFIED delta rests on a false code claim, and its normative body contradicts the change's own Non-Goal (BLOCKING)

`specs/outbox-helper-extraction/spec.md:11-18` states: *"the previous wording described an implementation that no
longer exists: it required the helper to build a new engine from the connection URL, bind a session to that engine,
and dispose of the engine in a `finally` block. The code was subsequently refactored to draw a session from the
application's existing session factory."*

**That implementation still exists.** `src/app/features/auth/service.py:481-524`:

```
492        if self._session_factory is not None:
493            async with self._session_factory() as session:
494                await with_outbox(...)
500            return
...
512        engine = create_async_engine(get_database_url())
513        try:
514            async with engine.begin() as conn:
515                session_ = AsyncSession(bind=conn)
516                await with_outbox(...)
523        finally:
524            await engine.dispose()
```

It is an `else` branch, and it is **reachable**: `src/app/features/auth/router.py:269` constructs
`AuthService(user_repo, token_repo)` with **no** `session_factory` (only
`src/app/features/auth/dependencies.py:43` passes one). So the OAuth-callback path uses the engine-per-call branch.

Consequences, all bad:

1. The delta's justification prose is factually false, and it inherited that error from
   `findings-openspec-baseline.md` §3, which read only `:481-500` and stopped one line before the `else`.
2. The delta's normative body — *"That helper SHALL obtain its session from the application's shared connection
   pool and MUST NOT construct a private connection resource per call"* — plus its first scenario's
   *"it SHALL NOT create, configure, or dispose of a connection resource of its own"* is **not satisfied by the
   code and is not made satisfied by this change**. `design.md` Non-Goals: *"**The second connection pool in the
   auth service.** It uses the right URL source but builds and disposes its own pool per operation. Deferred to
   change 1."* And `infrastructure-client-access`'s last requirement says such a site *"SHALL be recorded as an
   outstanding defect with a named owner"* — i.e. two capabilities in the same change say opposite things about
   the same lines.
3. The Migration Plan (steps 1-12) contains **no step** that touches `auth/service.py`.

So archiving this change deploys a *newly authored* false spec, which is precisely the disease
`design.md`'s closing section ("A spec can pass validation while being false") is written to diagnose.

**Fix:** either bring the engine branch's deletion into change 0 (it is ~20 lines and one constructor call at
`auth/router.py:269`), or restate the MODIFIED body to describe the actual two-branch helper and record the branch
removal as change 1's, deleting the false "no longer exists" paragraph either way.

---

### F2 — `transactional-outbox`'s ADDED requirement mandates behaviour the change explicitly declares out of scope, with no implementing step and a collision with an accepted spec (BLOCKING)

`specs/transactional-outbox/spec.md:59-96` adds *"A missing outbox relation SHALL fail loudly rather than silently
disabling the outbox"*, requiring: error-severity reporting distinguishable from a transient connection failure;
the missing relation named; the outbox subsystem recorded as unavailable *"in a form the readiness surface can
observe"*; detached-listener termination reported at error severity and recorded; and — the heaviest one —
*"the state change that the event was meant to accompany SHALL NOT remain committed without its event"*.

Every one of those is a change to `src/app/shared/outbox/relay.py:66` / `:81` and to the transaction boundary of
`auth/service.py:_publish_outbox_event`. The change disclaims all of it:

- `proposal.md:105-108`: *"Boot survives by accident, through an `except` that any tightening pass would remove —
  which is why narrowing that handler is sequenced *after* the relations exist and **is not attempted here**."*
- `design.md:85-89` Non-Goals: *"**Narrowing the outbox relay's catch-all handlers.** Sequenced deliberately…
  narrowing it is a requirement change that belongs with the pass that performs it."*
- Migration Plan steps 1-12: no step for it.

Two further problems:

- **It collides with an accepted spec this change chooses not to modify.**
  `openspec/specs/typed-exception-handling/spec.md:253` — *"Requirement: Degradation boundaries SHALL keep
  `except Exception` with `add_note`"* — carries scenario `:278` *"Outbox relay dead-letters on any failure"*.
  `design.md:87-89` correctly identifies that this capability *sanctions* the broad catch, then declines to touch
  it. After archive, two accepted specs disagree about `relay.py:66`.
- **`design.md:272-274` (D-9) says no `transactional-outbox` delta exists at all**: *"This change makes reality
  match that requirement rather than restating it, **so no delta is added there**."* The change ships a
  96-line delta with two MODIFIED and one ADDED requirement. The author lost track of their own scope.

Note the flip side, which does hold: **check #7 passes.** Nothing in this change tightens `relay.py:66` before the
tables exist; the sequencing statements are correct and appear in three places. The defect is that the spec
creates an obligation the change will not discharge.

**Fix:** move the ADDED requirement to the change that performs the narrowing (with the paired
`typed-exception-handling` MODIFIED that retires the sanctioning scenario), or bring the narrowing into change 0
ordered strictly after step 4 — and delete the D-9 sentence either way.

---

### F3 — the DDL requirements never name an index, so the BM25 half of the repair is illusory (BLOCKING)

The repo's BM25 SQL passes the **index name as a literal argument**, so it is part of the query contract, not a
naming convention:

```
src/app/features/search/repository.py:415,417,419,430,432,433
    c.content <@> to_bm25query(:query, 'search_chunks_bm25_idx')
src/app/features/search/repository.py:356,361,362
    search_text <@> to_bm25query(:query_text, 'clauses_bm25_idx')
src/app/features/search/constants.py:15
    SEARCH_CHUNKS_BM25_INDEX_NAME = "search_chunks_bm25_idx"
```

`grep -rn "bm25\|_idx" openspec/changes/cleanup-foundation` returns **three hits, none of them a requirement**:
`design.md:11`, `:13` (Context observations) and `:102` (the F8 Open Question). `migration-chain-integrity` refers
only to *"their vector, keyword and fuzzy retrieval indexes"* — abstract throughout. An implementation that creates
a correctly-defined `USING bm25` index under any other name satisfies every scenario in the change and leaves
`to_bm25query(..., 'search_chunks_bm25_idx')` failing exactly as it fails today. This is the identical
"illusory repair" shape the change correctly guards against for identity-vs-outbox (see §2, check 6) — and it is
unguarded here.

Related, and now stale: **F8 is closed** (`findings-database.md` §10 — `pg_textsearch` 1.3.0 installed, access
method literally `bm25`, opclasses `text_bm25_ops` / `text_array_bm25_ops`, and **no `bm25` index exists anywhere
in the database**). So `design.md:102-105` (Non-Goal), `design.md:278-286` (Risk 1) and `design.md:366-368`
(Open Question 3) all still describe F8 as open and assign it to change 1's step 0. They must be restated as
closed, and Risk 1's hostage scenario re-aimed (see F4d, which is the live version of it).

**Fix:** add a requirement to `migration-chain-integrity` (or wherever the DDL lives) that the keyword index is
created under the **exact** name the query text pins, that the name is a contract with `search/constants.py:15`
and is not renameable, and state explicitly whether `clauses_bm25_idx` survives change 2's retarget or is
deliberately not created.

---

### F4 — four false or self-contradicting factual claims, three of them inside the section written to fix exactly this defect class (BLOCKING)

**(a) `design.md:386-388` — measured false.** *"A from-base offline render does not complete at all:
`9f4a1b7c6d2e` alters the phantom `clauses` relation, so the render aborts before reaching the revisions the proof
was interested in. Whatever output was imagined for this command, it was never produced."*

Measured, just now, read-only, no database touched (alembic offline mode uses a dummy URL — `src/alembic/env.py:45`
`run_migrations_offline`):

```
$ uv run alembic upgrade heads --sql   → EXIT=0, 697 lines of SQL, ends with COMMIT;
   Running upgrade  -> c0c17c6eb1cc … -> 2bc7726317f6 -> a71f0d7d9c12
   Running upgrade 2bc7726317f6 -> 8a7d9b1c2e3f -> 9f4a1b7c6d2e -> 0001 -> 0002 -> 0003 -> 0004
$ grep -c "CREATE TABLE search_"  → 2
$ grep -n clauses                 → 237-255 (ALTER TABLE clauses ADD COLUMN …, UPDATE clauses SET …)
```

The render **completes**, and it emits the `clauses` ALTERs verbatim. Offline mode renders DDL as *text*; it never
executes it, so a non-existent relation is irrelevant to it. The `= 2` half of D14.3 is correct and remains
correct; the "never produced / aborts" reasoning appended to it is invented. This is the same sin D14.3's own
"Lesson worth keeping" names — an unexecuted claim about a command, written down as fact — committed inside the
correction that names it.

(`alembic upgrade head --sql` does fail today, but with `Multiple head revisions are present for given argument
'head'` — a two-heads error, not a `clauses` error. Post-merge it will succeed.)

**(b) `design.md:396` — false self-audit.** *"No wording in this `design.md`, in `proposal.md`, or in any spec file
in this change carries the from-base form."* Two do:
- `specs/migration-chain-integrity/spec.md:19-23`, scenario *"Rendering an upgrade to head"* — *"WHEN an upgrade to
  head is rendered without applying it THEN the render SHALL succeed"*. That is a from-base render.
- `design.md:335`, Migration Plan step 4 — *"Verify by rendering the upgrade offline"*.

Both happen to be *achievable* (per (a)), so they are not themselves defects — but the sweep the design claims to
have performed was not performed, and a reviewer relying on that sentence would stop looking.

**(c) `design.md:390-394` — misdescribes its own spec, leaving D14.3's binding re-scope with no home.**
The design says the re-scoped property is *"exactly the requirement stated in
`specs/migration-chain-integrity/spec.md` under *The authoritative revision SHALL NOT claim relations an earlier
revision already claims*"*. Read that requirement (`spec.md:105-115`): its body is about **reversal** —
*"Reversing the authoritative revision SHALL NOT remove relations whose creation an earlier revision also
claims"* — and its single scenario is *"Reversing the authoritative revision → the event-outbox relations SHALL be
left in place"*. It says nothing about what the revision **creates**. D14.3 is binding on change 0
(*"Change 0 must scope its equivalent check the same way"*) and change 0 has **no** requirement or proof asserting
that the authoritative revision's own rendering creates no `search_` relation. Check #2 therefore FAILS.

**(d) `design.md:419-421` — false, and it hides the real hazard.** *"The `diskann` index that exists in the live
database today works only because `vectorscale` happens to be pre-installed … **no revision in the chain has ever
created it**."* `src/alembic/versions/8a7d9b1c2e3f_add_search_documents_and_chunks.py:26` creates it:
`op.execute("CREATE EXTENSION IF NOT EXISTS vectorscale")` (visible at render line 154).

The true and far more useful statement, which the change never makes: **`a71f0d7d9c12` — the unstamped head that
*will* execute on the next `upgrade` — creates a `diskann` index without creating `vectorscale`.**

```
a71f0d7d9c12:23-26   uuid-ossp, vector, pg_trgm, pg_textsearch      ← no vectorscale
a71f0d7d9c12:99-101  CREATE INDEX chunks_embedding_idx ON chunks USING diskann (embedding vector_cosine_ops)
(also c0c17c6eb1cc:88, same shape)
```

On any instance without ambient `vectorscale`, that revision fails — **before** the authoritative revision's
outbox repair runs. That is Risk 1's "outbox repair held hostage by an unrelated index" scenario, live and
concrete, and the change aims Risk 1 at the (now closed) bm25 question instead. `design.md`'s
*"no revision in the chain ever creates it"* framing is what caused the miss.

---

### F5 — `design.md:436-438` asserts the archive turns `transactional-outbox` green; it will not (BLOCKING as a stated expectation)

*"The fifth, `transactional-outbox`, fails for a different reason — no requirement body carries SHALL or MUST. That
one *is* reachable, and this change's `specs/transactional-outbox/spec.md` delta fixes it: the modified requirement
bodies supply the normative keyword, and at archive time the merged spec will validate."*

`openspec/specs/transactional-outbox/spec.md` has **six** requirements and the validator emits
`6 × Requirement "<name>" must contain SHALL or MUST`. The delta MODIFIES **two** of them — `Outbox Table Schema`
and `Migration`. The other four keep bodies that are pure `WHEN`/`THEN` with no normative keyword:

```
### Requirement: Outbox Helper      (:11)  → "an outbox row is inserted and pg_notify fires"
### Requirement: Relay Process      (:16)  → "the relay publishes the event or dead-letters after 5 failures"
### Requirement: Relay Lifecycle    (:21)  → "the relay starts after deps are ready and drains on shutdown"
### Requirement: Dead Letter        (:26)  → "the event is moved to dead_letter_events"
```

So the merged spec still fails on four counts. The **acceptance criterion is unaffected** (baseline stays at 6
either way, and this change adds no failure), but the expectation as written is wrong and would be discovered at
archive time by whoever expected 5.

---

## 2. Non-blocking findings

### F6 — the "third URL flavour" does not exist; the corrected count is two (MAJOR)

`adrs.md:104` titles ADR-3 *"One connection-URL accessor with three driver flavours"*, `:110-112` claims
*"Three distinct consumers need the database connection string… embedded third-party components need whatever form
that component's own driver expects"*, `design.md` D-6 says *"There are three flavours to serve, not two"*, and
`specs/infrastructure-client-access/spec.md` encodes it as scenario **"An embedded third-party component"**.

Cognee — the only candidate for the third flavour — **is not a URL consumer**:

```
.venv/…/cognee/infrastructure/databases/relational/config.py:12-23   RelationalConfig exposes discrete fields
                                                                     (db_host, db_port, db_username, db_password,
                                                                      db_name, db_provider) — no URL field
.venv/…/relational/config.py:73-79                                   to_dict() returns those same seven keys
src/app/connections/cognee_client.py:91-101                          already passes the discrete fields
src/app/connections/cognee_client.py:111                             `postgres_url` sits in a separate else-branch
                                                                     dict that is only returned, never consumed
```

So the accessor has **two** flavours: SQLAlchemy+asyncpg and plain libpq/psycopg. A spec scenario that requires a
flavour for a consumer that takes discrete fields will produce dead API surface, and ADR-3 — an artifact explicitly
written to *"outlive this change"* and constrain *"every future consumer"* — will constrain them wrongly.

What the ADR gets **right**, and which I verified: the double-strip accounting. `src/app/lifecycle/lifespan.py:124`
strips `+asyncpg` and `src/app/shared/outbox/relay.py:71-72` strips it again from the already-stripped value; the
second is a silent no-op. Check #4's substance therefore holds on two of three counts and fails on the flavour count.

**Fix:** retitle ADR-3 to two flavours, replace the "embedded third-party component" scenario with a
discrete-fields requirement (the accessor exposes host/port/user/password/database as separate values for consumers
that assemble their own connection), and delete D-6's "three not two".

### F7 — D-3 undercounts the private registry six-to-two, and harvesting two zero-importer models contradicts D-2 (MAJOR)

`design.md` D-3 states *"Two models — the parent-document and clause models — are declared against a private
registry"*. `src/database/schemas/memory_schema.py:51` declares `class Base(DeclarativeBase)` and it carries **six**:

```
:55  Entity          :108 Relationship    :154 ParentDocument
:190 Clause          :247 Event           :272 MemoryVersion
```

The file is 302 lines with **zero importers repo-wide** (`grep -rn "memory_schema" src/ tests/` → the file itself
only). Two consequences the change does not confront:

1. `specs/orm-metadata-registration/spec.md`'s enumeration scenario requires the registry's models to be
   enumerated, but the design's own enumeration is wrong, so an implementer following D-3 will move two and silently
   leave four — including `Event`, whose name collides conceptually with the outbox event this change is repairing.
2. Moving zero-importer models into the shared registry makes them autogenerate-visible, i.e. it schedules DDL for
   relations no code reads. That is **exactly** what D-2 refuses elsewhere (*"DDL without a reader"* is D-2's stated
   ground for not creating `parent_documents`/`clauses`). The change applies the principle in one decision and
   violates it in the next.

**Fix:** state all six models, and decide explicitly per model whether it joins the shared registry or the file is
deleted along with the other nine dead trees. Deleting it is the option consistent with D-2.

### F8 — "live read or write path" is undefined, and read literally it commits the change to relations D-2 refuses to create (MAJOR)

`specs/migration-chain-integrity/spec.md:69-73` — *"Upgrading the deployed database to head SHALL leave no live read
or write path pointing at a relation that does not exist"* — with scenario `:88-91` *"the relations named by live
read and write paths are compared against the upgraded database → every such relation SHALL be present"*.

Nothing in the change defines "live". Taken at face value, mounted-and-reachable code names at least:
`search_documents`, `search_chunks` (`src/app/features/search/repository.py`), `clauses` (`:337,:356,:383`) and
`parent_documents`. D-2 explicitly declines to create the last two. So the requirement as written is unsatisfiable
by the change that adds it, and a conforming implementer must either create relations D-2 forbids or invent their
own definition of "live".

**Fix:** define "live" precisely — e.g. *reachable from a mounted route through a code path that is not itself
scheduled for deletion in changes 0-4* — and enumerate the relations it resolves to today, with `clauses` /
`parent_documents` named as deliberately excluded and why.

### F9 — the test-fixture Open Question rests on a number that is false today (MODERATE)

`design.md:359-360`: *"There is no working test client fixture — thirteen collection errors involve it."* Measured:

```
$ uv run pytest --collect-only -q  → 90 tests collected, 0 errors
```

The conclusion the Open Question reaches (that end-to-end verification of the repaired endpoints will need a fixture
that does not exist) may still be right, but its stated evidence is stale and a reader will lose confidence in the
surrounding paragraph. The same paragraph's companion figure is also off: the ruff baseline is **123**, of which
`todo_temp` contributes 3 — so step 2's deletion moves 123 → 120. The direction is intact; the numbers are not.

### F10 — enumeration mismatch in the proposal (MINOR)

`proposal.md` says *"**Nine** proven-dead trees and files are deleted"* and then enumerates **eight** groups (an
unparseable 783-line draft, an inverted 36-line parser, three zero-byte package trees, two zero-byte feature
packages, and the 1,129-line reconciliation subsystem). Either the ninth is missing from the list or the count is
wrong; a deletion manifest is the last place to leave that ambiguous, since the manifest is what the implementer
executes.

---

## 3. Check-by-check verdicts

| # | Check | Verdict | Where |
|---|---|---|---|
| 1 | D14.1 scope: outbox **and** document schema, outbox **ordered first**, ordering *encoded* | **HOLDS** | `migration-chain-integrity:63-67` scenario *"Ordering within the revision"* states the outbox relations are created **before** the document schema **and gives the reason** (a failure in the larger half cannot block the endpoint repair). `:31-40` requires one authoritative revision covering both halves. Ordering is normative, not incidental. |
| 2 | D14.3's re-scoped proof: no surviving `--sql` / `upgrade head` / `grep -c` claim that is false or unexecutable | **FAILS** | F4(a) false claim, F4(b) false self-audit, F4(c) the re-scope has no requirement to live in. Grep of the whole change for `--sql`, `upgrade head`, `grep -c` returns `design.md:335`, `:377-396` and `migration-chain-integrity:19-23`; the executable ones are fine, the narrative ones are wrong. |
| 3 | D14.4: four extensions named explicitly; `diskann` pinned to `vectorscale` | **PARTIAL** | Names all four explicitly and requires each *"created before the first index… that requires it"* (`:142-148`) — this satisfies D14.4's letter. But the pin is only *implied* (no index is ever named — see F3), and it is not applied to the one revision where it bites: `a71f0d7d9c12` builds a `diskann` index without `vectorscale` (F4d). |
| 4 | URL flavours covered; double-strip accounted for | **PARTIAL** | Both real flavours covered, `infrastructure-client-access` correctly adds percent-encoding and pool-ownership and *"No flavour is derived at the call site"*; double-strip correctly located. The third flavour is fictional — F6. |
| 5 | MODIFIED deltas reproduce scenario titles verbatim | **HOLDS** | Byte-compared every MODIFIED block against the deployed spec, title by title. `transactional-outbox` (`Outbox Table Schema`, `Migration`), `outbox-helper-extraction` (3 titles), and `typed-exception-handling` (which the brief did not list, and which also checks out) are all exact, including backticks and underscores. `--strict` agrees. The author's earlier `--strict` failure is genuinely fixed. |
| 6 | Identity repair bound to the outbox relations existing | **HOLDS** — strongest guard in the change | `request-identity-from-token:59-71`, *"Repairing identity SHALL leave the repaired endpoints working end to end"*, requires the repaired endpoint to return its normal response **and** the event row to be recorded — so a fix that only moves the 500 from the dependency layer to the outbox INSERT is non-conforming. Reinforced by D-4 and by Migration Plan step 8 sitting after step 4. An implementer cannot ship the illusory repair and pass. |
| 7 | Exception tightening sequenced after the tables exist | **HOLDS** (see F2 for the flip side) | Nothing in the change tightens `relay.py:66`. Stated three times: `proposal.md` Risks, `design.md` Non-Goals, `design.md` D-9's neighbourhood. The `except (PostgresError, Exception)` that keeps the app booting survives change 0 intact. |
| 8 | Deletion-manifest safety | **HOLDS** | All four couplings verified against source: `src/app/tasks/__init__.py:6-9` imports + `:18-20` re-exports; `src/app/features/__init__.py:3,8,9`; `profile/router.py:29,30` reads `app.state.storage` / `app.state.mongodb` while lifespan publishes `object_store` / `db` (a real latent `AttributeError`, correctly identified); and D6.1's `shared/agents/**` shadow — `registry.py:40-45` → `precedent_tools.py:21-22` / `get_obligation_chain.py:29` → the 30-byte `memory_scope.py`, which raises `ImportError` at import if change 0 deletes before change 3 rewrites importers. The change orders change 3 first and says why. |
| 9 | Honesty about the stamped-but-unapplied chain | **HOLDS** — best-argued part of the change | ADR-1 states the chain is *"permanently misleading"* as a **permanent consequence**, not a temporary state; forbids `downgrade` past the joined head in the imperative (*"Not discouraged — forbidden"*); records that autogenerate must be treated as unreliable on the affected relations; names the hollow revisions (`0001_add_outbox_tables`, `8a7d9b1c2e3f`, `9f4a1b7c6d2e`); and `migration-chain-integrity:117-135` turns the fresh-environment procedure into a normative requirement that must name the skipped revisions. Both rejected alternatives are attributed to the user and reasoned. |
| 10 | `--strict` clean and no 7th `--all` failure | **HOLDS** | §0: `--strict` valid, totals 21 passed / 6 failed, failing set byte-identical to baseline. |
| 11 | ≥15 concrete code claims audited | **DONE — 35 audited, 7 wrong** | §4. |

---

## 4. Code-claim audit — 35 claims checked, 7 wrong

Wrong (all cited above): `outbox-helper-extraction`'s "implementation that no longer exists" (F1);
`design.md:386-388` render aborts (F4a); `:396` no from-base wording (F4b); `:393` the cited requirement's subject
(F4c); `:419-421` no revision creates `vectorscale` (F4d); `:436-438` archive turns the spec green (F5); D-3's
two-model count (F7). Plus three stale figures: "thirteen collection errors" (F9), the ruff baseline, and
"nine" deletions (F10). Plus one scope-count error: three flavours (F6).

Correct, verified at the cited lines — a genuinely high hit rate on the mechanical claims:

- `src/app/connections/postgres.py:30-71` — all three latent defects real: `f"{parsed.username}:{password}@{parsed.hostname}"` performs no percent-encoding; `settings.POSTGRES_PASSWORD.get_secret_value() != "pass"` is a hard-coded sentinel; and `if parsed.port: netloc += f":{parsed.port}"` duplicates the port on the `else parsed.netloc` branch.
- `src/app/features/auth/router.py:269` — `AuthService(user_repo, token_repo)`, no session factory. `src/app/features/auth/dependencies.py:43` — passes one. Both exactly as claimed.
- `src/alembic/versions/a71f0d7d9c12_…py:23-26` — creates `uuid-ossp`, `vector`, `pg_trgm`, `pg_textsearch`; `:96-98` `chunks_bm25_idx … USING bm25(search_text)`; `:99-101` `chunks_embedding_idx … USING diskann`; `:102-104` `chunks_search_text_trgm_idx … gin_trgm_ops`. `documents` carries `updated_at` (nullable=False, **no** server_default); `chunks` carries **none** — which is precisely what `migration-chain-integrity:93-103` is for. Claim correct and well-aimed.
- `src/alembic/versions/8a7d9b1c2e3f_…py:26` — `CREATE EXTENSION IF NOT EXISTS vectorscale`. (Correct as code; it is what falsifies F4d's narrative.)
- `src/alembic/env.py:23-36` then `:39-42` — the billing and `app.shared.outbox.model` imports sit **above** the `try: target_metadata = Base.metadata / except ImportError:`, so that handler is unreachable. Claim correct.
- `src/app/shared/outbox/relay.py:66` `except (PostgresError, Exception)`; `:71-72` the second `.replace("+asyncpg","")`. Correct.
- `src/app/shared/outbox/relay.py` has **no** `shutdown` and no `_running` — so `session-required`'s third requirement is already satisfied and correctly needs no delta.
- `src/app/features/search/constants.py:15` and the nine `to_bm25query` sites. Correct.
- `openspec/specs/transactional-outbox/spec.md` — 6 requirements, every body pure WHEN/THEN, Purpose is the `TBD - created by archiving change celery-outbox-idempotency` stub. Correct (and see F5).
- `src/database/schemas/memory_schema.py:51` private `Base`, zero importers. Correct except the count (F7).
- `src/app/tasks/__init__.py:6-9,18-20`, `src/app/features/__init__.py:3,8,9`, `profile/router.py:29,30`, `registry.py:40-45`, `precedent_tools.py:21-22`, `get_obligation_chain.py:29`, 30-byte `memory_scope.py`. All correct.
- `src/app/connections/cognee_client.py:91-101` discrete-field call. Correct as code (and see F6).

---

## 5. Are the ADRs sufficient for an implementer to not get it wrong?

**Partly — and the two decisions most likely to be got wrong have no ADR at all.**

- **ADR-1 — sufficient.** It is the strongest artifact in the change. An implementer reading it will not edit a
  stamped revision, will not squash, will not offer a downgrade, and will know the catalog is the only authority.
  One correction to fold in: `:24-25`'s *"the history cannot be rendered offline from base at all"* is false (F4a),
  and it is load-bearing for the ADR's Context. Remove it — the ADR's decision survives without it.
- **ADR-2 — sufficient.** The decision, the two rejected alternatives, and the "closes a class, not an instance"
  consequence are all stated well enough that a future endpoint author cannot reach for mutable request state
  innocently. It should additionally name the 500 → 401 behaviour change on the affected endpoints as a
  breaking-change consequence, since that is what a caller will observe.
- **ADR-3 — not sufficient; factually wrong on its central count.** Three flavours where there are two (F6). An
  implementer who builds the third will ship dead surface, and — worse for an ADR meant to bind future work — the
  next person adding a consumer will look for a URL flavour when what they need is discrete fields.
- **Missing ADR: index names are a query contract.** `to_bm25query(:q, 'search_chunks_bm25_idx')` embeds the index
  name as a string literal in SQL. That makes index naming a cross-layer contract with `search/constants.py:15`,
  not a migration-local convention, and nothing in the change says so (F3). This is the single most likely thing an
  implementer gets wrong, because every other index in the codebase *is* freely renameable.
- **Missing ADR: what the outbox relay owes when a relation is absent.** The change simultaneously requires
  loud failure (`transactional-outbox` ADDED) and preserves the catch-all that guarantees silent failure
  (`typed-exception-handling`, untouched), and defers the reconciliation to an unnamed later pass (F2). An
  implementer has no way to know which spec wins. Either an ADR resolves it or the ADDED requirement leaves
  change 0.

---

## 6. Gate

`tasks.md` must not be authored yet. Blocking: **F1, F2, F3, F4, F5**. F1 and F2 are the serious ones — both would
archive a *newly written* false or unsatisfiable requirement into `openspec/specs/`, which is the exact failure mode
this change exists to end. F3 would let a conforming implementation leave BM25 as broken as it is today. F4 and F5
are factual repairs to `design.md` and cost nothing but attention.

F6-F10 should be fixed in the same pass since they are all single-paragraph edits, but none of them would cause an
implementer to build the wrong thing — except F6, which will cause them to build a thing nobody needs.


---

## Author response

Every finding was re-verified by measurement before acting. **Nothing was refuted** — all ten held, and an eleventh
(F11) was found by the coordinator cross-checking my delta against the deployed spec. Where a finding was right about
the defect but wrong about the underlying fact, both corrections are recorded, because acting on a right conclusion
from a wrong premise is how the next error gets made.

| # | Disposition | What changed, and where |
|---|---|---|
| **F1** | **Accepted — fixed** | Confirmed at `auth/service.py:505-528`: the `else` branch does `create_async_engine(get_database_url())` with `await engine.dispose()` in a `finally`, reachable from `auth/router.py:269` with no factory supplied. `specs/outbox-helper-extraction/spec.md` rewritten: the "no longer exists" claim is withdrawn, the requirement now states the property that *is* true, states plainly that the engine-per-call fallback exists and that this change does not remove it, and carries the residue as a named outstanding defect against `infrastructure-client-access`'s *Connection pools SHALL be owned by the startup sequence* with the connection-plumbing change as owner. All three scenario titles kept verbatim. `design.md` § *A spec can pass validation while being false* had its premise **inverted** and was rewritten. |
| **F2** | **Accepted — fixed, branch (b)** | The ADDED requirement *A missing outbox relation SHALL fail loudly* is **withdrawn** from `specs/transactional-outbox/spec.md`, replaced by an in-file comment recording the withdrawal and its grounds: no implementing step, and a direct collision with an accepted `typed-exception-handling` requirement that sanctions the relay's broad catch. The gap is a Non-Goal in `design.md`, and **ADR-5** decides precedence (the sanctioning requirement wins until the narrowing change ships relations, code and a paired MODIFIED together). **D-9's false "no delta is added there"** is corrected in full — it now enumerates the two MODIFIED requirements the change actually ships. |
| **F3** | **Accepted — fixed** | `migration-chain-integrity` gains *Retrieval indexes SHALL be created under the exact names the query text names*, four scenarios. Encodes: the index name is a literal SQL argument to the two-argument `to_bm25query`, pinned at `search/constants.py:15`; an index of correct shape under a different name matches nothing and raises nothing; this change creates `chunks_bm25_idx`, `chunks_embedding_idx` and `chunks_search_text_trgm_idx` under the exact names `a71f0d7d9c12` already uses so the two converge; it creates neither `search_chunks_bm25_idx` nor `clauses_bm25_idx` because their relations are not created; and it **SHALL NOT edit the query literals** — `clauses_bm25_idx` has four readers (`search/repository.py:356,361,362` and `ingestion_kb/nodes.py:751`) and **change 2 owns retargeting them**. Named as coordination, not collision. **ADR-4** records the durable rule. |
| **F4(a)** | **Accepted — fixed, and the over-correction avoided** | Measured: `alembic upgrade heads --sql` → exit **0**, 697 lines, one `COMMIT;`, `clauses` ALTERs present. It does **not** abort. `design.md` D14.3's "the render does not complete at all" is withdrawn. The true property is stated once — offline rendering is **from base regardless of live state**, because there is no database to read `alembic_version` from — and both invalid proof families are named: any proof depending on offline SQL being *incremental*, and any depending on it *aborting*. |
| **F4(b)** | **Accepted — fixed** | The self-audit sentence claiming no from-base wording survived anywhere is corrected to a checkable statement rather than a claim about all documents. |
| **F4(c)** | **Accepted — fixed** | The re-scoped proof had nowhere to live: `design.md:393` pointed at a requirement about **reversal**. Added *The authoritative revision's own rendering SHALL NOT create a relation an earlier revision creates*, whose second scenario also records that the from-base form is to be rejected as unmeasurable. Executable mechanism found and run: `alembic upgrade 2bc7726317f6:a71f0d7d9c12 --sql` → exit 0, 80 lines, 2 `CREATE TABLE` — a **range** render, which does work offline and emits only that range. |
| **F4(d)** | **Accepted — hazard relocated, not merely reworded** | `8a7d9b1c2e3f:26` **does** create `vectorscale`; "no revision in the chain has ever created it" is false and is withdrawn. The real hazard is narrower and worse: `a71f0d7d9c12` builds `diskann` indexes at `:97,100,103` and creates **no** extension; `8a7d9b1c2e3f` sits on the **other** side of the `2bc7726317f6` branch point so it is not an ancestor and cannot supply the dependency, and on the deployed database it was stamped rather than executed. `a71f0d7d9c12` is ordered **ahead** of the merge and the repair, so **the authoritative revision cannot fix it** — a failure aborts before the outbox repair. Encoded as two requirements plus a re-aimed Risk; the extension is a **precondition of the upgrade**, not a satisfied dependency. |
| **F5** | **Accepted — fixed, claim withdrawn** | Measured: all **six** `transactional-outbox` requirement bodies are non-normative. The delta modifies **two**, so **four** stay non-normative and the capability stays red after archive. The "at archive time the merged spec will validate" claim is withdrawn. Also recorded: the four `## Purpose` failures are **unreachable by any change**, because no delta section emits a `## Purpose` header — they need a direct spec edit. The sixth failure, `change/mintlify-documentation`, is named so the count of six is fully attributed. |
| **F6** | **Accepted — fixed in all three places** | The third URL flavour is fictional. Removed from the spec scenario (replaced by *The set of flavours is closed at two, and a third is not invented*), from **D-6** (fully rewritten), and from **ADR-3** — which was not salvageable as written and is **rewritten** around two flavours plus **discrete fields** for the embedded component that accepts no DSN, with a Status noting it supersedes the three-flavour draft and a method note that counts do not carry their own evidence. A new requirement covers the discrete-field consumer. |
| **F7** | **Accepted — fixed, and the resolution changed the plan** | The private registry carries **six** models (`memory_schema.py:51,55,108,151,187,247,272`), not two, and has **zero** importers. Harvesting them would contradict D-2 by scheduling creation of relations nothing reads — the mirror of the defect this change exists to close. **D-3 is rewritten**: the registry is retired by **deletion, not harvest**, which also removes the ordering constraint that harvest imposed and moots a Risk bullet. The module sits inside the reconciliation deletion group, so it costs no extra commit. |
| **F8** | **Accepted — fixed with a normative definition** | "Live" is now defined in `migration-chain-integrity`: *named by a code path reachable from a route mounted on a published API version, through code that is not itself scheduled for deletion or for retargeting by the sequenced changes.* The requirement enumerates the in-set relations and the deliberately-excluded ones **with the change that owns each**, and gains a scenario for a relation named only by a path a later change retargets. The `Goals` bullet now points at that definition instead of leaving "live" to the reader. |
| **F9** | **Accepted — figure withdrawn, decision re-made on merits** | Measured: `pytest --collect-only -q` → **90 tests, zero collection errors**. The "thirteen collection errors" figure is withdrawn. It was mislabelled **in kind, not in magnitude**: `pytest --no-cov -q` → *22 failed, 55 passed, 13 errors*, and all thirteen are **setup** errors, `fixture 'client' not found`, in `tests/integration/test_health.py` and `test_api_deprecation.py`. Re-decided in favour of the **direct probe** on three real grounds: no `client` fixture exists anywhere, so an endpoint-level test needs test infrastructure this change does not own; the coverage gate (`--cov-fail-under=80` against **22.16%**) makes the runner's exit code unusable as a proof for *any* task here; and the automated version is recorded as follow-up with an owner. The stale lint figure was corrected too — measured **123**, not 120. |
| **F10** | **Accepted — fixed** | `proposal.md` now says **seven** and enumerates seven; `design.md` D-8 enumerates the same seven in the same terms, states that four carry coupled edits, and names all four couplings. |
| **F11** | **Accepted — fixed, after one insufficient attempt** | Confirmed against `openspec/specs/typed-exception-handling/spec.md:149`. The first fix disclosed the omission in prose inside the requirement body — honest, but **still deleted the scenario on archive**, because prose is not a mechanism. Now fixed properly: **all six scenarios are reproduced verbatim in the accepted order**, `Reconciliation fetch failure catches PostgresError` first. `## REMOVED Requirements` was rejected as the alternative because it operates at requirement granularity and would retire the entire asyncpg guarantee; there is no scenario-level REMOVED. Retiring the one stale scenario is routed to the spec-hygiene pass that also owns the four `## Purpose` failures. The three-claimant lane split is unchanged and still has no title overlap. |

### Two things the review did not ask for, recorded because they change what an implementer does

**A route decision was missing entirely, and it is now ADR-6.** Measured read-only: `9f4a1b7c6d2e` creates
`parent_documents` but operates throughout on **`clauses`**, and **no revision creates `clauses` and no model declares
it** — it appears in exactly one file in the versions directory, the revision that mutates it. So that revision is
**unrunnable against any database**, which is *why* the stamp happened; the proof it never executed is that it is
marked applied while `parent_documents` is absent. This eliminates the rewind-and-re-upgrade repair route — it does not
terminate — leaving the forward idempotent revision as the only route. Resolving `clauses` is **not** a migration
question: it belongs to change 2 (item 184, Option A+, retarget the readers), and this change ships no `clauses` DDL.

**A cross-change correction I cannot make myself.** `openspec/changes/documents-unified-schema/tasks.md` states that
"A from-base render cannot even complete, because `9f4a1b7c6d2e:103` alters the phantom `clauses` relation." That is
the same error as F4(a) and it is measurably false. I did not edit it — it is outside this change's directory. Routed
to the coordinator.

### Validation

`openspec validate cleanup-foundation --type change --strict` → **valid, exit 0**.
`openspec validate --all` → **21 passed, 6 failed (27 items)** — the pre-existing set, unchanged, with
`change/cleanup-foundation` among the passes. No seventh failure was added.
