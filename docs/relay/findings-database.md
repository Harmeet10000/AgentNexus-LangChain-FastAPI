# Findings — live database probe (2026-08-17, orchestrator)

Established by connecting to the **actual** database with `asyncpg`, read-only (`pg_available_extensions`,
`pg_tables`, `to_regclass`, `count(*)`). Probe scripts: `/tmp/dbprobe.py`, `/tmp/dbprobe2.py`.
No DDL, no writes. These findings supersede every guess about the DB in the scout and plan reports.

---

## §1 — The database is Timescale **Cloud**, not the local compose service

`.env.development` `POSTGRES_URL` resolves to:

```
host = qbid1qrc75.nnro3dh8tf.tsdb.cloud.timescale.com
port = 39662        db = tsdb        user = tsdbadmin        sslmode = require
server_version = PostgreSQL 18.0.4
```

`docker-compose.yml`'s `timescale` service (`timescale/timescaledb-ha:pg18`) is therefore **not** what the app
talks to in development. Every "check the image" plan step is answering the wrong question — the extension set
that matters is the one the **managed cloud instance** exposes, and we do not control it.

Nothing is listening on `localhost:5432`, so the compose Postgres has never been up in this working copy either.

## §2 — `POSTGRES_URL` carries **no password**. Auth works only by environment side effect.

Password fingerprints (SHA-256 prefix, values never printed):

| Source | Fingerprint |
|---|---|
| password embedded in `POSTGRES_URL` | **`<unset>`** — there is none |
| `PGPASSWORD` | `6eaa139d0da2` (len 16) |
| `POSTGRES_PASSWORD` | `6eaa139d0da2` (len 16) — identical |

Connecting with the DSN as written fails: `InvalidPasswordError: password authentication failed for user
"tsdbadmin"`. Connecting with the `PG*` variables succeeds.

**Consequence — narrower than it first appears, because one consumer repairs the URL and the others do not.**

`src/app/connections/postgres.py:30-71` `get_database_url()` handles both problems deliberately:

- `:42-47` rewrites `postgres://` → `postgresql+asyncpg://` (and `:36-41` handles `postgresql://`), so the
  SQLAlchemy 2 dialect-alias removal is already accounted for.
- `:51-54` strips `?sslmode=require` / `&channel_binding=require` — correct, since those are libpq parameters
  asyncpg rejects as query args.
- `:57-70` **injects the password explicitly**: `if not parsed.password and
  settings.POSTGRES_PASSWORD.get_secret_value() != "pass"`, it rebuilds the netloc with the secret.

So the main engine path at `:80` works, and the earlier claim that auth depends on a `PGPASSWORD` environment
side channel is **wrong for that path**. My probe failed only because it used the raw DSN, which is what
`get_database_url()` exists to avoid.

**The defect is that two consumers bypass `get_database_url()` and use the raw, passwordless value:**

| Site | Uses | Result |
|---|---|---|
| `connections/postgres.py:80` | `get_database_url()` | correct — password injected, scheme rewritten |
| `shared/langgraph_layer/checkpointer.py:9` | `settings.POSTGRES_URL` raw | passwordless; also psycopg wants `postgresql://`, not `+asyncpg` (see §5) |
| `shared/langchain_layer/agents/memory/cognee_client.py:111` | `settings.POSTGRES_URL` raw | hands Cognee a **credential-less** URL |
| `features/auth/service.py:512` | `create_async_engine(get_database_url())` | correct source, but builds a **second engine** outside lifespan |

Change 0 owns the fix, and the shape is now clear: the repair belongs in `settings` (or a single accessor) so no
caller can obtain an unusable URL — not in three separate call sites. Note also that `get_database_url()`
returns an `+asyncpg` URL, which is exactly what `AsyncPostgresSaver` **cannot** accept (psycopg), so the
checkpointer needs a *psycopg-flavoured* accessor, not the existing one. That is why `checkpointer.py:9`'s
docstring recommending `postgresql+asyncpg://` is wrong in both directions.

Two smaller observations on `get_database_url()`, worth a task but not urgent:
- `:57`'s sentinel comparison against the literal `"pass"` couples the logic to `settings.py:140`'s placeholder.
- `:61-67` appends `:{port}` after the `else parsed.netloc` branch, which already contains the port — a
  duplicated port for any URL with no username.

## §3 — `pg_textsearch` **IS available**. The BM25 precondition closes favourably.

| Extension | default_version | state on this server |
|---|---|---|
| `pg_textsearch` | **1.3.0** | available, NOT installed |
| `vector` | 0.8.6 | **INSTALLED 0.8.2** (older than available) |
| `vectorscale` | 0.9.0 | **INSTALLED 0.9.0** |
| `timescaledb` | 2.28.3 | **INSTALLED 2.29.1** |
| `pg_trgm` | 1.6 | available, NOT installed |
| `unaccent` | 1.1 | available, NOT installed |
| `uuid-ossp` | 1.1 | available, NOT installed |
| `vchord` | 0.5.3 | available, NOT installed |
| `vchord_bm25` | — | **NOT AVAILABLE** |
| `pg_search` | — | **NOT AVAILABLE** |

Non-builtin index access methods present: `diskann` (vectorscale), `hnsw`, `ivfflat` (pgvector).
**No `bm25` access method, and `to_bm25query()` has 0 matching `pg_proc` rows** — expected, because
`pg_textsearch` is not installed *yet*. `CREATE EXTENSION IF NOT EXISTS pg_textsearch` will succeed.

So the D5.1 decision to keep the existing BM25 implementation stands on solid ground, and the earlier
UNVERIFIED precondition in `findings-deployment.md` §5 is **CLOSED — favourable**. Note the vendor is
TigerData/Timescale's `pg_textsearch`, confirming the correction away from "VectorChord".

## §4 — THE BIG ONE: the database was **stamped, not migrated**. The entire document/vector/search schema does not exist.

`alembic_version` holds exactly one row: **`0004`**.

Revision graph as written in `src/alembic/versions/`:

```
c0c17c6eb1cc  initial_schema_document_vectors_and_...   down=None      [root]
2bc7726317f6  rename_metadata_to_meta_data             down=c0c17c6eb1cc
├─ a71f0d7d9c12  add_unified_documents_and_chunks      down=2bc7726317f6      ***HEAD A (unstamped)***
└─ 8a7d9b1c2e3f  add_search_documents_and_chunks       down=2bc7726317f6
   └─ 9f4a1b7c6d2e  contract_kb_clauses_pg_textsearch  down=8a7d9b1c2e3f
      └─ 0001  add_outbox_tables                       down=9f4a1b7c6d2e
         └─ 0002  add_razorpay_saas_billing_tables      down=0001
            └─ 0003  billing_plan_versioning_and_audit  down=0002
               └─ 0004  subscriptions_allow_resubscribe down=0003            ***HEAD B (stamped)***
```

Exactly **two heads**, branching at `2bc7726317f6`. Being stamped at `0004` means alembic believes
`c0c17c6eb1cc → 2bc7726317f6 → 8a7d9b1c2e3f → 9f4a1b7c6d2e → 0001 → 0002 → 0003 → 0004` have all been applied.

**They have not.** Actual table inventory (16 tables, all of them billing/audit):

```
alembic_version, audit_logs, currencies, email_templates, fx_rates, invoice_batches,
invoice_line_items, invoice_voids, invoices, payment_receipts, payments, plans,
reports, subscriptions, trial_extensions, webhook_events
```

Existence checks:

| Table | Created by | Exists? |
|---|---|---|
| `documents` | `a71f0d7d9c12` | **NO** |
| `chunks` | `a71f0d7d9c12` | **NO** |
| `search_documents` | `8a7d9b1c2e3f` | **NO** |
| `search_chunks` | `8a7d9b1c2e3f` | **NO** |
| `clauses` | `9f4a1b7c6d2e` (index at `:132`) | **NO** |
| `parent_documents` | — | **NO** |
| `events` | `0001` outbox? | **NO** |
| `memory_versions` | — | **NO** |

So the branch from the root through `9f4a1b7c6d2e` is marked applied while **none of its tables exist**. Only the
billing lineage (`0002`–`0004`) is genuinely present. Someone ran `alembic stamp` rather than
`alembic upgrade`, or applied the billing revisions against a database where the earlier ones were skipped.

### Why this changes the plans

1. **The "clauses table doesn't exist" finding is not a `clauses` problem.** It is the whole document/search
   schema. Three independent angles in `findings-deployment.md` §6 pointed at one much larger hole.
2. **D5.1's `DROP TABLE` + retarget shape is unnecessary.** You cannot drop `search_documents`/`search_chunks`
   — they were never created. There are **zero rows anywhere** to migrate, backfill, or preserve. The strongest
   possible confirmation of "no write path is live".
3. **`alembic upgrade head` will not fix it.** Those revisions are already marked applied, so upgrade skips
   them. And `alembic downgrade` will *fail*, because the downgrade bodies drop tables that do not exist.
4. **The two-head merge is not the hard part.** A merge revision joins `a71f0d7d9c12` and `0004` cleanly, but
   the merged head still leaves the phantom revisions stamped-but-unapplied.
5. **Therefore the document/search schema is effectively greenfield.** The cheapest honest path is: merge the two
   heads, then add **one new migration that creates the target schema outright** (unified `documents`/`chunks`
   plus whatever `search/` capability survives D5.1), rather than trying to reconcile phantom history.
   Do **not** attempt `stamp base` + re-upgrade — the billing revisions sit downstream in the same lineage and
   would try to recreate the 15 tables that genuinely exist.
6. `pg_trgm` being uninstalled means `ix_search_chunks_content_trgm` (D5.1's "no target equivalent" capability)
   has never existed either. Trigram search is one of three RRF branches and has never run.

## §5 — `psycopg` cannot load libpq, so the checkpointer's dead-code fallback is the **live** path

```
langgraph-checkpoint-postgres : 3.0.4   (installed)
psycopg                       : 3.3.3   (installed)
psycopg-pool                  : 3.3.0   (installed)
psycopg-binary                : NOT INSTALLED
```

`from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver` raises:

```
ImportError: no pq wrapper available.
- couldn't import psycopg 'c' implementation: No module named 'psycopg_c'
- couldn't import psycopg 'binary' implementation: No module named 'psycopg_binary'
- couldn't import psycopg 'python' implementation: libpq library not found
```

**This corrects `plan-change1.md`'s Defect 4.** That analysis said the `AsyncPostgresSaver = Any` fallback at
`checkpointer.py:26-29` "is unreachable today" because the package is installed. The *package* is installed; its
**driver** cannot load. So the `except ImportError` branch fires, `AsyncPostgresSaver` **is** `typing.Any`, and
`setup_langgraph_checkpointer` short-circuits at `checkpointer.py:51-53`:

```python
if AsyncPostgresSaver is Any:
    logger.warning("LangGraph Postgres checkpointer is unavailable; skipping initialization")
    return None  # type: ignore[return-value]
```

Consequences, in order:

- Defect 1 (`from_conn_string` is an `@asynccontextmanager`, not a factory) is **currently unreachable** — the
  `is Any` guard returns before line 56. It becomes live the moment the driver is fixed. Both defects are real;
  the ordering is the opposite of what the plan assumed.
- The observable failure is a single `warning` log line, then `app.state.langgraph_checkpointer = None`, then
  `AttributeError` at `features/agent_saul/dependencies.py:45`, which reads it unguarded. Silent degradation into
  a crash at first request — the invisible-failure register again.
- **A new step-zero for change 1:** adding `psycopg[binary]` (or a system libpq) to the dependency set is a
  prerequisite for *any* checkpointer work. Rewriting `checkpointer.py` onto `AsyncConnectionPool` cannot even be
  imported until then. `psycopg-pool` already being installed suggests the pool shape was intended.
- Delete the `= Any` fallback as the plan says — but note it is currently the *only* reason the app boots at all
  on this machine, so the dependency fix must land in the same commit.

## §6 — CLOSED: `create_async_engine` does accept it, because `get_database_url()` repairs it first

Resolved by reading `connections/postgres.py:30-71` — see the corrected §2. `get_database_url()` rewrites the
scheme, strips the libpq-only query parameters, and injects the missing password. `auth/service.py:512` uses the
same helper, so it inherits the repair; its real issue is that it constructs a **second engine outside the
lifespan**, duplicating the pool the app already owns.

The open question this replaces: nothing needs a scheme fix. What needs fixing is that the repair is a function
callers must remember to use, and two of them don't.

## §7 — What is now closed

- `findings-deployment.md` §5 "does the image ship `pg_textsearch`" → **closed, available 1.3.0** (and the image
  was the wrong target; §1).
- `dispositions.md` "Fog still open": live-vs-orphan status of `parent_documents`, `events`, `memory_versions`
  → **closed, none of them exist**; nor do `documents`, `chunks`, `search_documents`, `search_chunks`, `clauses`.
- `plan-change4.md` **F8** (`entities` / `relationships` / `events` / `memory_versions`) → **closed, none exist.**
  The complete `pg_tables` inventory for schema `public` is the 16 billing/audit tables listed in §4; no table
  with any of those names is present, in this or any other form. Cognee's own alembic
  (`.venv/.../cognee/alembic/`) has therefore never run against this database either.
- `decisions.md` D5.1 "migration shape is DROP + retarget" → **refined: there is nothing to drop.**
- Vendor attribution → **`pg_textsearch` 1.3.0, not VectorChord.** Stronger than a naming correction:
  `vchord_bm25` is **not available on this server at all**, so the earlier attribution named an extension the
  deployment could not have installed. `vchord` 0.5.3 is available but unused. Corrected in `plan-change3.md`,
  `plan-change4.md`, and `scout-search.md`.

---

## §8 — The outbox tables do not exist either, and this breaks **public, mounted** endpoints

Found by the change-0 planner on its resumed pass; **verified independently by the orchestrator** (graphify
`explain` for the caller sets, `rg`/`sed` for the bodies). This is the most severe live break the relay has found,
and it ranks ahead of the document/search schema hole because it sits on already-shipped public surface.

### The tables

`outbox_events` and `dead_letter_events` are created **only** by `src/alembic/versions/0001_add_outbox_tables.py`
(`:24`, `:44`). Revision `0001` is **stamped as applied** (§4). Neither table appears in the 16-table live
inventory. So they do not exist, and `alembic upgrade head` will never create them.

This corrects §4's table, which guessed the name as `events` with a question mark. The real names are
`outbox_events` and `dead_letter_events`, and the row for `events` should be read as "no such table under any of
these names."

### Caller set (graphify `explain`, EXTRACTED edges — not inferred)

| Site | Reached from | Live? |
|---|---|---|
| `auth/service.py:245` → `_publish_outbox_event` | `POST /auth/resend-verification` (`auth/router.py:179-187`) | **YES — mounted, public, rate-limited** |
| `auth/service.py:272` → `_publish_outbox_event` | `POST /auth/forgot-password` (`auth/router.py:195-203`) | **YES — mounted, public, rate-limited** |
| `documents/service.py:184` → `upload_document()` | mounted documents router | **masked** — D5.2's `UserIdDep` `AttributeError` fires first |
| `search/service.py:106` → `ingest_document()` | unmounted `POST /search/ingest` | no |
| `lifespan.py:125` `_init_outbox_relay` → `run_startup_scan()` + `create_task(run_listener())` | boot | runs, but fails soft — see below |

### Two different failure modes, and the difference matters

**1. The relay fails SOFT and is therefore permanently, invisibly dead.**
`relay.py:66` catches `except (PostgresError, Exception)` — a catch-all (and a redundant tuple, since `Exception`
already subsumes `PostgresError`; worth a lint task on its own). `run_listener` likewise catches bare `Exception`
at `:80`, inside an `asyncio.create_task`, so it dies in the background with no propagation. The whole observable
signal is two warning lines: `outbox_startup_scan_skipped` and `outbox_listener_stopped`.

**This means the app boots fine.** The change-0 planner's summary implied a boot-path break; it is not one.
`lifespan.py:284-289` wraps `_init_outbox_relay` in `except (ConnectionError, TimeoutError, OSError,
RuntimeError, ValueError)` — which would **not** have caught `UndefinedTableError` (neither asyncpg's
`PostgresError` nor SQLAlchemy's `ProgrammingError` inherits from any of those five) — but the inner catch-all at
`relay.py:66` gets there first. Boot survives by accident, through a broad `except` that a future tightening pass
would remove. Record it that way: the resilience is unintentional.

**2. The writers fail HARD, after mutating user state.**
`with_outbox` (`shared/outbox/helper.py:31`) runs a raw `INSERT INTO outbox_events` in the caller's transaction.
`_publish_outbox_event` (`auth/service.py:481-500`) has **no** `try/except`. And in both auth callers the outbox
write happens *after* the user row is already saved:

```
forgot_password  (auth/service.py:263-278)
  resolved.reset_token_hash = hash_token(reset_token)   # mutate
  await self._user_repo.save(resolved)                  # PERSISTED
  await self._publish_outbox_event(...)                 # raises UndefinedTableError -> 500
```

So `POST /auth/forgot-password` and `POST /auth/resend-verification` **return 500 today**, having already written
a reset/verification token that no email will ever deliver. A public endpoint, on the billing-live half of the
app, failing after a partial write.

### Consequences for the plans

1. **Change 0 must create these two tables**, and this is independent of — and higher priority than — the
   document/search schema work. It is the only piece of change 0's migration that fixes a currently-500ing
   public endpoint.
2. **Ordering constraint, load-bearing:** fixing D5.2's `UserIdDep` without creating the outbox tables does not
   fix `POST /documents` — it moves the 500 from the dependency layer down to the outbox `INSERT`. The two fixes
   must land together or the endpoint's repair is illusory.
3. **A third URL flavour exists.** `lifespan.py:124` does `get_database_url().replace("+asyncpg", "")`, and
   `relay.py:71` strips it *again* on the already-stripped value. So change 0's single-accessor work has three
   consumers to serve, not two: SQLAlchemy+asyncpg (main engine), a plain libpq/psycopg DSN (relay listener via
   `asyncpg_listen`, and the checkpointer per §5), and Cognee. A single accessor returning one string cannot
   serve all three — it needs a flavour argument or three named accessors.
4. `relay.py:66`'s `except (PostgresError, Exception)` and `:80`'s bare `except Exception` belong in change 0's
   exception-tightening scope, but **only after** the tables exist — tightening first converts a silent
   degradation into a boot failure.
