# Orchestrator findings — deployment, Celery, and Postgres extensions

Established 2026-08-17 by the orchestrator while Leg 2 planners were running. These were NOT in any scout
report and they change several planned steps. Every claim here is from a command run in this repo.

---

## 1. The documented way to start a Celery worker is broken

`Makefile:52` — `uv run celery -A celery_config worker --loglevel=info`

**`celery_config` does not exist.** `find . -path ./.venv -prune -o -name 'celery_config*' -print` returns
nothing. The real app is `src/app/connections/celery.py`. So `make celery` fails at load with
"Unable to load celery application", before any task registration question arises.

## 2. There is no worker or beat service in the compose stack at all

`docker-compose.yml` services are exactly: `rabbitmq`, `timescale`, `caddy`, `ai-service-1`.
`ai-service-1` declares **no `command:`**, so it runs the image CMD (the API). **Nothing consumes the queue.**
Every task dispatched via the outbox — including `tasks.documents_ingest` from
`features/documents/service.py:188` — enqueues to rabbitmq and is never executed by anything.

**This supersedes the `include`-list framing.** The missing-`include` finding is real but is not the reason
ingestion does not run. Ranked causes:

1. No worker process exists in the deployment (this file, §2).
2. The documented command to start one is broken (§1).
3. Only then does the `include` list matter.

## 3. CORRECTION — `tasks.document_tasks` IS registered transitively

Earlier framing ("the live ingestion task is never registered by the worker") is **wrong** in its mechanism.
`src/tasks/__init__.py:4` does `from .document_tasks import ingest_document`. Celery's `include` list names
`tasks.example`, `tasks.search_tasks`, `tasks.billing_tasks`, `tasks.auth_email_tasks`
(`app/connections/celery.py:191-196`); importing any of them imports the `tasks` **package** first, which runs
`tasks/__init__.py`, which imports `document_tasks` — so its task decorators DO execute and the task IS in the
registry.

The `include` omission is therefore a **latent fragility, not a live break**: it works only by side effect of
`tasks/__init__.py`, and it breaks the moment that file is tidied. Plan it as "make the guarantee explicit",
not "fix a non-registering task".

`tasks/__init__.py:6-9` also imports the reconciliation module and re-exports at `:18-20` — the coupled edit
change 0 already knows about. Note the same file is what makes §3 true; edit it carefully.

## 4. The database bootstrap script referenced by compose does not exist

`docker-compose.yml` mounts `./scripts/init-db.sql:/docker-entrypoint-initdb.d/init.sql:ro`.
**`scripts/init-db.sql` does not exist.** With a missing bind-mount source Docker creates a *directory* at that
path, so the postgres entrypoint finds a directory where it expects `init.sql`. Either way there is no
pre-alembic bootstrap. Nothing outside alembic creates any extension (`rg -l 'create extension'` returns only
the three migrations plus a docstring in `src/database/schemas/memory_schema.py`).

## 5. Required Postgres extensions, and the platform they assume

| Extension | Required by | Notes |
|---|---|---|
| `vector` (pgvector) | `a71f0d7d9c12:24`, `8a7d9b1c2e3f:25`, `9f4a1b7c6d2e:25` | the `Vector(768)` columns |
| `vectorscale` (pgvectorscale) | `8a7d9b1c2e3f:26` | Timescale's DiskANN index |
| `pg_textsearch` | `a71f0d7d9c12:26`, `8a7d9b1c2e3f:27`, `9f4a1b7c6d2e:26` | provides the `bm25` index access method and `to_bm25query()` |
| `pg_trgm` | `a71f0d7d9c12:25`, `8a7d9b1c2e3f:28` | the trigram branch of RRF |
| `unaccent` | `8a7d9b1c2e3f:29` | — |
| `uuid-ossp` | `a71f0d7d9c12:23`, `9f4a1b7c6d2e:24` | — |

Compose image is `timescale/timescaledb-ha:pg18`, which is consistent with `vectorscale` and `pg_textsearch`
being expected (both are Timescale/TigerData extensions, not VectorChord — an earlier attribution to
VectorChord was wrong and has been corrected in `decisions.md`).

**UNVERIFIED and it must become a precondition, not an assumption:** whether `timescale/timescaledb-ha:pg18`
actually ships `pg_textsearch`. If it does not, every BM25 path is dead on arrival regardless of code quality,
and `CREATE EXTENSION IF NOT EXISTS pg_textsearch` fails the migration. Establishing command:
`docker run --rm timescale/timescaledb-ha:pg18 ls /usr/share/postgresql/18/extension/ | grep -E 'textsearch|vectorscale|vchord'`

## 6. A second BM25 index the code queries and no migration can create

`features/search/repository.py:356,361,362` query `to_bm25query(:query_text, 'clauses_bm25_idx')`.
That index is created at `9f4a1b7c6d2e:132` **on the `clauses` table — which no migration creates.**
So this is the `clauses` hole seen from a third angle: the migration that indexes it cannot run, and the code
that queries it cannot succeed. Reinforces the change-0 finding that a merge revision alone is insufficient.

## 7. An index name is centralised and then hardcoded anyway

`features/search/constants.py:15` defines `SEARCH_CHUNKS_BM25_INDEX_NAME = "search_chunks_bm25_idx"`, but the
SQL at `features/search/repository.py:415,417,419,430` embeds the literal string instead of using it. Same
defect class as the hardcoded `constraint="uq_search_chunks_document_chunk_index"` at `:157` that change 2
already owns. `pg_textsearch` requires naming the *index* inside the query (it needs the index's corpus
statistics), so a rename silently breaks these queries at runtime with no tooling warning — which makes the
unused constant a live hazard rather than a style nit.
