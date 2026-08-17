# Plan — openspec change 2: documents / schema consolidation

> Planner leg, 2026-08-17. Governed by `docs/relay/decisions.md` (D5.1, D5.2 binding).
> Written incrementally; sections appended in order.

## Shape

**This change is a de-duplication, not a port.** The brief and the scouts describe change 2 as
"collapse `search_*` onto `UnifiedDocument`/`UnifiedChunk`". Verified against source, that collapse
is **already ~80% built on the documents side** and nobody noticed:

- `src/app/features/documents/repository.py` already carries a full retargeted copy of search's
  retrieval surface against `chunks`/`documents` — `upsert_chunks` (`:216`, and it already names the
  *correct* constraint `uq_chunks_document_chunk_index` at `:222`), `analyze_chunks` (`:258`),
  `bm25_search` (`:310`), `vector_search` (`:355`), `trigram_search` (`:406`),
  `fetch_chunks_by_ids` (`:453`), `legal_rrf_search` (`:477`, a **superset** of search's — it adds
  `user_id`, `document_ids`, `clause_type`, `require_graphiti_verified`), plus module helpers
  `build_chunk_rows` (`:601`) and `build_search_filter_params` (`:607`).
- Migration `a71f0d7d9c12:97-103` already creates all three retrieval indexes on `chunks`:
  `chunks_bm25_idx` (`USING bm25(search_text)`), `chunks_embedding_idx` (`USING diskann`), and
  `chunks_search_text_trgm_idx` (`USING gin(search_text gin_trgm_ops)`).
  **This closes scout-search Fog #2 and #3 and refutes non-mechanical cell #4** — see
  `## Blocking decisions`.
- `src/app/features/documents/router.py` already exposes `unified_search` (`:64`), `unified_rag`
  (`:76`), `ask_corpus` (`:88`) and `ask_legal` (`:102`) on the **mounted** router (`api/v1.py:15`).
  Every read capability of search's unmounted router already has a mounted twin.

So `features/search/` is not the future of retrieval — it is the **pre-unified twin that was left
behind**, still holding the only copies of five schema-free helpers (`chunking`, `fusion`, `rag`,
`constants`, `embeddings`) that `documents/` imports back out of it (`documents/service.py:15-26`,
`documents/repository.py:15`, `documents/dto.py:7`). That inverted import is the actual coupling.

**Therefore change 2 = subtract the twin, relocate the helpers, repoint two structural consumers.**

1. Relocate the five schema-free modules from `features/search/` into `features/documents/`
   (documents has **no `constants.py`** today — it must be created), inverting the import direction.
2. Delete the schema-bound twin: `search/model.py` (both tables), the `search_*`-bound half of
   `search/repository.py`, `search/service.py`'s ingest path (`ingest_document:72`,
   `process_ingestion_document:291`, `run_ingestion_task:349`), `search/router.py`,
   `search/dependencies.py`, `src/tasks/search_tasks.py`.
3. `DROP TABLE search_chunks, search_documents` in a new alembic revision. **No backfill** — every
   writer is dead (D5.1) and we delete the writers rather than retarget them, which is what makes the
   `user_id`/`object_uri` NOT-NULL blocker dissolve instead of needing a sentinel tenant.
4. Repoint the two structural consumers: `retrieval_kb/nodes.py:26,172,181` (types and calls
   `SearchRepository.legal_rrf_search`, which queries the **`clauses`** table that no migration
   creates) onto `DocumentRepository.legal_rrf_search` over `chunks` — this *is* item 184, resolved
   as **A+ retarget**; and `src/tasks/__init__.py:11,16` which re-exports the deleted celery task.
5. Item 185 (`content_tsv` + its GIN index) is pure subtraction and rides along with the table drop.

Net effect on capability: **zero read capability is lost**, one dead write path is removed, and
nothing previously unreachable becomes reachable — so the D5.1 "do not mount search" constraint is
honoured by *deletion*, which is strictly safer than leaving an unauthenticated router in the tree.

## Ordering constraints

### Inbound, cross-change

| Gate | Owner | Why it blocks change 2 |
|---|---|---|
| **Alembic head merge** (`0004` + `a71f0d7d9c12` → one head) | change 0 | My `DROP TABLE` revision needs a single `down_revision` to chain from. Writing it against two heads produces a third head. **Hard block on step 7 only** — steps 1-6 and 8-10 are pure Python and land without it. |
| **`9f4a1b7c6d2e` phantom `clauses` ALTER made runnable** | change 0 | `alembic upgrade head` cannot run on a clean DB today (`batch_alter_table("clauses")` at `:63`, `UPDATE clauses` at `:101`, `clauses_bm25_idx` at `:132` — no revision creates `clauses`). Until fixed, **no migration Proof in this change can be executed against a real database**; every migration step falls back to `alembic upgrade --sql` offline rendering. This is the single largest limit on this plan's verifiability. |
| **`env.py` model registration** | change 0 | If change 0 adds `import app.features.search.model` to `alembic/env.py:23-24`, step 6 must delete that import in the same commit or `env.py` raises `ModuleNotFoundError` and **every** alembic command dies. Coordinate: change 0 registers `app.features.documents.model` only; it must **not** register `app.features.search.model`. |
| **`tasks/__init__.py` edit** | change 0 | Change 0 already rewrites `tasks/__init__.py` for the reconciliation deletion; my step 8 removes `ingest_search_document` from `:11,16`. Same file, adjacent lines — change 0 lands first, step 8 edits the residue. |
| **`UserIdDep` fix (D5.2)** | change 0 | **NOT a prerequisite.** Change 2 deletes search's copy (`search/dependencies.py:44-45`) rather than fixing it, and every new test calls services directly. `documents/dependencies.py:60-61` stays broken until change 0 — change 2 neither fixes nor worsens it. |
| **`pg_textsearch` availability in `timescale/timescaledb-ha:pg18`** | precondition, unverified (`findings-deployment.md §5`) | If the image lacks it, both `search_chunks_bm25_idx` and `chunks_bm25_idx` are equally unbuildable, so the *comparative* claim "no capability lost" survives — but step 9's runtime Proof cannot be executed and BM25 is dead product-wide. **Step 0 establishes this before anything else.** |

### The one genuine conflict: change 1 writes `clauses`, change 2 declares `chunks` sole truth

D1 promotes `ingestion_kb`, whose persistence nodes write `parent_documents` / `clauses` / `entities` /
`relationships` (`ingestion_kb/nodes.py:497,660,551,597`). Item 184 declares `documents`/`chunks` the **sole
retrieval truth**. Both cannot hold. Change 1 is sequenced *before* change 2 (D3), so left alone change 1 would
build a promoted pipeline writing the schema change 2 then abolishes.

**Resolution — the ADR-first / code-after split.** Change 2's schema ADR (`chunks` is sole retrieval truth;
`clauses` is not a table, it is a *chunk_kind*) must be authored and accepted **before change 1 implements its
persistence nodes**, so change 1 retargets `ingestion_kb`'s writes onto `chunks` from the start rather than
being rewritten afterwards. Change 2's *code* still lands after change 1's; only its ADR moves upstream.
Concretely: `openspec/changes/documents-unified-schema/adrs.md` is a deliverable of change 2 that change 1's
`tasks.md` cites as an input. If that inversion is rejected, the fallback is to accept that change 1 writes
`clauses` and change 2 grows a real backfill (`clauses` → `chunks`) — which converts this change from
subtraction into a data migration and roughly doubles it. **Recommend the ADR-first split.**

### Intra-change order

Steps are ordered so the repo boots and the suite collects after every single one:

```
0  extension precondition (no code change)
1  create features/documents/constants.py            (additive)
2  move chunking/fusion/rag helpers into documents/  (additive + re-export shim in search/)
3  flip documents/* imports off features.search      (documents no longer imports search)
4  retarget retrieval_kb/nodes.py onto DocumentRepository   ← item 184 (A+)
5  fold the hardcoded index/constraint names into constants ← findings §7 + cell #3
6  delete the schema-bound twin (model/repository/service/router/dependencies)
7  alembic revision: DROP TABLE search_chunks, search_documents   ← item 185 rides along
8  delete src/tasks/search_tasks.py + tasks/__init__ re-export
9  add the schema-break gate that does not exist today
10 UnifiedChunk.updated_at decision, executed          ← cell #2
```

Step 2 lands a **re-export shim** in `features/search/__init__.py` precisely so that step 2 and step 3 are
independently committable: after step 2 both import paths work, after step 3 only the new one does, and
`tests/conftest.py:56-83` (which imports 20 symbols from `app.features.search.*` at module level) keeps
collecting until step 6 rewrites it.

## Blocking decisions

Five decisions must be recorded before code lands. Four are the non-mechanical cells; the fifth is dedup
semantics. **One of the four is refuted by source and needs no decision at all — see cell #4.**

### B1 — user provenance: `chunks.user_id` / `documents.object_uri` NOT NULL with no source value

`documents/model.py:40` (`user_id`), `:43` (`object_uri`) and `:87` (`chunks.user_id`) are `nullable=False` with
no default. Search's ingest supplies neither: `search/service.py:72-110` takes `SearchIngestRequest`
(`title`, `content`, `source_uri`, `doc_metadata`) from an unauthenticated request body and never touches S3.

**Proposed resolution: delete the writer. Do not invent a tenant.**

Reject all three "make it fit" options:

| Option | Why rejected |
|---|---|
| Sentinel/system user (`user_id = "system"`) | Creates a tenant whose rows every user's `(user_id, content_hash)` dedup ignores and whose chunks no `ix_chunks_user_document` lookup will ever return for a real user. A row nobody can retrieve is not data, it is a leak waiting for the first query that forgets to filter. |
| Make `user_id` nullable | Destroys the tenant-isolation invariant on the **only** table that has one. Every `WHERE c.user_id = :user_id` in `documents/repository.py` (`:317,364,417,469,512`) silently becomes a partial scan of a table with unowned rows. |
| `object_uri` nullable / `''` | `object_uri` is the provenance link back to the immutable S3 object. An empty one means "this text came from nowhere" — unauditable in a legal product, and it removes the only way to re-parse a document after a chunker change. |

The correct move is that the **question dissolves**: the collapse is `DROP TABLE` + retarget (D5.1), and the code
being retargeted is the *reader* side, which already lives on `documents/` and already has both values. The
*writer* side (`ingest_document`, `process_ingestion_document`, `run_ingestion_task`, `POST /search/ingest`) is
deleted rather than ported, so no remaining writer lacks a `user_id`. `documents/service.py:118 upload_document`
already supplies both (`user_id` from the request, `object_uri` from `build_s3_key` at `:40`).

**Cost, stated honestly:** the repo loses the ability to ingest *raw text* (as opposed to an uploaded file) —
today's `POST /search/ingest` is the only such path. That capability is not lost to users, because the endpoint
was never mounted, but it is lost to *the codebase*. Anyone who wants it back must build it as
`POST /documents/ingest-text` behind real auth, writing a real `user_id` and synthesising an `object_uri` by
storing the raw text as an S3 object first. **Record that as a Non-Goal in `design.md`, gated on D5.2.**
This also means change 2 must NOT be the change that adds a text-ingest endpoint — doing so would create exactly
the unauthenticated reachable surface D5.1 forbids.

### B2 — `UnifiedChunk` has no `updated_at`

Search's upsert maintains it (`search/repository.py:162` conflict set, `:542` `build_chunk_rows`).
`UnifiedChunk` (`documents/model.py:81-116`) has `created_at` only, while its **parent** `UnifiedDocument:56`
does have `updated_at`. The asymmetry is unexplained.

**Proposed resolution: add `updated_at` to `UnifiedChunk`.** It has a live consumer, not a hypothetical one:
`documents/repository.py:216 upsert_chunks` is a real upsert with an 11-column conflict set, and the documents
ingest path rewrites **every chunk row twice per ingest** — once at `documents/service.py:520`, then again at
`:686` after `_verify_legal_chunks` mutates the dicts in place. Without `updated_at` there is no way to tell a
chunk whose embedding was written by the current model generation from one carried over from a prior one, which
is precisely the audit a re-embedding campaign needs (and `settings.EMBEDDING_DIMENSION` drift is already a
known live hazard).

**Cost:** one column + one migration + one line in the conflict set. ~10 lines total.
**Two traps to honour in the implementation:**

1. The column must be `nullable=False` with a **server default** (`server_default=text("now()")`), not just a
   Python-side `default=`, or the `ALTER TABLE ... ADD COLUMN` fails on any existing row. (The tables are
   believed empty, but the migration must be correct on a populated DB regardless — see Risks.)
2. SQLAlchemy's `onupdate=` fires for `update()` constructs; it does **not** fire for
   `insert(...).on_conflict_do_update(...)`. So `"updated_at": statement.excluded.updated_at` must be added to
   the conflict set explicitly **and** `build_chunk_rows` must put the value in the row dict — mirroring what
   search's `build_chunk_rows` (`search/repository.py:542`) already does and documents' (`:601`) does not.
   Miss this and the column exists, is NOT NULL, and never changes — the worst outcome (looks maintained, isn't).

**Alternative (drop on the record):** costs nothing now, and is defensible if chunk rows are treated as
immutable-and-replaced. But the double-upsert above proves they are not immutable today, so dropping it means
recording a known-unauditable mutation. Recommend adding.

### B3 — hardcoded SQL identifiers: the constraint name AND the BM25 index name

Two instances of one defect class:

- `search/repository.py:157` — `on_conflict_do_update(constraint="uq_search_chunks_document_chunk_index")`.
- `search/constants.py:15` defines `SEARCH_CHUNKS_BM25_INDEX_NAME` and `repository.py:415,417,419,430` embeds
  the literal anyway (`findings-deployment.md §7`).

The target side is **not clean either**: `documents/repository.py` hardcodes `'chunks_bm25_idx'` at
`:323,327,331` and `:533,537,540`, and there is no `CHUNKS_BM25_INDEX_NAME` constant anywhere. `pg_textsearch`
needs the index named *inside* the query (it reads that index's corpus statistics), so a rename is a silent
runtime break with no lint, type, or migration warning.

**Proposed resolution: a drift test, not string interpolation.** Interpolating an index name into a `text()`
SQL string trips ruff's hardcoded-SQL rules and buys nothing — the literal has to be *somewhere*. Instead add
`tests/unit/documents/test_sql_identifiers.py` asserting, by static scan of `src/`:

1. every identifier appearing as the second argument of `to_bm25query(...)` in `src/app/` is created by a
   `CREATE INDEX <name>` in some `src/alembic/versions/*.py`, **and** the table it indexes is created by a
   `create_table`/`CREATE TABLE` in some revision;
2. every `constraint="..."` in an `on_conflict_do_update` matches a `UniqueConstraint(name=...)` on the model.

Cost: one test file, no production change. Value beyond tidiness: rule (1) **fails today** on
`clauses_bm25_idx` (`search/repository.py:356,361,362`; index created at `9f4a1b7c6d2e:132`, table created
nowhere), so the test is simultaneously the proof for item 184 and a permanent guard against the whole
invisible-failure class that produced it. Keep `SEARCH_CHUNKS_BM25_INDEX_NAME` deleted with the module; add
`CHUNKS_BM25_INDEX_NAME` to `documents/constants.py` **only** as the value the test asserts against.

### B4 — `ix_search_chunks_content_trgm` "has no target equivalent" — **REFUTED, no decision needed**

The brief and `scout-search.md §2` both state the trigram branch has no target equivalent. Source says
otherwise. Migration `a71f0d7d9c12:97-103` creates all three retrieval indexes on `chunks`:

```
:97   CREATE INDEX chunks_bm25_idx            ON chunks USING bm25(search_text) WITH (text_config='english', k1=1.2, b=0.75)
:100  CREATE INDEX chunks_embedding_idx       ON chunks USING diskann (embedding vector_cosine_ops)
:103  CREATE INDEX chunks_search_text_trgm_idx ON chunks USING gin(search_text gin_trgm_ops)
```

and `documents/repository.py:406-428 trigram_search` already uses `c.search_text % :query` +
`similarity(c.search_text, :query)`. **This also closes `scout-search.md` Fog #2 (diskann on `chunks`) and Fog
#3 (trigram on `chunks`) — both exist.**

There is a real **semantic delta** to record, though, and it is not the one anyone flagged: the source matched
bare `content`; the target matches `search_text`, a STORED generated column concatenating
`clause_type ‖ ' ' ‖ preamble ‖ ' ' ‖ content` (`documents/model.py:100-109`). `pg_trgm`'s `similarity()` is a
normalized ratio over the whole string, so a long `preamble` **dilutes** the score of a match located in
`content`. The `TRIGRAM_SIMILARITY_THRESHOLD = 0.1` floor (`search/constants.py:16`) is therefore effectively
*stricter* on the target for any chunk with a preamble — and RRF only uses rank position, so the visible effect
is branch *recall*, not score.

**Proposed resolution: preserve the `search_text` variant as-is; record the dilution; re-tune the 0.1 floor in
change 1 as a retrieval-quality knob, not here.** Cost: zero code, one `design.md` Risk line.
**Rejected alternative:** add a second `gin(content gin_trgm_ops)` index to restore exact source behaviour —
that is a full extra GIN index (write amplification on the hottest table in the schema) to serve 1 of 3 RRF
branches in a variant nobody has ever tuned. Not worth it.

### B5 — dedup semantics: global `content_hash` → `(user_id, content_hash)`

Source: `search/model.py:34` `unique=True` on `content_hash` alone. Target: `documents/model.py:32`
`UniqueConstraint("user_id", "content_hash")`.

**What it means concretely:** the same document uploaded by two tenants is now stored twice — two `documents`
rows, two full chunk sets, two full embedding sets, two Graphiti episode sets. Cross-tenant duplicate storage
becomes the *normal* case rather than an error.

**This is the correct trade and should be accepted, not mitigated.** Global dedup on a multi-tenant corpus is a
cross-tenant information leak in three ways: (a) the second uploader's ingest returns the first uploader's
`document_id`, exposing that someone else holds that document; (b) `metadata_`, `jurisdiction`, `parties` on the
shared row come from whoever uploaded first; (c) a deletion or GDPR erasure by tenant A destroys tenant B's
document. `search_documents` has no `user_id` column at all (`model.py:22-46`), so it could not have done
better; `documents` can, and does.

**Cost:** storage and embedding spend scale with (tenants × shared documents). For a legal product where the
shared corpus is public statutes and standard-form contracts, that is real. **Recorded escape hatch, explicitly
a Non-Goal here:** content-addressed storage — one immutable blob + chunk set keyed by `content_hash`, plus a
per-tenant pointer row carrying tenant metadata and ACL. `object_uri` already gives content-addressed S3 keys,
so the groundwork exists. Do not build it in change 2; name it in `design.md` Non-Goals.

### B6 — item 199 (`DocumentQueryService.__init__` `object | None`) is **already fixed in-tree**

`documents/service.py:232-242` already reads `redis: Redis | None` / `graphiti: Graphiti | None` with matching
attribute annotations at `:239-242`. `docs/relay/plan-change0.md` does not exist yet, so I cannot confirm change
0 claims it; either way **there is nothing to do**. `dispositions.md:28` moved item 199 to change 0 — that row
should be marked already-satisfied rather than assigned.

Residue, adjacent but out of item 199 as worded: the only surviving `object | None` in this area is
`retrieval_kb/reranker.py:27` `self._model: object | None = None`. It is one line and it is inside the module
step 4 touches, so fold it into step 4 rather than leaving it for a future sweep.

## Steps

**Baseline traps that govern every Proof below.** `pyproject.toml:752-760` puts `--cov-fail-under=80` in
`addopts` and total coverage is 18.38%, so a fully green suite still exits 1. **Compare the printed summary
line, never `$?`.** Baselines: pytest `22 failed, 55 passed, 11 warnings, 13 errors`; `uv run ruff check src/`
**125** errors (→ ~123 after change 0 deletes `todo_temp.py`); `uv run ty check src/` **46** diagnostics;
`ast-grep scan src/` **4** errors with exit 0; `openspec validate --all` **16 passed / 6 failed** — criterion is
*no new failures beyond those 6*, never "all pass".

Before step 1, capture the tree state you are comparing against:

```bash
uv run pytest 2>&1 | tail -3            > /tmp/c2-baseline-pytest.txt
uv run pytest --collect-only -q 2>&1 | tail -3 > /tmp/c2-baseline-collect.txt
uv run ruff check src/ 2>&1 | tail -1   > /tmp/c2-baseline-ruff.txt
uv run ty check src/ 2>&1 | tail -1     > /tmp/c2-baseline-ty.txt
```

---

### Step 0 — establish the `pg_textsearch` precondition (no code change)

Inbound: none. Blocks the runtime half of step 9 and the whole BM25 story product-wide
(`findings-deployment.md §5`).

**Proof:**
```bash
docker run --rm timescale/timescaledb-ha:pg18 \
  ls /usr/share/postgresql/18/extension/ | grep -E 'textsearch|vectorscale|vchord|trgm'
```
Expected: control files for `pg_textsearch`, `vectorscale`, `pg_trgm` are listed.
**If `pg_textsearch` is absent:** stop and escalate — `CREATE EXTENSION IF NOT EXISTS pg_textsearch`
(`a71f0d7d9c12:26`) fails the migration, so neither `chunks_bm25_idx` nor `search_chunks_bm25_idx` can exist and
the BM25 branch is dead on both sides of the collapse. Record the result in `design.md` Context either way; it
does not change *this* change's shape (the comparative claim "no capability lost" holds regardless) but it makes
step 9's DB-backed Proof unrunnable and it is a release blocker owned by change 1.

---

### Step 1 — create `src/app/features/documents/constants.py`

`documents/` has **no `constants.py`**. Move the 16 constants from `search/constants.py` verbatim except:
drop `SEARCH_CHUNKS_BM25_INDEX_NAME`, add `CHUNKS_BM25_INDEX_NAME = "chunks_bm25_idx"` and
`CHUNKS_UNIQUE_CONSTRAINT = "uq_chunks_document_chunk_index"` (values that B3's drift test asserts against).
Leave `search/constants.py` in place re-exporting from the new module — additive only, nothing breaks.

Inbound: none.

**Proof:**
```bash
uv run python - <<'PY'
from app.features.documents import constants as d
from app.features.search import constants as s
assert d.RRF_K == s.RRF_K == 60
assert d.TRIGRAM_SIMILARITY_THRESHOLD == 0.1
assert d.CHUNKS_BM25_INDEX_NAME == "chunks_bm25_idx"
assert not hasattr(d, "SEARCH_CHUNKS_BM25_INDEX_NAME")
print("ok")
PY
uv run ruff check src/ 2>&1 | tail -1   # must not exceed the captured baseline
```
Expected: `ok`, and ruff count unchanged from `/tmp/c2-baseline-ruff.txt`.

---

### Step 2 — relocate the three schema-free helper modules into `documents/`

`git mv` `search/chunking.py`, `search/fusion.py`, `search/rag.py` → `features/documents/`. Leave
`features/search/__init__.py` as a **re-export shim** pointing at the new locations so both import paths resolve.
`search/embeddings.py` stays put for now — it is change 1's unified-embedder target (D5.1), and moving it here
would collide with that work; only its *importers* change, in step 3.

Inbound: step 1 (the moved modules import `constants`).

**Proof:**
```bash
uv run python -c "from app.features.documents.fusion import reciprocal_rank_fusion; \
from app.features.search import reciprocal_rank_fusion as old; assert old is reciprocal_rank_fusion; print('ok')"
uv run pytest tests/unit/search -q 2>&1 | tail -2
```
Expected: `ok`; `tests/unit/search/{test_chunking,test_fusion,test_rag}.py` still pass unchanged (they import
via `app.features.search`, which the shim serves). This is the step that proves the shim works — do not skip it.

---

### Step 3 — flip `documents/` off `features.search`

Rewrite `documents/service.py:15-26` (10 symbols), `documents/repository.py:15` (3 constants) and
`documents/dto.py:7` (3 constants) to import from `app.features.documents.{constants,fusion,rag}`. Keep
`build_embedding_client` importing from `app.features.search.embeddings` — flagged, not fixed here (change 1).
After this step, **`documents/` imports nothing from `search/` except the embedder**, which is the inversion the
brief calls the actual coupling.

Inbound: step 2.

**Proof:**
```bash
rg -n "features\.search" src/app/features/documents/
```
Expected: exactly one hit — `build_embedding_client` — with a `# change 1 owns this` comment. Then:
```bash
uv run pytest 2>&1 | tail -3    # summary line identical to /tmp/c2-baseline-pytest.txt
uv run ty check src/ 2>&1 | tail -1   # <= 46
```

---

### Step 4 — retarget `retrieval_kb` onto `DocumentRepository` (item 184, resolution A+)

This is item 184 executed. `retrieval_kb/nodes.py:26` imports `SearchRepository` under `TYPE_CHECKING`, `:172`
types the node factory on it, and `:181` calls `repo.legal_rrf_search(...)` — a method that queries the
**`clauses`** table (`search/repository.py:337,383`) with `clauses_bm25_idx` (`:356,361,362`), an index created
at `9f4a1b7c6d2e:132` on a table **no revision creates**. Three independent proofs it is stale
(`scout-persistence-docling.md §1`, `scout-tools-schema.md`, `findings-deployment.md §6`).

Retarget to `DocumentRepository.legal_rrf_search` (`documents/repository.py:477`), which queries `chunks`
(`:510`) with `chunks_bm25_idx` (`:533,537,540`) and is a **superset** of the signature — it adds `user_id`,
`document_ids`, `clause_type`, `require_graphiti_verified`. Two of the four are already available at the call
site with no new plumbing: `RetrievalState` already declares `user_id` (`retrieval_kb/state.py:64`) and
`search/service.py:266` already seeds it. `document_ids` maps from `state["doc_ids_filter"]`. `clause_type`
passes `plan.clause_type` if `QueryPlan` carries it, else `None`. `require_graphiti_verified` defaults `False`.

`graph.py:31` already types `repo: Any`, so **no change is needed in `graph.py`** — the retarget is confined to
`nodes.py`. Fold in the `reranker.py:27` `object | None` residue from B6 while in this module.

Also in scope for this step: delete `SearchRepository.legal_rrf_search` (`search/repository.py:308-405`, ~98
lines) — it is the only clause-reading SQL in the feature and after this step it has zero callers. The other
clause reader, `precedent_tools.py:237`, is a stub `return []` **owned by change 3** (`dispositions.md`,
change 3 table) — do not touch it here; record the handoff.

Inbound: step 3 (so `nodes.py` can import documents' helpers without a cycle).

**Proof:**
```bash
rg -n "clauses" src/app/features/search/ src/app/shared/langgraph_layer/retrieval_kb/
```
Expected: zero hits (docstrings included — rewrite them).
```bash
rg -n "SearchRepository" src/app/shared/
```
Expected: zero hits.
```bash
uv run ty check src/ 2>&1 | tail -1
```
Expected: `<= 46` — this is the step most likely to *reduce* it, since `nodes.py` gains a real type.

---

### Step 5 — the SQL-identifier drift test (B3)

Add `tests/unit/documents/test_sql_identifiers.py` implementing B3's two static rules. It **must fail before
step 4 and pass after**, which is what makes it a regression guard rather than a snapshot.

Inbound: step 4 (rule 1 fails while `clauses_bm25_idx` is still queried).

**Proof:**
```bash
git stash && uv run pytest tests/unit/documents/test_sql_identifiers.py -q 2>&1 | tail -2   # expect FAILED
git stash pop && uv run pytest tests/unit/documents/test_sql_identifiers.py -q 2>&1 | tail -2  # expect 2 passed
```
(Or equivalently, run it at `HEAD~1`.) Expected: red on the pre-step-4 tree naming `clauses_bm25_idx`, green
after. Two new passes → pytest summary becomes `22 failed, 57 passed`.

---

### Step 6 — delete the schema-bound twin

Delete: `search/model.py` (both tables), `search/repository.py`, `search/router.py`, `search/dependencies.py`,
and from `search/service.py` the ingest half (`ingest_document:72`, `get_ingest_status:129`,
`process_ingestion_document:291`, `run_ingestion_task:349`) plus the ingest DTOs in `search/dto.py`.

`ask_legal` (`search/service.py:257`) is the **only caller of `build_retrieval_graph`** (`retrieval_kb/graph.py:28`).
Do not delete it in this change — move `SearchService.ask_legal` onto `DocumentQueryService` as a graph-backed
sibling of the existing hand-rolled `ask` (`documents/service.py:357`), keeping it unexposed by any router.
**Rationale for the seam:** `documents/service.py:357-460` is a straight-line Python reimplementation of the same
pipeline the graph implements (plan → graphiti filter → embed → `legal_rrf_search` → grade → generate, 2
iterations) — exactly the D1 shape inverted, with the *graph* being the unreachable-but-better one. Choosing
between them, and wiring the reranker, is item 195 / change 1. Change 2's obligation is only to ensure the graph
is pointed at the unified schema; **do not promote it here.**

Then rewrite the test surface, which is the real work of this step: `tests/conftest.py:56-83` imports 20 symbols
from `app.features.search.*` at **module level**, and conftest is global — so a missing symbol is a collection
error for the **entire** suite, not just search tests. Repoint those to `app.features.documents.*`. Rewrite
`tests/integration/test_search.py` (12 `app.features.search.service` references, incl. the patch targets at
`:34,38,58,62,91,95`) against the documents services, and `git mv tests/unit/search tests/unit/documents_retrieval`.
Finally drop the `features/search/__init__.py` shim; what remains of `features/search/` is
`embeddings.py` + `__init__.py` awaiting change 1.

Inbound: steps 2-5. **Also inbound: change 0's `env.py` decision** — `alembic/env.py` must not import
`app.features.search.model`.

**Proof:**
```bash
uv run pytest --collect-only -q 2>&1 | tail -3
```
Expected: error count **not above** `/tmp/c2-baseline-collect.txt`. A jump means conftest broke — the single
highest-risk moment in this plan.
```bash
rg -n "search_chunks|search_documents|SearchChunk|SearchDocument|SearchRepository|SearchService" src/ tests/ --glob '!*.pyc'
```
Expected: zero hits outside `src/alembic/versions/` (historical revisions stay frozen).
```bash
uv run pytest 2>&1 | tail -3
```
Expected: `55 → 57 passed` (step 5's two), failures **not above** 22, errors not above 13.
```bash
uv run python -c "from app.api.v1 import router; print(sorted({r.path for r in router.routes}))"
```
Expected: the `/documents/*` paths unchanged and **no `/search/*` path** — proving no endpoint was gained.
This is the explicit D5.1 check: the change removes surface, never adds reachable unauthenticated surface.

---

### Step 7 — alembic revision: drop the search tables (item 185 rides along)

New revision, `down_revision` = change 0's merge head. Body:

```python
op.execute("DROP INDEX IF EXISTS ix_search_chunks_content_tsv_gin")   # item 185, the reader-less GIN
op.execute("DROP INDEX IF EXISTS ix_search_chunks_content_trgm")
op.execute("DROP INDEX IF EXISTS search_chunks_bm25_idx")
op.execute("DROP INDEX IF EXISTS search_chunks_embedding_idx")
op.drop_table("search_chunks")     # takes content_tsv (the STORED generated column) with it
op.drop_table("search_documents")
```

`downgrade()` must recreate both tables verbatim from `8a7d9b1c2e3f` — including `content_tsv` — or the
revision is not reversible. **Item 185 is fully discharged here**: `content_tsv` (`search/model.py:75-79`) and
its GIN index die with the table; there is no separate `DROP COLUMN` step and no reader to fix
(`scout-search.md §8`: zero `@@` / `ts_rank` / `plainto_tsquery` / `content_tsv` references outside the model and
the migration).

Inbound: **change 0's alembic head merge (hard)**; step 6 (drop the model before the table, so `env.py`
autogenerate and the DB agree in the same commit).

**Proof — offline, since no DB is reachable and change 0's `clauses` fix gates a real upgrade:**
```bash
uv run alembic upgrade <merge_head>:<new_rev> --sql 2>&1 | grep -iE 'DROP TABLE|DROP INDEX'
```
Expected: the six statements above, in that order (indexes before tables).
```bash
uv run alembic heads
```
Expected: exactly one head — the new revision.
```bash
uv run alembic downgrade <new_rev>:<merge_head> --sql 2>&1 | grep -icE 'CREATE TABLE'
```
Expected: `2`.
**Deferred proof, once change 0 makes a clean upgrade possible:** `alembic upgrade head` on an empty DB, then
`psql -c "\dt"` shows neither `search_documents` nor `search_chunks`, and
`psql -c "\d chunks"` shows `chunks_bm25_idx`, `chunks_embedding_idx`, `chunks_search_text_trgm_idx`.

---

### Step 8 — delete the celery ingest task

Delete `src/tasks/search_tasks.py` (69 lines) and its re-export at `src/tasks/__init__.py:11,16`. Remove
`"tasks.search_tasks"` from the `include` list at `src/app/connections/celery.py:191-196`. Note the
`CeleryTaskRegistry.register("tasks.search_ingest", SearchIngestPayload)` call goes with it, so the
`tasks.search_ingest` outbox event type becomes unroutable — correct, since its only emitter
(`search/service.py:110`) was deleted in step 6.

**Care:** `tests/conftest.py:31` does `sys.modules["tasks.search_tasks"] = MagicMock()`. Remove that line too or
it masks the deletion. And per `findings-deployment.md §3`, importing **any** `tasks.*` module runs
`tasks/__init__.py`, so a broken re-export there breaks every worker at import — this file is more load-bearing
than its size suggests.

Inbound: step 6; change 0's `tasks/__init__.py` rewrite lands first.

**Proof:**
```bash
rg -n "search_tasks|search_ingest|ingest_search_document" src/ tests/ --glob '!*.pyc'
```
Expected: zero hits.
```bash
uv run python -c "import tasks; print(sorted(tasks.__all__))"
```
Expected: imports cleanly, `ingest_search_document` absent.
```bash
uv run python -c "from app.connections.celery import celery_app; print(len(celery_app.tasks))"
```
Expected: one fewer registered task than before the step, and no import error.

---

### Step 9 — add the schema-break gate that does not exist today

**The existing tests cannot catch a schema break, and this is the plan's most important admission.**
`tests/integration/test_search.py:34` patches `SearchRepository`, `:38` patches `build_embedding_client`, and
`:45` makes the session an `AsyncMock`; `tests/conftest.py` has **no engine, no `create_all`, no testcontainer**.
So every SQL string, every column name, every index name, and every constraint name in this change is
**unverified by the suite** — a renamed column or a dropped index passes green.

Three gates, in increasing cost; land at least the first two:

1. **B3's drift test (step 5)** — catches index/constraint-name drift statically. Free.
2. **`alembic check`** — with `env.py` registering `app.features.documents.model` (change 0), autogenerate
   diffing model vs. migrations catches any column/index the ORM declares and no revision creates, and vice
   versa. Needs a reachable DB, so it is a CI gate, not a local one.
3. **A real-Postgres smoke test** — `pytest.mark.integration`, a `timescale/timescaledb-ha:pg18` container,
   `alembic upgrade head`, then execute each of `bm25_search`, `vector_search`, `trigram_search`,
   `legal_rrf_search`, `upsert_chunks` once against an empty table asserting *no exception* (not results). That
   single test would have caught `clauses_bm25_idx` years ago and is the only thing that can prove step 7 did
   not break retrieval. Gate it behind a marker so the default suite stays offline.

Inbound: steps 4-7; step 0 (gate 3 cannot run if the image lacks `pg_textsearch`).

**Proof:**
```bash
uv run pytest -m integration -q 2>&1 | tail -3     # gate 3, when a DB is available
uv run alembic check                                # gate 2; expect "No new upgrade operations detected."
```
Expected: green, or an explicit skip with a recorded reason. **If gate 3 is descoped, `design.md` must say in
Risks that this change's SQL is shipped unexecuted** — do not let that go unrecorded.

---

### Step 10 — execute the `updated_at` decision (B2)

Add `updated_at` to `UnifiedChunk` (`documents/model.py`, after `created_at:110`) with
`server_default=text("now()")`, `onupdate=...`, `nullable=False`; a migration `add_column`; `"updated_at":
statement.excluded.updated_at` in `upsert_chunks`' conflict set (`documents/repository.py:222-236`); and the
value written by `build_chunk_rows` (`:601`). Honour both traps in B2.

Inbound: step 7 (same alembic lineage — chain this revision after the drop so the two never race).

**Proof:**
```bash
uv run python -c "from app.features.documents.model import UnifiedChunk; \
c=UnifiedChunk.__table__.c['updated_at']; print(c.nullable, c.server_default is not None, c.onupdate is not None)"
```
Expected: `False True True`.
```bash
rg -n "updated_at" src/app/features/documents/repository.py
```
Expected: hits in **both** the conflict set and `build_chunk_rows` — one without the other is the silent-failure
mode B2 warns about.
```bash
uv run alembic upgrade <prev>:<this> --sql | grep -i "ADD COLUMN"
```
Expected: one `ALTER TABLE chunks ADD COLUMN updated_at TIMESTAMP WITH TIME ZONE DEFAULT now() NOT NULL`.
**If B2 is decided the other way (drop the behaviour), skip this step entirely and record it in `design.md`
Non-Goals — do not leave the column half-wired.**

### Proof-command calibration (measured, not assumed)

Two Proof commands above were executed on the current tree so their expected output is real, not guessed:

- `uv run pytest --collect-only -q` → **`76 tests collected, 2 errors in 19.17s`**, and crucially
  **`!!! Interrupted: 2 errors during collection !!!`** — collection *halts* on
  `tests/unit/billing/test_circuit_breaker.py` and `tests/unit/billing/test_tax.py`. So `--collect-only` is a
  **weaker gate than it looks**: the 76 is a partial count and it does not reach the 13 errors the full run
  reports. Use it as a fast "did conftest break" signal (any change to `76 / 2` is a finding), but the
  authoritative import gate remains the full `uv run pytest` summary line.
- `uv run python -c "from app.features.documents.router import router; print(len(router.routes))"` → **`6`**,
  imports cleanly with no settings or service dependency. Step 6's router Proof is therefore executable today.

## Openspec mapping

**Namespace check performed first, as `config.yaml:39-43` requires.** `openspec/specs/` holds **20**
capabilities: `cognee-v1-api`, `datetime-utc-cleanup`, `llm-injection`, `mcp-context-di`,
`mcp-directory-restructure`, `mcp-server-codemode`, `mcp-server-composition`, `mcp-server-pagination`,
`mcp-server-prompts`, `mcp-server-resources`, `mcp-telemetry`, `mcp-testing`, `noqa-documentation`,
`outbox-helper-extraction`, `pattern-matching-standard`, `session-required`, `settings-validation`,
`test-mock-isolation`, `transactional-outbox`, `typed-exception-handling`.
**None covers documents, search, retrieval, schema, or migrations** — confirming `scout-search.md` Fog #8 and
`scout-tools-schema.md`. So there is no existing capability to extend, and every delta here is `ADDED`.

Change ID (bare slug, D12): **`documents-unified-schema`**.
`.openspec.yaml`: `schema: spec-gated`, `created: 2026-08-17`, **no `skip_specs`** — there are real deltas.
Change class **L** (multi-module + data migration), so `design.md` is mandatory and `adrs.md` is warranted.

### New capability: `document-retrieval-schema`

One capability, not two. Splitting schema from retrieval would put the index requirements in a different spec
from the columns they index, and the two cannot be satisfied independently.

| Req | Normative statement | Scenario spine |
|---|---|---|
| R1 **Single retrieval truth** | All document retrieval SHALL read from the unified document and chunk store; no other table SHALL serve retrieval. | WHEN a retrieval query executes THEN it SHALL target the unified chunk store, and no query SHALL reference a table that no migration creates. |
| R2 **Per-tenant document identity** | Document identity SHALL be the pair (owner, content hash); identical content owned by different tenants SHALL be stored as separate documents. | WHEN two tenants ingest byte-identical content THEN two independent documents SHALL exist, and deleting one SHALL NOT affect the other. |
| R3 **Mandatory chunk provenance** | Every chunk SHALL carry an owner and every document SHALL carry an immutable object reference; the system SHALL NOT accept ingestion that cannot supply both. | WHEN an ingest request supplies no owner THEN the system SHALL reject it rather than assign a shared or sentinel owner. |
| R4 **Three retrieval branches** | The chunk store SHALL support keyword (BM25), vector-similarity, and fuzzy (trigram) retrieval, fused by reciprocal rank. | WHEN a hybrid query runs THEN all three branches SHALL execute against indexed columns and their results SHALL be rank-fused. |
| R5 **Referenced identifiers exist** | Every database index named inside a query SHALL be created by a migration, on a table created by a migration. | WHEN the codebase names an index in a query THEN a migration SHALL create that index and its table. |
| R6 **Chunk modification time** *(only if B2 is accepted)* | Every chunk SHALL record when its content or embedding was last written. | WHEN a chunk is re-upserted with new content THEN its modification time SHALL advance. |

Formatting traps that will silently eat this spec if missed (D12, `schema.yaml:164-165`):
**scenario headers take exactly four hashtags** — three fails *silently*; every requirement needs ≥1 scenario;
SHALL/MUST only.

### What deliberately gets NO delta

- **The removal of raw-text ingest.** `REMOVED Requirements` deltas apply to requirements that exist in
  `openspec/specs/`; no spec ever described `POST /search/ingest`, so there is nothing to remove. Its
  disappearance is captured by R3's normative "SHALL NOT accept ingestion that cannot supply both" plus a
  `design.md` Non-Goal — **not** by inventing a requirement in order to delete it (`schema.yaml:49-59`
  explicitly forbids padding).
- **`content_tsv`** (item 185). Zero readers means zero observable behaviour, so it is a pure-refactor detail for
  `design.md` Migration Plan, not a spec delta.
- **Mounting the search router.** Out of scope (D5.1) and gated on D5.2. No delta, and no task.

### ADR

`adrs.md` gets **one** durable entry: *"`documents`/`chunks` is the sole retrieval schema; `clauses`,
`search_chunks`, `statutes` and `document_vectors` are not."* It outlives the change, it is the contract changes
1 and 3 build on, and per `## Ordering constraints` it must be **accepted before change 1 implements its
persistence nodes**. Its Consequences section carries B5's per-tenant duplication cost and B4's trigram-dilution
delta. The Graphiti/Cognee boundary ADR (D2) belongs to change 4, not here.

### Validation

```bash
openspec validate --all 2>&1 | tail -5
```
Expected: `16 passed / 6 failed` becomes **`17 passed / 6 failed`** — the six pre-existing failures
(`spec/cognee-v1-api`, `change/mintlify-documentation`, `spec/noqa-documentation`,
`spec/pattern-matching-standard`, `spec/transactional-outbox`, `spec/typed-exception-handling`) are the
acceptance floor. A 7th failure is a defect in our artifacts. Never assert "all pass".

Artifact order is gated (`schema.yaml`): `proposal` → `specs` + `design` → `review` (**`VERDICT:` must not be
`CHANGES-REQUESTED`**) → `tasks` → `apply`. The `review.md` is written by a **fresh subagent, never the author**
(`schema.yaml:321`); its Standards axis will check `.opencode/instructions/` — most relevant here,
`RESULT-PATTERN.md` (both repositories already return `AppResult`; the retarget must not break that) and
**no `match/case` on Success/Failure**.

## Risks

Format is `[Risk] → Mitigation`, matching `design.md`'s required style.

1. **[`tests/conftest.py` is a single point of failure for the whole suite]** — it imports 20 symbols from
   `app.features.search.*` at module level (`:56-83`), and conftest is global, so one missing symbol is a
   collection error for **every** test, not just search's. → Step 2's re-export shim keeps both paths alive until
   step 6 rewrites conftest in the same commit as the deletion; step 6's Proof compares
   `pytest --collect-only` against `76 tests collected, 2 errors`. Never delete a module and its conftest
   reference in separate commits.

2. **[This change's SQL is shipped unexecuted]** — no test in the repo touches a database
   (`test_search.py:34,38,45` mock repo, embedder *and* session; `conftest.py` has no engine or `create_all`), so
   a renamed column, dropped index, or wrong constraint name passes green. → Step 9's three gates. If gate 3
   (real Postgres) is descoped, that fact must appear verbatim in `design.md` Risks. **This is the risk most
   likely to be waved through, and the one that produced `clauses_bm25_idx`.**

3. **[Change 1 lands first and builds a promoted pipeline writing `clauses`]** — D1 promotes `ingestion_kb`,
   whose nodes write `clauses`/`parent_documents`/`entities`/`relationships`, while item 184 declares `chunks`
   sole truth. → The ADR-first/code-after split in `## Ordering constraints`: change 2's schema ADR is accepted
   before change 1 writes its persistence nodes. If rejected, change 2 grows a real `clauses`→`chunks` backfill
   and roughly doubles in size — decide before change 1 starts, not after.

4. **[No migration Proof can run against a real database]** — `9f4a1b7c6d2e` cannot execute on a clean DB
   (`batch_alter_table("clauses")` at `:63`, `UPDATE clauses` at `:101`, `clauses_bm25_idx` at `:132`; no
   revision creates `clauses`), and `scripts/init-db.sql` referenced by compose does not exist
   (`findings-deployment.md §4`). → Steps 7 and 10 use `alembic upgrade --sql` offline rendering as the primary
   Proof and mark the DB-backed check as **deferred, owned by change 0**. Do not silently skip it; record it as
   deferred.

5. **[`pg_textsearch` may not exist in `timescale/timescaledb-ha:pg18`]** — unverified
   (`findings-deployment.md §5`). If absent, `CREATE EXTENSION IF NOT EXISTS pg_textsearch` fails
   `a71f0d7d9c12:26` and BM25 is dead on both sides. → Step 0 establishes it before any code moves. Note the
   comparative claim survives either way — the collapse loses no capability the target did not already lack.

6. **[The tables are believed empty on the strength of code reading alone]** — I confirm the scouts: no
   reachable writer, no seed, no fixture, no factory. But nobody has run `SELECT count(*) FROM search_chunks`
   against staging or production, and `findings-deployment.md §4` argues some tables were created out-of-band, so
   the deployed DB's provenance is unknown. → `DROP TABLE` is irreversible for data even though the revision is
   reversible for schema. Before step 7 executes anywhere non-local, run
   `SELECT count(*) FROM search_documents; SELECT count(*) FROM search_chunks;` and require `0`. Put that as a
   literal precondition line in `tasks.md`, not as an assumption in prose.

7. **[Deleting `SearchService.ingest_document` removes the repo's only raw-text ingest]** — a real capability
   loss to the codebase (B1). → Recorded as a `design.md` Non-Goal with the exact shape of its replacement
   (`POST /documents/ingest-text`, behind auth, real `user_id`, raw text stored to S3 first for `object_uri`),
   gated on D5.2. Change 2 must not build it — doing so would create the unauthenticated reachable surface
   D5.1 forbids.

8. **[`build_retrieval_graph` loses its only caller if `ask_legal` is deleted rather than moved]** —
   `search/service.py:259` is the sole caller (`retrieval_kb/graph.py:28`), so a careless step 6 orphans the
   entire retrieval graph, which is change 1's foundation. → Step 6 **moves** `ask_legal` onto
   `DocumentQueryService`, unexposed by any router, rather than deleting it. Deciding between it and the
   hand-rolled `documents/service.py:357 ask` is change 1's item 195.

9. **[`updated_at` added but never written]** — SQLAlchemy's `onupdate=` does not fire for
   `insert(...).on_conflict_do_update(...)`. → B2's trap 2: the conflict set **and** `build_chunk_rows` must both
   be edited, and step 10's Proof greps for both. A NOT NULL column that never changes is worse than no column.

10. **[Two changes edit `src/tasks/__init__.py` and `alembic/env.py`]** — change 0 rewrites both for the
    reconciliation deletion and model registration; steps 6 and 8 edit the same lines. → Change 0 lands first;
    change 2 edits the residue. Explicit coordination point: change 0 must **not** register
    `app.features.search.model` in `env.py`.

11. **[Ruff/ty counts move for reasons unrelated to correctness]** — deleting ~1,100 lines of `search/` will
    change both counts, and `todo_temp.py`'s two `invalid-syntax` errors mean `src/` does not fully parse today.
    → Every Proof states `<=` against a captured baseline file, never an absolute number, and never `$?`.

## Late corrections (verified after the Steps section was written)

1. **Step 4 — `clause_type` cannot be passed from `QueryPlan`.** `retrieval_kb/state.py:11-23` `QueryPlan` has
   **no `clause_type` field** and declares `model_config = ConfigDict(extra="forbid")`, so
   `plan.clause_type` would raise. `RetrievalState` has no such field either. Step 4 must call
   `DocumentRepository.legal_rrf_search(..., clause_type=None, require_graphiti_verified=False)` and leave
   plumbing a real `clause_type` through the graph to change 1 (it is a retrieval-quality knob and touches a
   shared state model). Note the asymmetry this reveals: the hand-rolled `documents/service.py:357 ask` *does*
   filter by `clause_type` (from its request payload), so the graph path is strictly weaker until change 1 —
   record that in `design.md` rather than quietly matching the weaker behaviour.

2. **`scout-search.md` Fog #4 is CLOSED — search's `chunk_metadata` maps to `metadata_`, not
   `custom_metadata`.** Evidence: search filters on `c.chunk_metadata @> CAST(:metadata_filter AS jsonb)`
   (`search/repository.py:418`); the documents equivalent filters on `c.metadata_` and
   `build_search_filter_params` (`documents/repository.py:607-620`) feeds it from `metadata_filter["metadata_"]`;
   the GIN index is `ix_chunks_metadata_gin` on `metadata_` (`documents/model.py:77`). `custom_metadata` is
   *carried* (selected by `legal_rrf_search`, surfaced at `retrieval_kb/nodes.py:383` into `RetrievedChunk`) but
   **never filtered and never indexed** — it is a passthrough written by `ingestion_kb/nodes.py:650-678`. Mapping
   `chunk_metadata` anywhere but `metadata_` would silently drop the filter's index.

## Fog

Open questions this plan could not close, each with what would close it and who owns it.

| # | Fog | What closes it | Owner |
|---|---|---|---|
| 1 | **Do `search_documents` / `search_chunks` hold rows anywhere?** I re-confirmed the code argument (no reachable writer, no seed, no fixture, no factory) but `DROP TABLE` is irreversible for data. `findings-deployment.md §4` argues some tables were created outside alembic, so the deployed DB's provenance is genuinely unknown. | `SELECT count(*) FROM search_documents; SELECT count(*) FROM search_chunks;` on staging **and** production. Require `0`. | precondition on step 7, must be a literal line in `tasks.md` |
| 2 | **Does `timescale/timescaledb-ha:pg18` ship `pg_textsearch`?** If not, BM25 is dead product-wide and step 9's DB Proof cannot run. | Step 0's `docker run … ls …/extension/` | step 0; release-blocking consequence owned by change 1 |
| 3 | **Will `chunks.embedding` hold mixed normalization?** Verified: `documents/service.py` calls `normalize_embedding` **only** at `:829` (query side); stored vectors are raw. `ingestion_kb/nodes.py:733` normalizes every stored vector. Once change 1 unifies ingestion, one column can hold both conventions. `chunks_embedding_idx` is `vector_cosine_ops` (scale-invariant) so ranking is safe **today**; a switch to inner-product or L2 would silently mis-rank. | A decision in change 1's unified-embedder design: normalize always, or never. Then a one-off audit query on `vector_norm(embedding)`. | change 1 — but flagged here because `chunks` is change 2's table and R4 asserts its branches work |
| 4 | **Is the ADR-first / code-after inversion acceptable to the user?** It is the only way I can see to stop change 1 building a promoted pipeline that writes a schema change 2 abolishes. The alternative doubles change 2 into a real backfill. | A user decision, before change 1 starts. **This is the one item I would escalate rather than assume.** | orchestrator → user |
| 5 | **B2's direction** — add `updated_at` to `UnifiedChunk` or drop upsert-timestamp tracking. I recommend adding and gave the cost, but "is chunk mutation auditable" is a product question. | User or product decision; step 10 is skippable either way. | orchestrator → user |
| 6 | **Whether `openspec/specs/test-mock-isolation` constrains step 9's real-Postgres gate.** I did not read it, and it is the one existing capability whose name suggests it has an opinion about integration tests touching real services. | `Read openspec/specs/test-mock-isolation/spec.md` before writing step 9's task. | change 2 author, before `tasks.md` |
| 7 | **`ruff format --check src/` baseline is unestablished** (`baseline-tests.md` Fog). Step 2's `git mv` + step 3's import rewrites are exactly the kind of change that trips it. | Run `uv run ruff format --check src/ \| tail -1` on a clean tree before step 1 and store it alongside the other four baselines. | change 2 author, before step 1 |
| 8 | **Does anything outside `src/` reference the search endpoints?** I verified no client, test, or OpenAPI reference inside the repo, but not deployment configs, Caddy routes, or external consumers. Deleting `search/router.py` is safe only if nothing routes to `/search/*`. | `rg -n 'search' docker-compose.yml Caddyfile* 2>/dev/null` plus a check of any API gateway config outside the repo. | change 2 author, before step 6 |
| 9 | **`features/search/__pycache__` holds `dependency.cpython-312.pyc` and `handler.cpython-312.pyc`** for source files that no longer exist — evidence of an earlier rename I could not reconstruct (carried over from `scout-search.md` Fog #9). Harmless, but it means the feature has been restructured before without the history being obvious, which weakens "nobody ever used this" arguments slightly. | `git log --diff-filter=D --name-only -- 'src/app/features/search/*'` | anyone, cheap |

**What I deliberately did not decide, and why it is not fog:** whether the graph-backed retrieval path
(`build_retrieval_graph`) or the hand-rolled `DocumentQueryService.ask` survives. That is item 195 in change 1
(`dispositions.md`), it needs the reranker wiring to be settled first, and change 2 has no reason to prejudge it.
Change 2's obligation is discharged by pointing the graph at the unified schema and leaving both alive.

---

# REVISION 1 — live database connected (2026-08-17, after the plan above was written)

`docs/relay/findings-database.md` §3-§4 changes this change materially. **Read this revision as authoritative
wherever it contradicts the sections above.** Nothing above is deleted; the superseded parts are named
explicitly so the reasoning trail survives.

## Revised Shape

**The document/search schema is greenfield. Not "empty tables" — no tables.**

`alembic_version` holds one row, `0004`, on the billing lineage. That marks
`c0c17c6eb1cc → 2bc7726317f6 → 8a7d9b1c2e3f → 9f4a1b7c6d2e → 0001 → … → 0004` as applied. **None of those
revisions' tables exist.** Live inventory is 16 tables, all billing/audit. `documents`, `chunks`,
`search_documents`, `search_chunks`, `clauses`, `parent_documents`, `events`, `memory_versions` — and
`document_vectors` from `c0c17c6eb1cc`, which `scout-persistence-docling.md §1` called "the only fully
round-tripped table" — **all absent**. Someone ran `alembic stamp`, not `alembic upgrade`.

So the framing that has driven this whole refactor needs correcting at the root:

> **There are not five parallel document schemas in production. There are zero.**

Every consequence below follows from that one fact.

1. **Step 7 as written is void.** You cannot `DROP TABLE search_chunks` — it was never created. Delete every
   step, task, and Proof concerning dropping, backfilling, preserving, or counting rows. `## Risks` #6 ("tables
   believed empty on the strength of code reading alone") is **closed favourably** — not merely zero rows, zero
   tables. This is the strongest possible confirmation that no write path was ever live, and it retires B1's
   entire data dimension.
2. **Item 185 (`content_tsv`) is now a pure source-code edit** with no migration component at all: a STORED
   generated column and a GIN index on a table that does not exist. Zero migration risk, zero rollback risk.
3. **`documents`/`chunks` do not exist either** — so change 2's *target* has never run either. The mounted
   documents router (`api/v1.py:15`) would fail `UndefinedTable` on every retrieval call; that break is currently
   masked because `documents/dependencies.py:60-61` `UserIdDep` raises `AttributeError` first (D5.2). Two live
   breaks stacked, and fixing only the outer one exposes the inner one. **Change 0 must know this**: fixing
   `UserIdDep` without creating the schema converts an `AttributeError` into an `UndefinedTable` — visible
   progress, still broken.
4. **The migration content of this change moves upstream into change 0.** `findings-database.md §4.5` prescribes
   the honest repair: merge the two heads, then **one new migration that creates the target schema outright**,
   rather than reconciling phantom history (and explicitly *not* `stamp base` + re-upgrade, because the 15 real
   billing tables sit downstream in the same lineage). Consequence for change 2: **it ships no migration of its
   own.** B2's `updated_at`, B4's trigram index, and B1/R3's NOT NULL provenance columns must all be baked into
   change 0's single `CREATE TABLE`. Change 2 contributes the *column and index specification*; change 0 executes
   it. This is the same ADR-first/code-after inversion as the change-1 conflict, one level sharper.
5. **`alembic check` (step 9 gate 2) is not usable against this database.** With the stamp a fiction,
   autogenerate would report the entire document schema as missing and every diff as new — the tool cannot
   distinguish "model drifted from migrations" from "migrations were never run". Gate 2 becomes meaningful only
   *after* change 0's rebaseline, and only against a database built by `upgrade`, not `stamp`.
6. **Item 184 (A+ retarget) gets cheaper and more obviously correct.** Retargeting the clause readers costs
   nothing in data or deployment terms because no data and no environment depends on `clauses` —
   `findings-deployment.md §6` (the index that cannot be created), `scout-persistence-docling.md §1` (no revision
   creates the table) and now §4 (the table does not exist) are three angles on the same hole. "Leave it stale"
   (Option A) would mean preserving code that reads a table nothing has ever created in any environment.

**Revised one-line shape:** change 2 is now *pure source-code subtraction plus a schema specification handed to
change 0* — delete the pre-unified twin, relocate its schema-free helpers, retarget the graph off `clauses`, and
hand change 0 the exact column/index list for the one `CREATE TABLE` that will build `documents`/`chunks` for the
first time.

## Revision — live database findings: revised blocking decisions

### B1 revised — the resolution gets *stronger*, and changes category

**Category change: this was never a data problem, and now it is not even a hypothetical one.** `chunks.user_id`
and `documents.object_uri` being NOT NULL with no source value is a **forward contract decision only** — there is
no legacy row to strand, no sentinel tenant to assign, no nullable-column escape hatch to argue about, because
there is no table.

The recommendation is unchanged and better supported: **delete the writer; do not invent a tenant.** New
supporting evidence: the three rejected options were all migration-shaped ("what do we put in the column for
existing rows?"), and that question no longer exists. What remains is purely: *what must an ingest path supply?*
Answer: an authenticated owner and an immutable object reference, **enforced at the service boundary and failing
loudly**, not defaulted. Concretely — `NOT NULL` in the DDL *and* a required (non-`| None`) field on the ingest
DTO, so the failure is a 4xx at the edge rather than an `IntegrityError` from the driver. That distinction
matters now precisely because the columns are being written for the first time by change 0: a
`server_default=''` added "for safety" would permanently weaken the contract with nothing to justify it.

### B4 revised — the refutation stands, with the qualifier the orchestrator asked for

**Precise statement, to be quoted by later legs verbatim:**

> The target schema **defines** a trigram equivalent — `a71f0d7d9c12:103` creates
> `chunks_search_text_trgm_idx ON chunks USING gin(search_text gin_trgm_ops)`, and
> `documents/repository.py:406-428` queries it via `c.search_text % :query` — **but that migration has never been
> applied and `pg_trgm` is not installed on the live server. Nothing has ever run trigram search, on either
> schema.** `ix_search_chunks_content_trgm` never existed either.

So my refutation of the brief's cell #4 is about **schema definition**, not about working capability. No later leg
may read it as "trigram search works today". Same qualifier applies to `chunks_bm25_idx` and
`chunks_embedding_idx`: defined at `a71f0d7d9c12:97,100`, never created.

This turns B4 from *preserve-or-drop an existing capability* into a **fresh build-or-skip decision**:

| Option | Cost | Verdict |
|---|---|---|
| **Build it** — `CREATE EXTENSION pg_trgm` + the GIN index in change 0's create-schema migration, keep the 3-branch RRF | one extension (available at 1.6, not installed) + one GIN index's write amplification on the hottest table | **Recommended.** Fuzzy matching is the branch that survives OCR noise and typos in scanned legal PDFs — the repo's actual corpus. `pg_trgm` is available on the server, so the cost is an index, not a platform fight. |
| **Skip it** — 2-branch RRF (bm25 + vector), drop the index and `trigram_search` | zero build cost; loses typo/OCR recall; `RRF_K=60` was tuned for nothing in particular, so 2 branches is not a regression from a tuned state | Defensible, and cheaper. Choose this only if `CREATE EXTENSION pg_trgm` is refused by Timescale Cloud's role permissions. |

Either way, the `TRIGRAM_SIMILARITY_THRESHOLD = 0.1` floor is a **first-time calibration**, not a re-tune, and the
`search_text`-vs-`content` dilution noted in the original B4 applies to a branch running for the first time.
**Add a step-0 check for extension-creation permission** (below) — that is what decides this row.

### B3 revised — both hardcoded identifiers refer to objects that never existed

`on_conflict_do_update(constraint="uq_search_chunks_document_chunk_index")` (`search/repository.py:157`) names a
constraint that has never been created, and `SEARCH_CHUNKS_BM25_INDEX_NAME` (`search/constants.py:15`) names an
index that has never been created while `:415,417,419,430` bypass the constant anyway. The drift test in B3 is
therefore not just a guard against future renames — **it is red on three counts today**
(`clauses_bm25_idx`, `search_chunks_bm25_idx`, `uq_search_chunks_document_chunk_index`), which makes it the
cheapest single artifact in this plan for exposing the whole invisible-failure class. Keep it, and require it red
before step 4 / green after step 6.

### B2 and B5 — unchanged in substance, changed in venue

B5 (per-tenant dedup) is unaffected: it was always a contract decision. B2 (`updated_at`) is unaffected in
substance but **changes venue** — with no table to `ALTER`, it is no longer "add a column in change 2's
migration" but "include a column in change 0's `CREATE TABLE`". The `server_default` trap in B2 trap 1 becomes
moot (no existing rows), but **trap 2 stands unchanged and is now the entire risk**: SQLAlchemy's `onupdate=`
does not fire for `insert(...).on_conflict_do_update(...)`, so the conflict set and `build_chunk_rows` must both
write it or the column is inert.

## Revision — live database findings: revised steps

### Step 0 REPLACED — the extension question is answered; a permission question replaces it

`findings-database.md §3` answers the original step 0 against the **wrong target**: the local compose image is
not what the app talks to. The live server is Timescale Cloud, PG 18.0.4, and:

| Extension | Available | Installed |
|---|---|---|
| `vector` | 0.8.6 | **yes, 0.8.2** |
| `vectorscale` | 0.9.0 | **yes, 0.9.0** |
| `pg_textsearch` | **1.3.0** | no |
| `pg_trgm` | 1.6 | no |
| `unaccent`, `uuid-ossp`, `vchord` | yes | no |

Access methods `diskann`, `hnsw`, `ivfflat` are present. **Two consequences.** First, favourably: BM25 is
achievable — `pg_textsearch` is available, so `chunks_bm25_idx` (`a71f0d7d9c12:97`) will build once the migration
actually runs. Second, and worth noting because it could have been a blocker and is not: `a71f0d7d9c12` never
creates `vectorscale` (only the sibling branch's `8a7d9b1c2e3f:26` does, and that revision is stamped-but-unrun),
so its `USING diskann` index at `:100` depends on an extension its own lineage does not install — it survives
only because `vectorscale` is **already installed**. Record that as a latent fragility for change 0's
create-schema migration: it must create every extension it depends on rather than inheriting them.

**New step 0 Proof** — the only remaining unknown is whether the app's role may create extensions:
```sql
SELECT rolsuper, rolcreatedb FROM pg_roles WHERE rolname = current_user;
CREATE EXTENSION IF NOT EXISTS pg_trgm;      -- in a transaction that ROLLBACKs
CREATE EXTENSION IF NOT EXISTS pg_textsearch;
ROLLBACK;
```
Expected: both succeed. **If `pg_trgm` is refused, B4 resolves to "skip it" and RRF ships 2-branch.** If
`pg_textsearch` is refused, escalate — BM25 is unbuildable and R4 cannot be satisfied.

### Step 7 REPLACED — no drop migration. And to answer the question directly: **no, I do not want one.**

The original step 7 (`DROP TABLE search_chunks, search_documents` + a reversible `downgrade`) is void.
`op.drop_table` on a nonexistent table raises; only `DROP TABLE IF EXISTS` would be safe — and it would be a
migration whose entire body is a no-op in every environment that exists. **Recommendation: do not ship it.** A
no-op revision adds a lineage node, a `downgrade` that must recreate tables nobody wants, and a permanent
invitation for a future reader to conclude those tables once mattered.

But the tables *would* be created on any **fresh** database, because `8a7d9b1c2e3f` remains in the lineage and a
clean `alembic upgrade head` runs it. So something must still prevent that. Two routes:

| Route | Owner | Assessment |
|---|---|---|
| **Preferred — change 0 removes the search DDL from `8a7d9b1c2e3f` as part of its rebaseline** | change 0 | Editing "applied" history is normally forbidden, but here the applied state is *already a fiction*: the revision is stamped in the one real database and its tables exist nowhere. `findings-database.md §4.5` already commits change 0 to a rebaseline rather than reconciliation, so this rides along at no extra cost and leaves no dead node. |
| **Fallback — change 2 ships one `DROP TABLE IF EXISTS` revision** | change 2 | Only if change 0 declines to touch `8a7d9b1c2e3f`. Decoupled and safe, at the cost of the dead lineage node above. |

**Step 7 therefore becomes: delete the ORM models and assert no revision creates the tables.** No DDL of its own.

Inbound: step 6 (models deleted); change 0's rebaseline decision.

**Proof:**
```bash
rg -n "search_documents|search_chunks" src/alembic/versions/
```
Expected under the preferred route: **zero hits** — the DDL is gone from history entirely.
Under the fallback: hits only in `8a7d9b1c2e3f` (creating) and the new revision (`DROP … IF EXISTS`).
```bash
uv run alembic heads
```
Expected: exactly one head.
```bash
uv run alembic upgrade head --sql 2>&1 | grep -icE 'CREATE TABLE (search_documents|search_chunks)'
```
Expected: `0` — this is the assertion that matters, and it is the one that protects **fresh** databases (CI, dev,
a new tenant), which is where the risk actually lives now that production has no tables at all.

### Step 10 RELOCATED — `updated_at` moves into change 0's `CREATE TABLE`

No `ALTER TABLE ADD COLUMN` step exists any more. Change 2's deliverable is the **column and index specification**
handed to change 0: `documents`/`chunks` exactly as `documents/model.py` declares them, **plus** `updated_at` on
`UnifiedChunk` if B2 is accepted, **plus** the trigram index if B4 resolves to "build it". The ORM edit and the
repository conflict-set edit (B2 trap 2) stay in change 2.

**Proof (source-side only; the DDL Proof belongs to change 0):**
```bash
uv run python -c "from app.features.documents.model import UnifiedChunk; print('updated_at' in UnifiedChunk.__table__.c)"
rg -n "updated_at" src/app/features/documents/repository.py
```
Expected: `True`, and hits in **both** the conflict set and `build_chunk_rows`.

### Step 9 REVISED — gate 2 is unusable; gate 3 is promoted from nice-to-have to the only real gate

`alembic check` cannot distinguish "model drifted" from "migrations were never run" against a stamped-not-migrated
database, so **gate 2 is deferred until after change 0's rebaseline** and only against a DB built by `upgrade`.
That leaves the static drift test (gate 1, free) and the real-Postgres smoke test (gate 3) — and gate 3 is now the
**only** thing in the entire repo that can prove `documents`/`chunks` can be created and queried at all, since
they have never existed anywhere. Promote it from optional to required: a container, `alembic upgrade head`, then
one call each to `bm25_search`, `vector_search`, `trigram_search`, `legal_rrf_search`, `upsert_chunks` asserting
no exception. **If the trigram branch is dropped per B4, drop it from this test too rather than leaving a skip.**

## Revision — live database findings: escalations, revised risks, revised fog

### The two escalations, one sentence each, quotable verbatim

> **Escalation 1 — the ADR-first / code-after inversion.** Change 1 promotes `ingestion_kb`, whose persistence
> nodes write `clauses`/`parent_documents`/`entities`/`relationships` (`ingestion_kb/nodes.py:497,551,597,660`),
> while change 2's item 184 declares `documents`/`chunks` the sole retrieval truth — so either (**A, recommended**)
> change 2's schema ADR is authored and accepted *before* change 1 implements its persistence nodes, so change 1
> writes `chunks` from the start, or (**B**) change 1 ships writes to `clauses` and change 2 grows a real
> `clauses`→`chunks` migration, roughly doubling it; the live-database finding strengthens A, because `clauses` does
> not exist and never has, so there is nothing to preserve by choosing B.

> **Escalation 2 — B2, `UnifiedChunk.updated_at`.** `UnifiedChunk` has no `updated_at` while its parent
> `UnifiedDocument` does, and the documents ingest path rewrites every chunk row twice per ingest
> (`documents/service.py:520` then `:686`), so either (**A, recommended**) include `updated_at` in change 0's
> `CREATE TABLE` and write it from both the upsert conflict set and `build_chunk_rows` — about ten lines, and the
> only way a future re-embedding campaign can tell a current-generation embedding from a carried-over one — or
> (**B**) drop upsert-timestamp tracking on the record and accept that chunk mutation is unauditable.

### Revised risks

- **`## Risks` #6 — CLOSED favourably.** Not "believed empty": no tables exist. No data is at stake anywhere in
  this change. Delete the staging/production `count(*)` precondition from `tasks.md`.
- **`## Risks` #4 — half closed, half worse.** The reason no migration Proof can run against a real DB is no
  longer only `9f4a1b7c6d2e`'s phantom `clauses` ALTER; it is that the whole branch is stamped-not-applied, so
  `upgrade` skips it and `downgrade` would fail dropping tables that do not exist. Change 0 owns the repair; change
  2 ships no DDL, so change 2 has **no migration Proof left to defer** — which is a genuine simplification.
- **NEW risk — fresh-DB vs production divergence.** Production has zero document tables; a clean
  `alembic upgrade head` on CI or a dev machine runs `8a7d9b1c2e3f` (creating `search_documents`/`search_chunks`)
  and then fails at `9f4a1b7c6d2e`'s `clauses` ALTER. So *no* environment is currently correct, and CI and
  production are wrong in **opposite** directions. → Step 7's `alembic upgrade head --sql | grep -c 'CREATE TABLE
  search_'` = 0 assertion is the guard, and it protects the fresh-DB direction specifically.
- **NEW risk — the mounted documents router is broken twice over.** `api/v1.py:15` mounts it;
  `documents/dependencies.py:60-61` raises `AttributeError` (D5.2) and, behind that, every query would raise
  `UndefinedTable`. → Change 0 must fix both or neither; fixing `UserIdDep` alone converts one error into another
  and reads as progress. Change 2 must not be credited with, or blamed for, either.

### Revised fog

- **Fog #1 (rows in `search_*`) — CLOSED.** No tables, no rows. The strongest available confirmation that no write
  path was ever live.
- **Fog #2 (`pg_textsearch` in the compose image) — CLOSED, and it was the wrong question.** The app talks to
  Timescale Cloud, not the compose service; `pg_textsearch` is available there at 1.3.0.
- **NEW fog — may the app's role `CREATE EXTENSION`?** `pg_trgm` and `pg_textsearch` are available but not
  installed, and Timescale Cloud restricts extension creation by role. This is now the **only** unknown that can
  change change 2's content: it decides B4 (3-branch vs 2-branch RRF). Closed by the new step 0.
- **NEW fog — how did the stamp happen, and is anything else stamped-not-run?** `alembic_version = 0004` with 15 of
  the lineage's tables present and all document/vector tables absent means someone stamped past a branch. Whether
  the billing tables were themselves created by `upgrade` or by hand is unverified, and it matters for trusting
  *any* migration Proof in this refactor. → `git log --oneline -- src/alembic/versions/` alongside the deploy
  history; owned by change 0.
- **Fog #3 (mixed `normalize_embedding` conventions) — now a design question, not an audit.** With zero rows in
  `chunks`, there is no mixed data to detect: change 1 simply picks one convention before the first row is written.
  Cheapest possible moment; say so in change 1's design.
- **Fog #6, #7, #8, #9 — unchanged** (read `test-mock-isolation`; establish the `ruff format --check` baseline;
  check for external `/search/*` routes; the stale `__pycache__`).
