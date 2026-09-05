# Fresh environment: repeatable route to the target schema

A plain `upgrade head` from an empty database is **not** a supported route:
revision `0005` operates throughout on the `clauses` relation
(`batch_alter_table`, `UPDATE`s, foreign key, indexes) while **no revision
in the chain creates `clauses`**, so an upgrade that reaches it aborts with
an undefined-relation error. `0005` cannot be edited or rewound through —
see `0012`'s docstring ("`0010` is unrunnable" in the old numbering) and
ADR-6. The procedure below reaches the target schema without editing any
existing revision.

## Procedure

```bash
# 0. Single-head check (works offline — no database needed).
uv run alembic heads
# Expected: exactly one head, currently `0017 (head)`.
# CI gate: tests/unit/test_migration_chain.py (single head + sequential IDs).

# 1. Mark the unrunnable revision applied, on the EMPTY database.
uv run alembic stamp 0005

# 2. Upgrade to the single head (same command as deploy: Makefile `migrate-up`,
#    README.md, .github/workflows/test.yml).
uv run alembic upgrade head
```

Why this works: stamping `0005` skips exactly that revision; the upgrade
then executes everything else — `0001`–`0004` and `0006`–`0011` (creating
chat, document-vector, unified document/chunk, outbox, billing and credit
relations), the `0011`/`0012` merges, and `0013`–`0017` on top.

## Revisions deliberately skipped, and what each would have created

| Revision | Would have created | Why skipping is safe / intended |
|---|---|---|
| `0005` | `parent_documents`; all `clauses` ALTERs, `UPDATE`s, FK and indexes (`clauses_bm25_idx`, `clauses_embedding_idx`) | Unrunnable on any database — `clauses` is created by no revision. `parent_documents`/`clauses` fate belongs to the search-consolidation change; this change creates no DDL for them. |
| `0004` | `search_documents`, `search_chunks` + their indexes | Skipped as a side effect of the `0005` stamp (it is reachable only through `0005`). Recreated in ORM-compiled shape by `0014` (deliberately without a `search_chunks` index — see its docstring). |

## Extension preconditions

The first executed revision that needs extensions is `0003`, which builds a
`diskann` index but does **not** create `vectorscale` — so before step 2,
the instance must already provide, or the role must be able to create:
`vector`, `vectorscale`, `pg_trgm`, `pg_textsearch`, `uuid-ossp`.
Per the trust table in `0013`'s docstring: `vector`, `vectorscale` and
`pg_textsearch` are **not** trusted (a non-superuser cannot install them —
they must come with the image, as on Timescale Cloud); `pg_trgm` and
`uuid-ossp` are trusted (any role with `CREATE` on the database can install
them). `CREATE EXTENSION IF NOT EXISTS` skips existing extensions but does
**not** soften missing privilege.

## Authoritative-revision notes (`0012` / `0013` docstrings)

- `0012` joins the two heads with an empty body; **reversal below it is
  unsupported** — the reversals would drop relations that were never created.
  `downgrade` past this point is forbidden, not discouraged.
- `0013` is the single revision that defines the whole target shape:
  outbox relations **first** (they repair the two `500`ing public auth
  endpoints independently of everything below), then the four extensions,
  then `documents`/`chunks` with `chunks.updated_at`, then the retrieval
  indexes under their exact query-contract names (`chunks_bm25_idx`,
  `chunks_embedding_idx`, `chunks_search_text_trgm_idx`). All DDL is
  `IF NOT EXISTS`, and its `downgrade()` is an intentional no-op.
- `0015` drops the server default on `chunks.updated_at`
  (`ALTER TABLE chunks ALTER COLUMN updated_at DROP DEFAULT`) — default
  ownership is ORM-side, mirroring `documents.updated_at`; see its docstring.

## Known fresh-vs-deployed divergence

On a fresh database `0002` executes (renaming `document_vectors.metadata`
to `meta_data`), while `0014`'s table-level `IF NOT EXISTS` then no-ops —
so a fresh instance keeps `meta_data` where the deployed instance (which
never ran `0002`) holds the ORM-declared `metadata`. `alembic check` will
report this drift on fresh environments; it is inherited from the
phantom-branch history, not from this procedure.
