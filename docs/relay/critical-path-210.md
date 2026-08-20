# Item 210 — critical path

`tests/performance/todo.md:284` — *"fix ingestion -> docuements -> tools -> cognee"*.

One line of backlog. Four hops. Five openspec changes. This file is the single place that says **what actually
blocks it** and **in what order the work must run**, because neither answer is visible from inside any one change.

Correction to my own earlier record: 210 is at **`:284`**, not `:297`. `:297` is sub-todo **(j)** (tenacity/retries).

---

## The reading of the arrow

I flagged early that `ingestion -> documents -> tools -> cognee` reads two ways: a **work order** (fix these in
sequence) or a **data flow** (each feeds the next). That ambiguity is real but **non-binding** — the two readings
agree on sequence. Data flows downstream, so the work must too:

```
ingestion          documents           tools              cognee
  writes rows  →   defines the     →   read that      →   consumes what
                   schema they         schema             tools surface
                   land in
```

| Hop | Change | What it is |
|---|---|---|
| ingestion | **1** `ingestion-pipeline-unification` | 368-line tasks.md |
| documents | **2** `documents-unified-schema` | 388-line tasks.md |
| tools | **3** `agent-tools-unification` | 281-line tasks.md |
| cognee | **4** `cognee-agent-memory` | 285-line tasks.md |
| *prerequisite* | **0** `cleanup-foundation` | in flight |

The user's follow-on constraint — *"my old tools and ingestion is written according to old schema… migrated to new
document based schema and tools"* — makes **change 2 the hub**, not merely the second hop. Hops 1, 3 and 4 all
target the schema change 2 defines.

---

## The blocker: all four hops terminate in tables that do not exist

Full evidence in `findings-database.md` §11. Measured against the live Timescale Cloud instance, read-only:

- `alembic_version` holds **one row: `0004`**
- **16 public tables** — 15 billing + `alembic_version`
- **all eleven** chain tables absent: `documents`, `chunks`, `document_vectors`, `search_documents`,
  `search_chunks`, `parent_documents`, `clauses`, `chat_messages`, `chat_sessions`, `outbox_events`,
  `dead_letter_events`
- **zero `bm25` indexes**

### Why — the chain is branched and its middle is unrunnable

```
<base> → c0c17c6eb1cc → 2bc7726317f6          ← BRANCHPOINT
                             ├→ 8a7d9b1c2e3f → 9f4a1b7c6d2e → 0001 → 0002 → 0003 → 0004   ← HEAD 1 (stamped)
                             └→ a71f0d7d9c12                                              ← HEAD 2 (never applied)
```

`9f4a1b7c6d2e` **cannot execute.** It manipulates a `clauses` table that nothing creates —
`:63 batch_alter_table("clauses")`, `:101-102 UPDATE clauses SET …`, `:103-105` three `alter_column`, `:108` FK,
`:115-125` four indexes, `:132 CREATE INDEX clauses_bm25_idx ON clauses`, `:138 clauses_embedding_idx … USING
diskann` — while its own `op.create_table` at `:28` creates **`parent_documents`**. No migration creates `clauses`;
no ORM declares `__tablename__ = "clauses"`.

**Proof it never ran:** it is marked applied, yet `parent_documents` is absent.

So the `stamp` to `0004` was **a workaround for a broken migration**, and it silently marked five revisions' tables
as applied. Three consequences:

1. `upgrade heads` today creates only `documents` + `chunks` — `a71f0d7d9c12` is the one revision that executes.
   That is also exactly where change 0's **F4(d)** `diskann`/`vectorscale` hazard bites.
2. The other tables are unreachable by `upgrade` **forever** — no future revision can reach them through a
   revision already marked applied.
3. The stamp-down route **re-enters the unrunnable revision**, so `clauses` must be resolved *first*.

This is a **schema-reachability defect**. No code repair in any of the five changes fixes 210 without it.

### What it costs each hop, today

| Hop | Live consequence of the missing schema |
|---|---|
| **ingestion** | Every write targets `documents`/`chunks` — absent. `outbox_events`/`dead_letter_events` absent too, so `with_outbox` (`auth/service.py:503-524`) fails → **two public auth endpoints 500 today**. |
| **documents** | The unified schema *is* the repair. `clauses` is promoted from tidy-up (item 184, Option A+) to **load-bearing**, because the stamp-down route runs through it. |
| **tools** | Readers point at absent tables: `precedent_tools.py:237` is a stub returning `[]`; `search/repository.py:308-405` reads `clauses`. Four competing result envelopes. Three Saul agents get `tools=[]` (`shared/langgraph_layer/agent_saul/factory.py:116,122,128`). |
| **cognee** | `cognify` has **zero call sites** — nothing was ever ingested. Plus 3072-vs-768 embedding dims and a `lancedb` default instead of the app's Postgres. |

---

## Execution order — and why it is not 0→1→2→3→4

Three hard sequencing constraints cut across the numbering. Two were known; the third is new from §11.

| # | Constraint | Source |
|---|---|---|
| S1 | Change 3's `shared/agents/**` importer rewrite **must precede** change 0's deletion of the 30-byte shadow, or `registry.py:41-46`'s eager imports raise `ImportError` at boot | D6.1 |
| S2 | Change 0's migration repair **needs change 2's `clauses` decision** (item 184, Option A+) — stamp-down re-enters `9f4a1b7c6d2e` | **§11, new** |
| S3 | Change 4's consolidation beat entry is **inert until change 1 lands** the worker/beat runtime | A3 |

S2 is the one that reorders things: it makes a **change-2 decision** a precondition for a **change-0 task**. The
resolution is that the decision is *paper only* — an ADR, no code — so it can be pulled forward without dragging
change 2's implementation with it.

### The resulting order

```
 0. change 2's clauses ADR (item 184, Option A+)      ← paper only, pulled forward per S2
 1. change 0: migration repair, proven on a local scratch DB first
 2. change 3: shared/agents/** importer rewrite       ← per S1, before any deletion
 3. change 0: shadow deletion + remaining foundation
 4. change 1: ingestion  ─┐
 5. change 2: documents   ├─ the four hops, now against a reachable schema
 6. change 3: tools       │
 7. change 4: cognee     ─┘
```

Steps 4–7 keep 210's own order. Steps 0–3 exist only because the schema has to be reachable before any of them
mean anything.

**On live-database authorization:** the repair is proven against a **local scratch database first**, which needs no
permission. The live question goes to the user *once the repair is proven*, not before. The single authorization for
`CREATE EXTENSION IF NOT EXISTS pg_textsearch` is **spent**; any further DDL needs fresh authorization.

---

## Two obligations that ride along

Both fall out of the `pg_textsearch` probe (`findings-database.md` §10, Q-B) and both are change 0's:

1. **No `bm25` index exists anywhere** in the database.
2. Because the two-argument `to_bm25query` overload takes the index name as a **literal SQL argument** — pinned at
   `search/constants.py:15` — the index name is part of the **query contract**, not a naming convention. An index of
   the right shape under a different name **silently matches nothing**. Both indexes must be created by exact name.

The good news from the same probe: the repo's existing BM25 SQL is **already correct** — right overload, right
negation, right ordering. No rewrite.

---

## Related backlog items that are really this item

- **220** (`:302`) — *"check the alembic warning having 2 heads"*. Filed as housekeeping; it is **210's root cause**.
- **184** (`:272-275`) — clause code disposition. Promoted to load-bearing by S2.
- **185** (`:276`) — remove `ts_vector`. Pure subtraction: `content_tsv` is a STORED generated column with a live
  GIN index and **zero readers**.
