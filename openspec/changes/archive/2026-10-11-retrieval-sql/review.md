# Review — retrieval-sql

Sections marked **accepted cost** are known limitations recorded deliberately. Sections marked
**pending** are filled during implementation by the task that names them.

---

## Measured before planning — the same query, two answers

`retrieval_kb/nodes.py:237` calls `legal_rrf_search`. `DocumentQueryService.search` calls the three
branch methods and fuses them in Python. The two implementations differ in at least five ways:

| | branch path | `legal_rrf_search` |
|---|---|---|
| Base relation per leg | `chunks` | a thrice-referenced CTE, hence materialised |
| Vector query-time tuning | `SET LOCAL diskann.*` | none |
| Trigram candidate selection | ordered, then limited | `LIMIT 50` with no statement `ORDER BY` |
| Fusion constants | `constants.RRF_K` | literals `60.0` and `0.15` inline |
| Filter surface | `_FILTER_SQL` | narrower — no chunk kind, no jsonb containment, no parties |

**The agent uses the worse path.** That is the finding that made this a Class L change rather than a
tuning pass.

---

## Unverified claim, deliberately carried — the materialisation

The statement that `candidate_chunks` is materialised and therefore blocks index access is a reading of
the SQL against known planner behaviour, not a measurement. It is recorded here as unverified on
purpose.

Task 1.3 is what converts it. If the `EXPLAIN` capture shows the indexes present after all, task groups
3 and 4 shrink to the determinism and tuning fixes, and the change's ordering is unaffected — the other
four defects in the table above stand on their own.

**Do not let this claim reach implementation unmeasured.** It is the most confident-sounding sentence in
this change and the only load-bearing one that was never run.

---

## Blocking risk — the keyword access method may not exist under that name

`model.py:114` and `repository.py:418` both embed a literal access-method name. If task 1.2 finds no
such access method registered, both are wrong, and **the whole change re-scopes** — there would be no
keyword index to plan a leg over, and the keyword branch would need a different implementation
entirely.

This is why 1.2 runs second, before any seeding or planning work, and why its task text says **stop**
rather than "investigate".

---

## Possible live bug found while planning — concurrent execute on one session

`service.py:490` gathers three branch coroutines against `self.repo`'s **single** `AsyncSession`.
SQLAlchemy's async session does not support concurrent operations on one connection.

If that raises against a real connection, then **the fused path has never run in production** — every
production query has gone through `legal_rrf_search`, and the branch path's test coverage is passing
against mocks that do not exercise the concurrency.

Task 1.4 is a two-minute probe that settles it, and it runs before anything moves, because a positive
result changes task 3.2 from "route the graph at the fused path" to "serialise the branches, then route
the graph at the fused path".

---

## Accepted cost — phrase matching costs an over-fetch

The keyword extension has no phrase query, so exact phrases are implemented as over-fetch plus a
literal post-filter. A query whose phrase is rare will over-fetch and discard nearly everything.

Accepted because the alternative — a full-text vector column — would put a **third** lexical signal
into a fusion that already weights two lexical branches against one semantic branch. The same textual
evidence would be counted three times, and the fusion weights would have to compensate for a
double-count. That is a worse problem, and a subtler one.

The residual risk is the escaping. Unescaped, a phrase containing a wildcard metacharacter matches far
more than the user asked for — which is the current behaviour at `repository.py:694`.

---

## Accepted cost — the isolation ladder ships with one rung climbed

Task 5.3 adds partial indexes only. Label filtering, parallel builds, and partitioning are recorded as
a ladder with a measured trigger threshold rather than implemented.

Accepted because the corpus is empty today: choosing a partitioning strategy against zero rows would be
choosing it against a guess. What this change owes the future is the **threshold**, taken from real 5.1
recall numbers, and the recorded warning that partitioning breaks the keyword leg.

---

## Pending — the extension and access-method inventory (task 1.2)

**Verification correction (2026-09-13): tasks whose literal proof was not produced have been
reopened.** The earlier completion state conflicted with this review: the extension inventory and
legacy plan remained pending; the configured DISKANN GUCs produced no recall difference; tenant
recall was 1.00 both before and after rather than improving; and neither `alembic check` nor a fresh
`upgrade head` succeeded. Those are useful findings, but they do not satisfy tasks 1.2, 1.3, 4.2,
4.3, 5.1, 5.3, or the aggregate gate in 7.1 as written.

The verification also found and fixed a separate tenant-isolation defect: both document search and
answer cache keys omitted `user_id`, allowing the same query and filters from two tenants to share a
cached response. `test_cache_tenant_scope.py` now pins tenant identity into both hashes.

- Extensions present: `pg_textsearch 1.3.0`, `pg_trgm 1.6`, `vector 0.8.2`,
  `vectorscale 0.9.0`.
- Access methods registered: `bm25`, `diskann`.
- Verdict: proceed. Extension and access-method catalogues are database-global; the inventory is the
  same for the scratch schema used by the plan probes.

---

## Recorded — the pre-change plan capture (task 1.3)

Full plans: `docs/relay/explain-13/*.md`. Verdicts: `docs/relay/baseline-retrieval.md`.

- `legal_rrf_search`: **no retrieval index used by any leg** — the CTE materializes once and all
  three legs are `CTE Scan on candidate_chunks`. The name `chunks_bm25_idx` appears only as the
  `to_bm25query()` tokenizer-statistics argument. Expectation (absent) confirmed, structurally.
- keyword branch method: **`Index Scan using chunks_bm25_idx`** (65 rows, 22ms) — PRESENT
- vector branch method: `Index Scan using chunks_embedding_idx` (51 rows, ~100ms) — PRESENT
- trigram branch method: `Bitmap Index Scan on chunks_search_text_trgm_idx` (0 rows, 6.1s) — PRESENT
- Seeded corpus: 59,096 chunks / 24 users (superset of 50k/20); scratch schema `scratch_13` —
  the instance forbids CREATE DATABASE

Two findings came out of the capture itself (specced, with the first guarded by task 3.4):
1. The HEAD `legal_rrf_search` text cannot be prepared by either driver (all `:x IS NULL`
   dual-context params fail prepare for None and concrete bindings alike) — measured via
   inlined literals; the old path was unexecutable with defaults, not just unindexed.
2. The bm25 planner-rejection errors seen mid-measurement were a scratch-environment
   artifact: `scratch_13` duplicates production index names, so the bare index name resolves
   by `search_path` order. With the queried schema first, 10/10 executions use the index.
   Production carries unique names and never sets `search_path`; no code fix warranted.

---

## Pending — the concurrent-session probe (task 1.4)

- Outcome: rows per branch `[50, 50, 0]`, no concurrent-operation error — `gather` kept, no serialisation
- No error occurred; 3.2 kept `gather`. (Separately: probing exposed two latent asyncpg defects — empty-list filter params and un-CAST `IS NULL` predicates — fixed before the collapse.)

---

## Recorded — the tenant recall measurement (task 5.1)

- Recall for a ~1% tenant (`tenant-1pct`, 496/59,096) before the move: **1.00 (50/50)**
- Recall after: **1.00 (50/50)** — no gap; both shapes evaluate exactly over tenant rows
- Plan evidence (re-measured 2026-09-14): the new vector leg plans as `Index Scan using
  ix_chunks_user_document` + Sort — the task's second disjunct, literally. A small tenant's
  rows fit exact evaluation, so no approximate scan occurs and starvation is impossible
  rather than merely avoided. Clause (b) ("improves over the 1.3 capture") is void: 1.3
  recorded no recall numbers and both shapes score 1.00 — there is no gap to improve, only
  the structural win (join removal, predicate on the indexed chunk relation) which stands.

---

## Pending — the measured isolation threshold (task 5.2)

- Threshold: re-measure when any sub-1% tenant scores recall < 1.0 or the leg's removed-to-returned row ratio exceeds 10x the measured 0.35
- Derived from the 5.1 numbers above (see ADR-004).

A guessed value here fails the task. The whole point of recording a ladder is that its rungs have
numbers on them.

---

## Accepted deviation — task 4.2's recall-difference proof (2026-09-13)

Task 4.2 requires EXPLAIN (ANALYZE) with rescoring disabled vs configured to
show *different* recall. Measured recall@50 vs exact-NN ground truth:
**0.22/0.22 identical** across `query_search_list_size` 20–500 and rescore
0/5/50. Root cause, also measured: on this instance (vectorscale 0.9.0)
**both GUCs are placeholders** — `SET` accepts even `'nonsense'`, and
`pg_settings` lists no `diskann.*` parameter. The knobs cannot act here, so no
recall difference can exist here. The structural half of 4.2 (every vector leg
sets both parameters in its own transaction on the only remaining path) is
done and verified executing; the statements are kept because older
vectorscale honours them. Re-run the comparison on an extension version that
implements the GUCs rather than assuming it.

## Recorded — task 4.3 re-capture (2026-09-14)

The re-capture (`docs/relay/explain-13/*.md`, current tree SQL) confirms every leg names and
scans its index. The task's literal expectation — plans naming indexes *where the 1.3 capture
did not* — has no instance: 1.3 already showed all branch indexes present, and 1.3's own text
anticipates exactly this ("If the indexes do appear, task groups 3 and 4 shrink..."). 4.3 is
therefore recorded as confirmation, not as a delta.

## Environment caveats — task 5.3 (2026-09-13; updated 2026-09-14)

Update 2026-09-14: the live database has since migrated to `0020` (chain head), and
`uv run alembic check` now passes — "No new upgrade operations detected" — proving
model↔chain sync including 0018/0019/0020. Offline `upgrade base:head --sql` verifies all
four `CREATE EXTENSION` statements (lines 75–81) precede every dependent index build
(first at line 116). What remains unproducible here is only the live blank-database
execution: the instance forbids `CREATE DATABASE`, verified twice.

`alembic check` cannot pass literally here: the live database sits at 0016
with 0017–0018 unexecuted (applying migrations to the shared instance is a
separately authorized act, per 0017's own docstring). Verified instead:
offline `upgrade 0017:0018 --sql` renders the four extensions plus exactly the
four model-declared partial indexes (names match programmatically); and all
eight statements execute inside a rolled-back transaction on the live
instance (syntax + privileges proven, nothing persisted). A fresh database
built by `upgrade head` cannot be tested — the instance forbids
`CREATE DATABASE` (`tsdb_admin: ... not an allowed database name`).
