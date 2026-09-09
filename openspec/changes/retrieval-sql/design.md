# Design — retrieval-sql

## The CTE optimisation fence is the whole diagnosis

`legal_rrf_search` (`repository.py:596-728`) is written as a single statement: a `candidate_chunks` CTE
that applies the filters once, then three legs selecting `FROM candidate_chunks`, then a weighted
reciprocal-rank fusion over the three.

Read as SQL that is elegant. It is also the reason the query cannot use its indexes.

Postgres inlines a `WITH` clause only when it is referenced **exactly once** and is free of side
effects. `candidate_chunks` is referenced **three times**, so the planner materialises it into a
temporary relation. Every leg then scans that temporary relation — and `chunks_bm25_idx` and
`chunks_embedding_idx` are built on `chunks`, not on the temporary result. They are simply not
reachable from this query shape, no matter how the leg's ranking expression is written.

This is the non-obvious part, and it is worth stating precisely because it inverts the intuition: the
CTE looks like an optimisation (filter once, reuse three times) and is in fact the thing that makes
the query slow. Filtering once costs a materialisation; filtering three times costs three index scans
that each start narrow.

Three consequences follow directly:

- **The vector leg never gets its tuning.** No `SET LOCAL diskann.*` runs on this path, so the
  approximate search uses server defaults rather than the configured search-list size and rescore
  count that the branch path sets.
- **The trigram leg ranks an arbitrary sample.** `LIMIT 50` with no statement-level `ORDER BY` takes
  fifty rows in plan order and ranks *those*. The rank is computed correctly over the wrong fifty rows.
- **Constants fork.** `60.0` and a `0.15` trigram weight are literals inside the query, sitting beside
  `constants.RRF_K`. Two definitions of the same constant is one definition too many.

**The Proof discipline this imposes:** all of the above is a reading of the SQL, not a measurement.
Task 1.3 captures `EXPLAIN (ANALYZE, BUFFERS)` before anything is touched. If the indexes *do* appear —
if a planner version inlines it after all — then task groups 3 and 4 shrink to the determinism and
tuning fixes, and the change's ordering is unaffected. The diagnosis is falsifiable and the plan
survives being wrong about it.

## Rejected shape — make the CTE the single path

The symmetric alternative is to keep `legal_rrf_search` as the one implementation and delete the Python
fusion. It is a smaller diff and it puts fusion in the database, which is usually the right instinct.

Rejected for two reasons:

1. **A single statement fails all-or-nothing.** `_fuse_search_branches` (`service.py:469-512`) already
   distinguishes "this branch returned nothing" from "this branch raised", and
   `tests/unit/documents/test_hybrid_search_failure.py` guards that distinction. In one statement, a
   keyword-leg failure is a query failure — there is no surviving-branch behaviour to specify. That
   distinction is load-bearing: an empty keyword result on a purely semantic query is normal, and
   collapsing it into a failure would make the fused path fragile in exactly the common case.
2. **The CTE shape is what lost index access.** Consolidating onto it would mean either keeping the
   fence or restructuring the statement so thoroughly that the "smaller diff" argument evaporates.

So the three branch methods become the source of truth, each over base `chunks`, each with its own
ordering and limit in the same statement as its ranking expression.

## Tenancy: the ANN candidate-pool starvation

`chunks.user_id` and `ix_chunks_user_document` **already exist** (`model.py:155`). Every search leg
nevertheless reaches tenancy through `JOIN documents ... WHERE d.user_id = :user_id`.

For the keyword and trigram legs that is merely a wasted join. For the vector leg it is a correctness
problem with a specific shape:

An approximate-nearest-neighbour index returns a fixed-size candidate pool of the globally nearest
vectors. Filtering *after* that scan discards every candidate belonging to another tenant. A user who
owns one percent of the corpus therefore expects roughly one percent of the pool to survive — and if
the pool is smaller than a hundred times the requested count, they get fewer results than exist, with
no error and no signal. Recall degrades exactly for the smallest tenants, which is the hardest failure
mode to notice in testing and the easiest for a customer to hit.

Moving the predicate onto `chunks.user_id` lets the planner either use `ix_chunks_user_document` to
narrow before scanning, or — where the extension supports it — push the label into the approximate
scan itself. Either way the pool is drawn from the tenant's own rows.

Task 5.1's Proof is a **measured recall comparison against exact nearest-neighbour ground truth** for a
one-percent tenant, not a plan inspection, because the plan can look right while recall stays poor.

## The isolation ladder, and why partitioning is last

Recorded as a ladder rather than a decision because the right rung depends on numbers this change
measures:

| Rung | When | Cost |
|---|---|---|
| Partial indexes on stable low-cardinality values | 3–5 stable jurisdiction / document-kind values | one index per value; trivial |
| Approximate-index label filtering **or** parallel index builds | many tenants | mutually exclusive; a real build-time cost |
| List partitioning | last resort | **breaks the keyword leg** — see below |

The last rung carries a trap that is easy to walk into. Keyword relevance statistics on a partitioned
table are **partition-local**: each partition's scores are computed against its own corpus statistics.
A cross-partition `ORDER BY relevance LIMIT n` therefore compares numbers that were never on the same
scale, and the mis-ordering is silent — the query succeeds and returns plausible rows in the wrong
order.

That is why the ADR must reach partitioning last, and why task 5.2's Proof requires a **measured**
trigger threshold from the 5.1 recall numbers rather than a guessed one.

## Phrase search without a full-text vector column

The keyword extension provides no phrase query and no boolean query syntax. The vendor's prescribed
remedy is to over-fetch ranked candidates and post-filter them with a literal pattern match.

The repository already does this at `repository.py:694` — but unescaped, and on only one path. Both
defects matter: unescaped, a phrase containing a wildcard metacharacter matches far more than the user
asked for; on one path only, the same phrase query behaves differently depending on which door it
entered, which is the change's central complaint in miniature.

**A `tsvector` column is not added**, and this is a decision rather than an omission. Adding one would
put a second lexical signal into a three-branch fusion that already has two lexical branches — keyword
and trigram — so the same textual evidence would be counted three times against one semantic branch.
The fusion's weights would then need to compensate for a double-count, which is a worse problem than
the one it solves. Task 2.2's guard test is what keeps this decision from eroding.

## Why the modification to `legal-corpus-retrieval` widens rather than replaces

The existing requirement says an **agent tool** shall not introduce a second ranking or fusion
implementation. Measurement shows the second implementation arrived through a **repository method**
instead.

That satisfies the letter of the requirement precisely while defeating its purpose — which is the most
useful kind of spec finding, because it is evidence about the requirement's shape rather than about the
code. The requirement was written from the assumption that agent tools were the likely source of
duplication. They were not.

So the fix widens the binding to every caller and adds the observable consequence the original left
implicit: two callers issuing the same query with the same filters get the same ranked identifiers in
the same order. Both original scenarios are preserved verbatim, because a `## MODIFIED` block replaces
its requirement wholesale on archive and an omitted scenario would be silently deleted.

## Migration discipline against `ingestion-chunking`

Two changes in this cluster migrate `chunks`: this one (indexes and extensions) and
`ingestion-chunking` (identity columns). Authored from the same head, Alembic produces a branch.

The mitigation is structural: **this change adds no columns.** A rebase onto whichever migration lands
first is therefore mechanical rather than a merge of two schema intents. Task 5.3's `alembic check` is
where a branch would first become visible.
