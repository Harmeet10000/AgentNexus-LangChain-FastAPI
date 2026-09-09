# ADRs — retrieval-sql

## ADR-001 — The three branch methods become the single implementation; the monolithic CTE is deleted

**Status:** accepted.

**Context.** Two implementations of the same three-branch fusion are reachable: three branch methods
over base `chunks` fused in Python, and `legal_rrf_search`, a single weighted CTE the retrieval graph
calls. The CTE references `candidate_chunks` three times, so Postgres materialises it and the keyword
and vector indexes become unreachable from the query.

**Decision.** Keep the branch methods as the source of truth for each leg's SQL and delete
`legal_rrf_search`. Route the retrieval graph through the shared fusion.

**Consequences.** Each leg can be planned over its own index, and its ranking, ordering, and limit live
in one statement. The failure semantics that `_fuse_search_branches` already implements — a branch
returning nothing is not a branch raising — survive, which one statement could not provide.

The rejected alternative was to make the CTE the single path and delete the Python fusion. It loses on
both counts: a single statement fails all-or-nothing, collapsing "no keyword matches" (normal on a
semantic query) into a query failure; and the CTE shape is the thing that lost index access, so
consolidating onto it would require restructuring it anyway.

## ADR-002 — The materialisation diagnosis is measured before it is acted on

**Status:** accepted.

**Context.** The claim that `candidate_chunks` is materialised and therefore blocks index access is a
reading of the SQL against known planner behaviour. It is not a measurement.

**Decision.** Task 1.3 captures `EXPLAIN (ANALYZE, BUFFERS)` for the CTE path and each branch method,
on seeded data, **before** any behaviour moves. Every later plan Proof compares to that capture.

**Consequences.** If the indexes do appear — a planner version that inlines despite three references —
task groups 3 and 4 shrink to the determinism and tuning fixes, and the change's ordering is unaffected.
The diagnosis is falsifiable and the plan survives being wrong about it, which is the property that
matters. The cost is one task producing no product behaviour.

## ADR-003 — Tenancy moves onto the chunk relation, and it needs no migration

**Status:** accepted.

**Context.** Every leg reaches tenancy through a join to `documents`. `chunks.user_id` and
`ix_chunks_user_document` already exist (`model.py:155`). For the vector leg the join means filtering
happens **after** the approximate scan, so a tenant owning a small fraction of the corpus can have their
candidate pool consumed by other tenants' rows before filtering.

**Decision.** Apply the tenant predicate on `chunks` in every leg. Keep the document join only where a
document column is projected.

**Consequences.** Recall stops degrading for small tenants — the failure mode hardest to notice in
testing and easiest for a customer to hit, because it produces fewer results with no error and no
signal. No migration is required, which was not obvious before measurement. Task 5.1's Proof is a
measured recall comparison against exact nearest-neighbour ground truth rather than a plan inspection,
because a plan can look correct while recall stays poor.

## ADR-004 — The isolation ladder reaches partitioning last

**Status:** accepted, with the trigger threshold to be filled from task 5.1's measurements.

**Context.** Scaling tenant isolation past partial indexes has several options with very different
costs.

**Decision.** Climb in this order: partial indexes on the three-to-five stable jurisdiction and
document-kind values; then approximate-index label filtering **or** parallel index builds, which are
mutually exclusive; then list partitioning, last.

**Consequences.** Partitioning is last because it **breaks the keyword leg**. Keyword relevance
statistics on a partitioned table are partition-local: each partition scores against its own corpus
statistics, so a cross-partition ordering by relevance compares numbers that were never on the same
scale. The failure is silent — the query succeeds and returns plausible rows in the wrong order. The
threshold for climbing each rung must come from the 5.1 recall numbers rather than from a guess, which
is why this ADR is accepted with a value still to be measured.

## ADR-005 — No `tsvector` column is added

**Status:** accepted (user ruling: remove `tsvector`, todo 185 closed as verify-only with a guard).

**Context.** The keyword extension provides no phrase query and no boolean query syntax. The obvious
reach is for a full-text vector column to supply them.

**Decision.** Do not add one. Implement exact phrases as over-fetch plus an escaped literal
post-filter, which is the vendor-prescribed remedy. Enforce the prohibition with a test over
`src/app/`, excluding migration history as an immutable record and stating that exclusion in the test.

**Consequences.** The lexical signal is not double-counted. The fusion already has two lexical branches
— keyword and trigram — against one semantic branch; a third would mean the same textual evidence
counted three times, and the weights would have to compensate for a double-count, which is a worse
problem than the one it solves. The cost is that phrase matching costs an over-fetch, and the escape
handling must be correct: unescaped, a phrase containing a wildcard metacharacter matches far more than
the user asked for.

## ADR-006 — Fusion weights are keyword-only and default to unweighted

**Status:** accepted.

**Context.** Per-leg weights are needed so the retrieval graph's query plan can express a lexical or
semantic preference. But `reciprocal_rank_fusion` has a **second consumer** at
`search_legal_precedents.py:188`.

**Decision.** Add weights as a keyword-only argument defaulting to unweighted, with named constants in
`constants.py`.

**Consequences.** The second consumer is untouched and needs no edit, so the weight work cannot
silently change agent-tool ranking. The literals `60.0` and `0.15` currently embedded in
`legal_rrf_search` — sitting beside `constants.RRF_K`, which is a second definition of the same
constant — disappear with the method. The cost is that "unweighted" is now a live default that must
stay meaningful.

## ADR-007 — `legal-corpus-retrieval` is widened rather than replaced

**Status:** accepted.

**Context.** The archived requirement forbids **an agent tool** from introducing a second ranking or
fusion implementation. The second implementation arrived through a repository method, satisfying the
requirement's letter exactly while defeating its purpose.

**Decision.** Widen the binding to every caller — agent tool, repository method, service, or graph node
— and add the observable consequence the original left implicit: two callers issuing the same query
with the same filters receive the same ranked identifiers in the same order. Preserve both existing
scenarios verbatim.

**Consequences.** The requirement now names the failure that actually occurred rather than the one that
was anticipated. Preserving the scenarios verbatim is not courtesy: a `## MODIFIED` block replaces its
requirement wholesale on archive, so an omitted scenario is silently deleted from the baseline and
strict validation cannot detect the loss.

## ADR-008 — This change adds no columns

**Status:** accepted.

**Context.** Two changes in this cluster migrate `chunks`: this one for indexes and extensions,
`ingestion-chunking` for identity columns. Authored from the same head, Alembic produces a branch.

**Decision.** Restrict this change's schema work to index and extension DDL. Add no columns.

**Consequences.** A rebase onto whichever migration lands first is mechanical rather than a merge of two
schema intents. Task 5.3's `alembic check` is where a branch would first become visible. The cost is
that the shared filter surface this change delivers is where `ingestion-chunking`'s version predicate
will later land — a seam recorded in both changes rather than resolved in either.
