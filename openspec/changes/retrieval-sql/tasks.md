# Tasks — retrieval-sql

## How to read the Proofs

1. **Never use a test-process exit code as a Proof of test outcome.** Compare **summary pass and
   failure counts** against the baseline captured in task 1.1, re-measured immediately before the task.
2. **Never prove a schema fact by rendering migrations offline.** Prove it against a database built
   from `alembic upgrade head`.
3. **Never make a Proof depend on a durable outbound event firing.** The outbox tables do not exist.
4. **The live database may be used** — zero data, zero users, ruled available.
5. **No Proof in this change may cite an answer-quality metric.** Only tier-1 retrieval metrics exist.
6. **Every plan claim is measured, never asserted.** The materialised-CTE diagnosis below is a reading
   of the SQL; task 1.3 is what makes it a fact.

**Blocked by `rag-tree-repair`** (the tree does not import today) and **`rag-eval-harness`** (task 5.1
claims a recall improvement, which needs a before-number).

## 1 Baseline — measure before touching anything

- [ ] 1.1 Capture the gate baseline into `docs/relay/baseline-retrieval.md`:
  `uv run pytest tests/unit/documents tests/unit/shared/langgraph_layer -q`,
  `uv run ruff check --no-cache src/`, `uv run ty check src/`.
  **Proof:** all three counts recorded verbatim. Every later Proof compares to these, never to an
  absolute.
- [ ] 1.2 **Settle whether the keyword access method exists under the name the code uses.** Create a
  scratch database on the live instance, run `CREATE EXTENSION IF NOT EXISTS` for the vector,
  vector-scaling, trigram, and keyword extensions, then
  `SELECT name, installed_version FROM pg_available_extensions WHERE name IN (...)` and
  `SELECT amname FROM pg_am WHERE amname IN ('bm25','diskann')`.
  **Proof:** both `amname` rows present; the version table recorded. **If the keyword access method is
  absent under that name, stop** — `model.py:114` and `repository.py:418` both embed the literal, and
  the whole change re-scopes.
- [ ] 1.3 Build the scratch schema from the ORM, seed roughly fifty thousand synthetic chunks across
  twenty users, and capture `EXPLAIN (ANALYZE, BUFFERS)` for `legal_rrf_search` and for each of the
  three branch methods.
  **Proof:** the captured plans record whether each index appears. The expectation is absent for
  `legal_rrf_search` and present for the branch methods — **but the expectation is not the Proof.** If
  the indexes do appear, task groups 3 and 4 shrink to the determinism and tuning fixes and the
  ordering is unaffected.
- [ ] 1.4 **Run the three-branch fused path once against a real session on the scratch database.**
  `service.py:490` gathers three `execute()` calls on one `AsyncSession`.
  **Proof:** record whether it returns rows or raises a concurrent-operation error. A raise means the
  fused path has **never run in production**, and task 3.2 must serialise the branches before routing
  the graph through it.

## 2 Guards — land before any behaviour moves

- [ ] 2.1 `tests/unit/documents/test_bm25_sign_convention.py`: static assertions over `repository.py`
  query text that the relevance expression is ordered ascending and filtered below zero, plus a
  fusion-level test that a more-negative raw score outranks a less-negative one.
  **Proof:** the test passes; inverting the ordering in the source makes it fail.
- [ ] 2.2 `tests/unit/documents/test_no_tsvector_in_app_code.py` asserting no full-text vector, query,
  or construction call appears under `src/app/`. Exclude migrations as an immutable historical record
  and **state the exclusion in the test** rather than leaving it implicit.
  **Proof:** the test passes now; adding a vector-construction call to any `src/app/` file makes it
  fail.
- [ ] 2.3 Pin the filter surface: every branch method's SQL contains the shared filter block, and every
  key in the built filter parameters is consumed by every branch.
  **Proof:** `uv run pytest tests/unit/documents -q` gains one test over the 1.1 baseline; removing one
  predicate from one branch fails it.

## 3 Collapse to one fused path

- [ ] 3.1 Add per-leg weights to `reciprocal_rank_fusion` as a **keyword-only argument defaulting to
  unweighted**, with named constants in `constants.py`, weighted toward the lexical legs.
  **Proof:** `uv run pytest tests/unit/documents/test_fusion.py -q` — existing tests pass unchanged,
  because the default is unweighted and `search_legal_precedents.py:188` is therefore untouched — plus
  a new test showing the order changes when a weight changes.
- [ ] 3.2 Point `make_hybrid_retrieval_node` at the shared fused path instead of `legal_rrf_search`,
  mapping the query plan's vector and keyword weights onto the weight arguments, and its phrase and
  threshold onto the branch inputs. If 1.4 recorded a concurrent-operation error, serialise the branch
  execution here.
  **Proof:** `uv run pytest tests/unit/shared/langgraph_layer/test_retrieval_retry_shape.py -q` passes;
  a new test asserts the graph path and `DocumentQueryService.search` return **identical chunk-id
  order** for one query.
- [ ] 3.3 Delete `legal_rrf_search` and its executable stub at `src/app/examples/policy_examples.py:142`.
  **Proof:** `rg -n 'legal_rrf_search' src/ tests/` returns nothing;
  `uv run python -c "import app.main"` exits `0`; the pytest summary pass count is ≥ the 1.1 baseline.
  Note that `policy_examples.py:122-160` is executable and asserts branch names — it breaks on any
  branch-registry edit.

## 4 Index and query tuning

- [ ] 4.1 Give every leg a deterministic tiebreaker in both its ranking-expression ordering and its
  statement ordering.
  **Proof:** the same query run twice on the scratch database yields byte-identical chunk-id order.
- [ ] 4.2 Move the approximate-search query-time parameters onto the fused path so every vector leg sets
  them in its own transaction.
  **Proof:** `EXPLAIN (ANALYZE)` with rescoring disabled versus the configured value shows different
  recall against an exact nearest-neighbour ground truth computed once; both recorded against 1.3.
- [ ] 4.3 Re-capture the full `EXPLAIN` set from 1.3.
  **Proof:** every leg's plan now names its index where the 1.3 capture did not.

## 5 Tenant isolation

- [ ] 5.1 Move the tenant predicate onto `chunks.user_id` in every leg — the column and its index
  already exist, so **no migration is needed for this**. Keep the document join only where a document
  column is projected.
  **Proof:** `EXPLAIN` shows the vector leg filtering before the approximate scan or using
  `ix_chunks_user_document`; and recall for a user owning roughly one percent of chunks, measured
  against exact nearest-neighbour ground truth, improves over the 1.3 capture.
- [ ] 5.2 Record the isolation-ladder decision in `adrs.md`: partial indexes for the three-to-five
  stable jurisdiction and document-kind values first; approximate-index label filtering **or** parallel
  builds — mutually exclusive — for many tenants; list partitioning last. Record that keyword
  statistics are **partition-local**, so a partitioned keyword leg produces scores that are not
  comparable across partitions.
  **Proof:** the ADR names a **measured** trigger threshold taken from the 5.1 recall numbers, not a
  guessed one.
- [ ] 5.3 Add the chosen partial indexes in a migration that also carries `CREATE EXTENSION IF NOT
  EXISTS` for all four extensions, and register them in `model.py.__table_args__`.
  **Proof:** `uv run alembic check` proposes no diff; and a fresh scratch database built by
  `alembic upgrade head` with **no pre-installed extensions** reaches head successfully.

## 6 Phrase search

- [ ] 6.1 Move the phrase post-filter into the keyword leg as over-fetch plus an **escaped** literal
  pattern filter — escaping the wildcard and escape characters in the phrase — and expose it on the
  shared branch input so both callers get it.
  **Proof:** `uv run pytest tests/unit/documents -q` gains a test where a phrase containing a wildcard
  metacharacter matches literally, and a chunk containing the words separately but not the phrase is
  excluded.
- [ ] 6.2 Record in `design.md` that the keyword extension supports neither phrase nor boolean query
  syntax, that over-fetch plus post-filter is the vendor-prescribed remedy, and that a full-text vector
  column is therefore **not** added because it would double-count the lexical signal in a three-branch
  fusion.
  **Proof:** the vendor citation appears in `design.md`; the 2.2 guard keeps it true.

## 7 Close out

- [ ] 7.1 **Proof:** `openspec validate retrieval-sql --strict` exits `0`;
  `uv run ruff check --no-cache src/`, `uv run ty check src/`, and `uv run pytest -q` are each equal to
  or better than the 1.1 baseline.
- [ ] 7.2 Re-verify the named blast radius:
  `tests/unit/documents/{test_hybrid_search_failure,test_fusion,test_rag,test_vector_width_configured}.py`,
  `tests/unit/shared/langgraph_layer/test_retrieval_retry_shape.py`,
  `tests/unit/test_feature_error_exhaustiveness.py`, and the executable stubs in
  `src/app/examples/policy_examples.py`.
  **Proof:** each shows zero failures.
