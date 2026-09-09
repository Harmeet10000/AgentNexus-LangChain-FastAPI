# Tasks — rag-eval-harness

## How to read the Proofs

Every task carries a **Proof**: a command whose output settles whether the task is done. Five rules
govern them across this cluster.

1. **Never use a test-process exit code as a Proof of test outcome.** The configured coverage floor is
   far above actual coverage, so a fully green suite still exits non-zero. Compare **summary pass and
   failure counts** against the baseline files recorded by `rag-tree-repair`, and re-measure
   immediately before the task.
2. **Never prove a schema fact by rendering migrations offline.** Offline rendering always starts from
   base, so it cannot show what one revision adds. Schema facts are proven against the live catalogue.
3. **Never make a Proof depend on a durable outbound event firing.** The outbox tables do not exist.
4. **The live database may be used.** Zero data, zero users, ruled available by the user. This
   supersedes the archived rule requiring a local scratch instance, and extends to nothing else.
5. **A green offline suite proves nothing about wiring.** Every offline Proof in this change runs
   against a fake retriever. Task 3.4 exists precisely because those Proofs cannot detect a harness
   connected to nothing.

Baseline files cited by name: `docs/relay/baseline-ruff-after.txt`, `docs/relay/baseline-ty.txt`,
`docs/relay/baseline-pytest.txt` — all produced by `rag-tree-repair`.

**Blocked by `rag-tree-repair`.** Nothing here can be collected until the tree imports.

## 1 Metric core, pure and without I/O

- [ ] 1.1 `src/app/shared/evaluation/metrics.py` — `recall_at_k`, `reciprocal_rank`, `ndcg_at_k`,
  `precision_at_k`, each over a `Sequence[str]` of ranked identifiers plus a set of expected
  identifiers. Fully annotated; no blanket type suppression.
  **Proof:** `uv run pytest tests/unit/shared/evaluation/test_metrics.py -q` summary shows zero
  failures; `uv run ty check src/` diagnostic count ≤ `docs/relay/baseline-ty.txt`.
- [ ] 1.2 Assert the module is genuinely pure — no import of a database, HTTP, or model-provider
  symbol.
  **Proof:** `rg -n "sqlalchemy|httpx|langchain|asyncpg|redis" src/app/shared/evaluation/metrics.py; test $? -eq 1`
  → exit `0`.
- [ ] 1.3 Property tests using the existing `hypothesis` dependency and `property` marker: recall@k is
  non-decreasing in k, and every metric lies in `[0, 1]`.
  **Proof:** `uv run pytest tests/property/test_eval_metric_properties.py -q` summary shows zero
  failures.

## 2 Golden-set schema and loader

- [ ] 2.1 `src/app/shared/evaluation/schema.py` — Pydantic v2 `GoldenQuery` (query text, expected chunk
  identifiers, expected document identifiers, jurisdiction and document-kind filters, difficulty,
  notes) and `GoldenSet` carrying a version string.
  **Proof:** `uv run pytest tests/unit/shared/evaluation/test_golden_schema.py -q` summary shows zero
  failures.
- [ ] 2.2 Loader returning a `Result` failure for a malformed row and raising a typed project exception
  for an absent file, per `.opencode/instructions/RESULT-PATTERN.md`.
  **Proof:** one test asserts the failure carries the offending row index; one asserts the typed
  exception type. `uv run pytest tests/unit/shared/evaluation -q` summary shows zero failures.
- [ ] 2.3 Seed corpus `evals/golden/legal_retrieval_v1.jsonl` — deliberately small, covering contracts,
  statutes, judgments, and filings, with each row flagged as awaiting subject-matter expansion.
  **Proof:** `uv run pytest tests/unit/shared/evaluation/test_golden_set_loads.py -q` summary shows
  zero failures, with the test asserting schema validity and coverage of all four families.

## 3 Runner, report, and the liveness probe

- [ ] 3.1 `src/app/shared/evaluation/runner.py` — `run_retrieval_eval(*, queries, retrieve)` where
  `retrieve` is an injected async `Protocol`. The module imports nothing from `features/`.
  **Proof:** a unit test with a fake retriever produces aggregates equal to hand-computed values; and
  `rg -n "^from app\.features|^import app\.features" src/app/shared/evaluation/; test $? -eq 1` → exit
  `0`.
- [ ] 3.2 `report.py` — JSON carrying commit identifier, timestamp, golden-set version, per-query rows,
  and aggregates.
  **Proof:** a round-trip unit test passes, and a second asserts the report names the golden-set
  version it scored.
- [ ] 3.3 Binding entrypoint reaching retrieval through the documents **service**, never the
  repository. Marked `requires_db`, which `addopts` already deselects.
  **Proof:** `uv run pytest -q --collect-only 2>&1 | tail -1` shows the new integration test as
  deselected; `uv run pytest -q` failure count ≤ `docs/relay/baseline-pytest.txt`.
- [ ] 3.4 **Liveness probe — the harness must be proven to reach real retrieval.** Every Proof above
  runs against a fake retriever, so a fully green offline suite is consistent with a harness wired to
  nothing. Seed the database, run the binding entrypoint against the live documents service, and
  assert the report's per-query rows carry chunk identifiers that exist in the chunk store.
  **Proof:** `uv run pytest -m requires_db tests/integration/evaluation -q` summary shows zero
  failures, and the emitted report's retrieved identifiers are a non-empty subset of a chunk-identifier
  snapshot taken inside the same test. An all-zero metric result is an acceptable *score*; an empty
  identifier set is a wiring failure and fails this task.

## 4 Baseline capture

- [ ] 4.1 Produce the first report against the seeded live path from 3.4. "No environment available"
  is not an acceptable outcome — the database is available; only a recorded command failure is.
  **Proof:** `test -s evals/reports/baseline.json; echo $?` → `0`, and the file carries either real
  aggregates or a `reason` string quoting the verbatim failure of the command that could not run.
- [ ] 4.2 Confirm the default gate is unchanged by this change.
  **Proof:** `uv run ruff check --no-cache src/` line count ≤ `docs/relay/baseline-ruff-after.txt`;
  `uv run pytest -q 2>&1 | tail -1` failure count ≤ `docs/relay/baseline-pytest.txt`.

## 5 The judged layer's seam

- [ ] 5.1 Define the report's judged section as an optional, absent-by-default block, and assert that
  retrieval-only mode makes no provider call.
  **Proof:** a unit test runs the harness with a provider double that raises on any call, and the run
  completes; the emitted report has no judged section.
- [ ] 5.2 Record in `review.md` what the later judged layer will add and what it must not change:
  retrieval metrics for an unchanged input stay identical, and the judged metrics are a separate
  report section.
  **Proof:** `review.md` names both constraints.

## 6 Close out

- [ ] 6.1 **Proof:** `openspec validate rag-eval-harness --strict` exits `0`; ruff and ty counts ≤ their
  baselines; pytest failure count ≤ `docs/relay/baseline-pytest.txt`.
