# Plan — PLATFORM group (`rag-tree-repair`, `rag-eval-harness`, `graph-lifecycle`)

Leg 2 of relay, 2026-09-09, `main` @ `7cca750`. Binds to `docs/relay/decisions-rag-cluster.md` (authoritative)
and `docs/relay/scout-rag-cluster.md`.

**Baseline rule for every Proof below:** compare against a captured baseline file, never an absolute
number. `rag-tree-repair` task group 4 creates those files; changes 2 and 3 cite them.

---

## Change 1 — `rag-tree-repair`

**Capability:** `source-tree-integrity` (no collision with the 28 live capabilities, nor with the 10
D22 restores).

**Shape.** Finish D1's seven-item repoint so `import app.main` succeeds, **track the untracked
`docling/` tree**, drop the stale per-file-ignores, add the empty `src/app/features/__init__.py` (D8),
**restore the 10 archived capabilities D22 names into `openspec/specs/`**, repair `alembic/env.py`,
then *measure* — ruff, ty, pytest and `alembic current` — writing each result to a baseline file the
rest of the cluster compares against.

**Rejected shape.** `git checkout -- src/app/shared/rag/document_processing/` and delete the untracked
`docling/` tree: one command, instantly green. Lost because D1 locks the rename as the user's and
intended to survive.

**Why (proposal shape).** The working tree does not import: `src/app/shared/rag/document_processing/`
is deleted-and-staged while its replacement `src/app/shared/rag/docling/` is untracked and still
imports the package it replaced. Nothing in the cluster can be collected, linted honestly, or measured
until this lands. It also clears the 12 pre-existing `INP001` by restoring
`src/app/features/__init__.py` as an empty file — deleted in `8e25352` when model-import/router-import
coupling was severed, so it must stay empty. And per D22 the spec baseline is itself broken: ten
capabilities this cluster's changes must attach deltas to exist only inside archived changes, so a
`## MODIFIED` block in any later change would have no base to modify.

### Requirements

**### Requirement: Importable application tree**
The application package SHALL import without error, no module under `src/` or `tests/` SHALL
reference the retired `app.shared.rag.document_processing` path, and every module the application
imports SHALL be tracked by version control.
- `#### Scenario:` WHEN the application entrypoint module is imported THEN import SHALL complete
  without a module-resolution error.
- `#### Scenario:` WHEN the source and test trees are searched for the retired package path THEN there
  SHALL be zero matches.
- `#### Scenario:` WHEN version control is queried for untracked files under the source tree THEN no
  module reachable from the application entrypoint SHALL be reported.

**### Requirement: Lint configuration tracks the package layout**
Per-file lint exemptions SHALL name only paths that exist on disk.
- `#### Scenario:` WHEN an exemption names a module that the rename did not carry over THEN that
  exemption SHALL be removed rather than repointed.
- `#### Scenario:` WHEN the linter runs THEN no import-outside-top-level or unsorted-import diagnostic
  SHALL be attributable to the rename.

**### Requirement: The features directory is a regular package with no import side effects**
`src/app/features/` SHALL contain an `__init__.py`, and that file SHALL execute no imports.
- `#### Scenario:` WHEN the linter runs over the features tree THEN no implicit-namespace-package
  diagnostic SHALL be reported.
- `#### Scenario:` WHEN the features package is imported THEN no feature router or ORM model module
  SHALL become loaded as a side effect.

**### Requirement: A recorded gate baseline for the cluster**
The repair SHALL record measured lint, type, test and migration-revision outputs to durable files.
- `#### Scenario:` WHEN a later change in this cluster claims a gate improvement THEN it SHALL compare
  against these recorded files rather than a number written in prose.
- `#### Scenario:` WHEN a baseline command cannot execute THEN the recorded file SHALL carry the
  verbatim failure text instead of an assumed value.

**### Requirement: Migration tooling executes against the live database**
The migration environment module SHALL permit the standard migration commands to run without editing.
- `#### Scenario:` WHEN the current-revision command is issued THEN it SHALL report a revision
  identifier that exists on disk, rather than failing during environment import.
- `#### Scenario:` WHEN a later change authors a migration THEN it SHALL derive its parent revision
  from that recorded identifier rather than from a value written in prose.

**### Requirement: Every capability a change amends is present in the specification baseline**
Capabilities whose requirements this cluster amends SHALL exist under the live specification tree
before any change declares a modification against them.
- `#### Scenario:` WHEN a change proposes a modification to a requirement THEN that requirement SHALL
  already be present in the live specification baseline.
- `#### Scenario:` WHEN a capability is restored from an archived change THEN its requirement and
  scenario text SHALL be reproduced without alteration.
- `#### Scenario:` WHEN the specification tree is validated in strict mode THEN every restored
  capability SHALL pass.

### Tasks

**## 1 Confirm the toolchain without mutating it**
- [ ] 1.1 Check lock freshness; do **not** run bare `uv sync` (it prunes `pytest-asyncio`: test deps
  sit outside `default-groups = ["dev"]`, `pyproject.toml:240,252`).
  **Proof:** `uv lock --check; echo $?` → `0`.
- [ ] 1.2 Capture the pre-repair lint baseline.
  **Proof:** `uv run ruff check --no-cache --output-format concise src/ > docs/relay/baseline-ruff-before.txt; wc -l < docs/relay/baseline-ruff-before.txt`
  → `24`, matching the decisions doc.

**## 2 Repoint the rename (D1's seven items, in D1's order)**
- [ ] 2.0 **Track the replacement tree first.** `src/app/shared/rag/docling/` is untracked
  (`git status` shows `?? src/app/shared/rag/docling/`) while `document_processing/` is
  deleted-and-staged. Every later Proof in this cluster runs against a tree that version control
  cannot see, and the anchor leg would commit a repo that does not import. `git add
  src/app/shared/rag/docling/`.
  **Proof:** `git status --porcelain src/app/shared/rag/ | rg '^\?\?'; test $? -eq 1` → exit `0`.
- [ ] 2.1 `src/app/shared/rag/docling/__init__.py:3,9,21,27,37` — the new package imports the old one.
  **Proof:** `uv run python -c "import app.shared.rag.docling"; echo $?` → `0`.
- [ ] 2.2 `src/app/features/documents/classification.py:13,14` and `parser.py:11,12`.
  **Proof:** `uv run python -c "import app.features.documents.parser, app.features.documents.classification"; echo $?` → `0`.
- [ ] 2.3 `src/app/shared/langgraph_layer/ingestion_kb/nodes.py:31` (`table_markdown`) and
  `src/app/examples/policy_examples.py:36,40`.
  **Proof:** `uv run python -c "import app.shared.langgraph_layer.ingestion_kb.nodes, app.examples.policy_examples"; echo $?` → `0`.
- [ ] 2.4 Whole-app import.
  **Proof:** `uv run python -c "import app.main"; echo $?` → `0`.
- [ ] 2.5 Tests: `tests/unit/test_auth_documents_feature_errors.py:19`,
  `tests/unit/shared/rag/test_chunker_tokenizer_cache.py:33,34,40`,
  `tests/unit/shared/rag/test_embedder_no_substitution.py:26,27,34`,
  `tests/unit/shared/rag/test_rag_agent_embedder_import.py:38` (a **string** target, not an import — a
  symbol-import grep cannot see it).
  **Proof:** `uv run pytest -q --collect-only > /dev/null; echo $?` → `0` (a string grep alone is an
  insufficient probe here — project memory, "Proof-mechanism blind spots").
- [ ] 2.6 `pyproject.toml:536-553` — repoint five entries to `src/app/shared/rag/docling/*`, and
  **delete** the `document_processing/ingest.py` entry: the new tree has seven files
  (`__init__, chunker, docling_enhanced, embedder, entity_extractor, ingest_v2, models`) and
  `ingest.py` is not among them.
  **Proof:** `rg -n "document_processing" pyproject.toml; test $? -eq 1` → exit `0`.
- [ ] 2.7 Non-load-bearing refs: `src/app/utils/embedding.py:23`,
  `src/alembic/versions/0013_*.py:101`, `src/app/shared/rag/langextract/langextract_to_graph.py:1`.
  **Proof:** `rg -n "rag\.document_processing|rag/document_processing" src/ tests/ pyproject.toml; test $? -eq 1`
  → exit `0`.

**## 3 The features package (D8)**
- [ ] 3.1 Create `src/app/features/__init__.py` **empty**.
  **Proof:** `test ! -s src/app/features/__init__.py; echo $?` → `0`.
- [ ] 3.2 Confirm eager coupling did not return.
  **Proof:** `uv run python -c "import app.features, sys; assert not [m for m in sys.modules if m.startswith('app.features.') and m.endswith('.router')]"; echo $?`
  → `0`. If red, fall back to a scoped `INP001` per-file-ignore (D8's stated fallback) and record why.

**## 4 Measure and record the cluster baseline**
- [ ] 4.1 Lint after.
  **Proof:** `uv run ruff check --no-cache --output-format concise src/ > docs/relay/baseline-ruff-after.txt; rg -o "[A-Z]+[0-9]+" docs/relay/baseline-ruff-after.txt | sort -u`
  → `PLC2701` only (owned by `ingestion-chunking`), and the line count is strictly less than
  `baseline-ruff-before.txt`.
- [ ] 4.2 **First-ever pytest measurement.** Budget for it: the scout's single import probe exceeded
  120 s (Fog #2); `addopts` already carries `--timeout=60` and deselects `integration`/`requires_db`.
  **Proof:** `uv run pytest -q > docs/relay/baseline-pytest.txt 2>&1; tail -1 docs/relay/baseline-pytest.txt`
  → a summary line; that file is the baseline every later change cites.
- [ ] 4.3 Type baseline (measure, do not trust prior counts — project memory records ty reported 2 when
  it was 46).
  **Proof:** `uv run ty check src/ > docs/relay/baseline-ty.txt 2>&1; tail -1 docs/relay/baseline-ty.txt`.
- [ ] 4.4 **O7, first attempt.** Run `uv run alembic current`. It is expected to fail —
  `alembic/env.py` is known to break migration commands, and it needs the now-repaired tree to import
  at all.
  **Proof:** `uv run alembic current > docs/relay/baseline-alembic.txt 2>&1 || true; cat docs/relay/baseline-alembic.txt`
  → the file holds either a revision id or the verbatim failure.
- [ ] 4.5 **Repair `alembic/env.py` until 4.4 succeeds.** Scope is strictly the environment module —
  no revision may be authored, edited, or applied here.
  **Proof:** `uv run alembic current > docs/relay/baseline-alembic.txt 2>&1; echo $?` → `0`, and the
  recorded value matches a `revision` literal present under `src/alembic/versions/`. Reconcile the
  three disagreeing sources in the change's `review.md`: disk carries `0001`–`0017`, the handover
  claims `b3e7c41d92af` (on no revision on disk), project memory says the DB is stamped at `0004`.

**## 5 Restore the specification baseline (D22)**
- [ ] 5.1 Copy the ten archived capability specs into `openspec/specs/<capability>/spec.md`, verbatim
  apart from replacing the `## ADDED Requirements` header with `## Requirements`: seven from
  `2026-09-07-ingestion-pipeline-unification` (`celery-worker-deployment`,
  `document-ingestion-pipeline`, `graph-entity-canonicalisation`, `hierarchical-document-chunking`,
  `hybrid-retrieval-ranking`, `langgraph-checkpointing`, `unified-embedding`), one from
  `2026-09-07-documents-unified-schema` (`document-retrieval-schema`), two from
  `2026-09-07-agent-tools-unification` (`legal-corpus-retrieval`, `agent-tool-registry`).
  **Proof:** requirement and scenario counts per file match the archived source exactly —
  `8/19, 10/27, 4/12, 7/16, 4/12, 8/24, 10/24, 10/27, 4/11, 5/15` — and
  `openspec validate --specs` exits `0`. Format authority is
  `.opencode/skills/openspec-sync-specs/SKILL.md`'s "Main Spec Format Reference": a main spec carries
  `# <capability> Specification`, `## Purpose` (the delta's Purpose body verbatim), then a single
  `## Requirements`, and **never** a `## ADDED/MODIFIED/REMOVED/RENAMED` header.
- [ ] 5.2 Record, in the change's `review.md`, the two capabilities deliberately **not** restored and
  why: `graphiti-init-order` and `embedding-dimension-config` predate the requirement grammar
  (`## Scope`/`## Problem`/`## Solution`/`## Verification`, zero `### Requirement:` blocks), so there
  is nothing to restore.
  **Proof:** `rg -c '^### Requirement:' openspec/changes/archive/2026-06-22-quality-fixes-batch-2/specs/graphiti-init-order/spec.md; test $? -eq 1` → exit `0`.
- [ ] 5.3 Record the four measurements behind D22 so the restore is auditable rather than asserted.
  **Proof:** `review.md` carries the `typed-exception-handling` comparison (four delta requirements,
  none among the thirteen live), the six archive tick counts, the `embedder.py:44-48` B1 IOU quoted
  verbatim, and the `AgentToolBundle` "empty tool lists" docstring.

**O7 ruling (revised — supersedes the measurement-only reading).** Repairing `alembic/env.py` is
**in scope here**. D19 makes the live database fully available, which turns O7 from an unanswerable
question into a command that must be made to run; and `ingestion-chunking`'s D12 migration is blocked
on a real `down_revision`, which cannot be guessed. Authoring migrations stays out of scope — this
change fixes the tooling and records the head, nothing more.

**Dependencies / seams.** Nothing precedes it. Must not touch: the 3 `PLC2701` `_build_chat_model`
imports, embedder unification, any chunking or retrieval SQL — all owned by other groups. It restores
spec files but writes **no delta** against the ten restored capabilities; the changes that amend them
own that.

---

## Change 2 — `rag-eval-harness`

**Capability:** `retrieval-evaluation`.

**Shape.** A pure metric core (recall@k, MRR, nDCG@k, precision@k) over ranked id lists, a
Pydantic-validated golden-set file, and a runner that takes an *injected* async retriever. Fast,
deterministic pieces run in the default suite; anything needing a database or a provider carries the
existing `requires_db` marker, which `addopts` already deselects — so no new gate machinery is
invented.

**Rejected shape.** Adopt `ragas` and drive it from pytest as the primary gate. Lost because ragas
routes even retrieval metrics through an LLM, making the fast gate network-bound and
non-deterministic, and it adds a second eval stack against D14's dependency-weight posture; the four
rank metrics needed here are pure arithmetic.

> **D18 settles this.** The user ruled "both — pure core now, RAGAS layer later". So the rejection
> above is scoped to *tier 1 only*: ragas is not rejected, it is deferred to a later change and this
> change must leave the seam for it. That is why the "Answer-quality judging is separable from
> retrieval scoring" requirement exists — it is the seam, specified now so the later layer is an
> addition rather than a rewrite.

**Why (proposal shape).** Every other recommendation in this cluster is a prior, not a measurement
(D13). Uber's real unlock in the EAg-RAG work was cutting evaluation from weeks to minutes, so they
could tell which priors were wrong on their corpus. Building this after the retrieval work means
`ingestion-chunking` and `retrieval-sql` ship unmeasured. It runs immediately after `rag-tree-repair`
because nothing collects until the tree imports.

### Requirements

**### Requirement: The golden set is a versioned, validated artifact**
Golden-set rows SHALL be schema-validated on load, and a malformed row SHALL be reported as an expected
failure rather than an exception.
- `#### Scenario:` WHEN a row omits a required field THEN the loader SHALL return a failure identifying
  the offending row index.
- `#### Scenario:` WHEN the golden-set file is absent or unreadable THEN the loader SHALL raise a typed
  exception from the project hierarchy.
- `#### Scenario:` WHEN the golden set is loaded THEN it SHALL contain at least one query for each of
  the four legal document families (D10).

**### Requirement: Deterministic retrieval metrics**
Retrieval metrics SHALL be computed from ranked identifier lists alone, with no model or database
access.
- `#### Scenario:` WHEN an expected chunk appears at rank one THEN recall-at-k and reciprocal rank
  SHALL both be one for that query.
- `#### Scenario:` WHEN no expected chunk appears within the top k THEN recall-at-k and reciprocal rank
  SHALL both be zero.
- `#### Scenario:` WHEN the same inputs are scored twice THEN identical values SHALL be produced.

**### Requirement: The harness never blocks the default test gate**
- `#### Scenario:` WHEN the default test selection runs THEN no scenario requiring a live database or a
  model provider SHALL execute.
- `#### Scenario:` WHEN the harness is invoked explicitly THEN it SHALL emit a machine-readable report
  carrying per-query rows, aggregates, and the commit identifier of the tree it measured.

**### Requirement: A recorded retrieval baseline exists before retrieval changes**
- `#### Scenario:` WHEN a retrieval-affecting change reports an improvement THEN it SHALL cite a report
  produced by this harness against the recorded baseline report.
- `#### Scenario:` WHEN no environment is available to produce a first baseline THEN the baseline
  artifact SHALL record that fact and its reason explicitly.

**### Requirement: Answer-quality judging is separable from retrieval scoring**
- `#### Scenario:` WHEN the harness runs in retrieval-only mode THEN no model-provider call SHALL be
  made.

### Tasks

**## 1 Metric core (pure, no I/O)**
- [ ] 1.1 `src/app/shared/evaluation/metrics.py` — `recall_at_k`, `reciprocal_rank`, `ndcg_at_k`,
  `precision_at_k` over `Sequence[str]` + expected-id sets. Fully annotated, no `# type: ignore`.
  **Proof:** `uv run pytest tests/unit/shared/evaluation/test_metrics.py -q` passes;
  `uv run ty check src/` diagnostic count ≤ `docs/relay/baseline-ty.txt`.
- [ ] 1.2 Property tests (repo already has `hypothesis` and a `property` marker): recall@k
  non-decreasing in k; every metric in `[0, 1]`.
  **Proof:** `uv run pytest tests/property/test_eval_metric_properties.py -q` passes.

**## 2 Golden-set schema and loader**
- [ ] 2.1 `schema.py` — Pydantic v2 `GoldenQuery` (query text, expected chunk ids, expected document
  ids, jurisdiction / document-kind filters, difficulty, notes) and `GoldenSet`.
  **Proof:** `uv run pytest tests/unit/shared/evaluation/test_golden_schema.py -q` passes.
- [ ] 2.2 Loader returning `Result` for malformed rows and raising for an absent file, per
  `.opencode/instructions/RESULT-PATTERN.md`.
  **Proof:** one test asserts `Failure` carrying the row index; one asserts the typed exception.
  `uv run pytest tests/unit/shared/evaluation -q` passes.
- [ ] 2.3 Seed corpus `evals/golden/legal_retrieval_v1.jsonl` — a deliberately small stub covering
  contracts, statutes, case law, filings, flagged as awaiting SME expansion (D13's accepted cost).
  **Proof:** `uv run pytest tests/unit/shared/evaluation/test_golden_set_loads.py -q` passes, asserting
  schema validity and four-family coverage.

**## 3 Runner and report**
- [ ] 3.1 `runner.py` — `run_retrieval_eval(*, queries, retrieve)` where `retrieve` is an injected async
  `Protocol`; the module imports nothing from `features/`.
  **Proof:** unit test with a fake retriever produces exact hand-computed aggregates;
  `rg -n "^from app\.features" src/app/shared/evaluation/; test $? -eq 1`.
- [ ] 3.2 Report writer: JSON with commit sha, timestamp, per-query rows, aggregates.
  **Proof:** round-trip unit test passes.
- [ ] 3.3 Binding entrypoint that reaches retrieval through the documents **service** (never the
  repository — layering), marked `requires_db`.
  **Proof:** `uv run pytest -q --collect-only 2>&1 | tail -1` shows the new integration test as
  deselected, and `uv run pytest -q` failure count ≤ `docs/relay/baseline-pytest.txt`.
- [ ] 3.4 **Liveness probe — the harness must be proven to reach real retrieval.** Every proof above
  runs against a fake retriever, so a fully green tier-1 suite is consistent with a harness that is
  wired to nothing. Seed the scratch database (D19 authorises this outright), run the binding
  entrypoint against the live documents service, and assert the report's per-query rows carry chunk
  ids that exist in `chunks`.
  **Proof:** `uv run pytest -m requires_db tests/integration/evaluation -q` passes, and the emitted
  report's `retrieved_ids` are a non-empty subset of a `SELECT id FROM chunks` snapshot taken in the
  same test. A report of all-zero metrics is an acceptable *result*; an empty `retrieved_ids` is a
  wiring failure and fails the task.

**## 4 Baseline capture**
- [ ] 4.1 Produce the first report. **D19 removes the old escape hatch** — the database is fully
  available, so "unmeasured, no environment" is no longer an acceptable outcome; only a recorded
  failure with its verbatim reason is.
  **Proof:** `test -s evals/reports/baseline.json; echo $?` → `0`, and the file carries real
  aggregates produced by task 3.4's live path, or a `reason` string quoting the command that failed.
- [ ] 4.2 Default gate unchanged.
  **Proof:** `uv run ruff check --no-cache src/` count ≤ `docs/relay/baseline-ruff-after.txt`;
  `uv run pytest -q` failure count ≤ `docs/relay/baseline-pytest.txt`.

**Dependencies / seams.** After `rag-tree-repair` only. Must not touch retrieval SQL, fusion, chunking,
the embedder, or the reranker — the harness only *calls* the existing service search seam.
`agentic-retrieval` depends on this change.

---

## Change 3 — `graph-lifecycle`

**Capability:** `compiled-graph-lifecycle`, plus a delta against the D22-restored
**`langgraph-checkpointing`** — the checkpointer's ownership and teardown semantics are already
specified there ("The constructing process owns the checkpointer pool, and teardown distinguishes
nothing-to-close from a close"), so this change **cites** that requirement and does not restate it.
Only genuinely new behaviour — process-scoped compilation, the `configurable` seam, Saul provisioning
— goes into the new capability.

**Shape.** Make `build_document_ingestion_graph` repository-free (the repository arrives per invocation
via `config["configurable"]`, per D7), add `src/app/lifecycle/graphs.py` holding pure provider
functions, and register each graph as a `StartupPolicy` entry — reusing the degrade machinery already
at `lifespan.py:171-342` rather than restoring hand-rolled try/except blocks. The Celery worker gets
the same providers through `worker_process_init`, and releases them through `worker_process_shutdown`.

**Forced ordering — checkpointer → Saul → `configurable` seam → Celery → KB graph → globals.** This is
not a preference. `build_saul_graph`'s `checkpointer` parameter is **not optional**, so Saul cannot be
provisioned before the checkpointer policy exists; the Celery wiring cannot pass a repository until the
`configurable` seam accepts one; and the import-time global compiles are last because they are the only
step with no consumer waiting on them. The task groups below follow that order, which is a deliberate
change from the earlier draft's "seam first" arrangement.

**Rejected shape.** Uncomment the blocks at `lifespan.py:522-537` and `:549-565` in place. Fewest lines,
but it re-adds exactly the hand-rolled degrade blocks `STARTUP_POLICIES` was built to remove (its own
comment, `lifespan.py:281-283`), and it offers no seam the Celery worker can share — so the per-job
recompile, which is the actual content of todo 235, would survive untouched.

**Why (proposal shape).** Three graphs are in the wrong place. `build_document_ingestion_graph` is
recompiled inside every Celery job (`service.py:929`). `app.state.ingestion_graph`, `pageindex_client`,
and the checkpointer block are commented out. `app.state.saul_graph` is *never assigned*, so
`features/agent_saul/dependencies.py:37-45` raises service-unavailable on every request and
`agent_saul` 503s today (D9). Compilation moves to one place per process and consumers read the
compiled graph from state.

**Conflict this change must resolve in its Why, with evidence.** Two prose sites currently forbid this
wiring, citing an *earlier* change's D17 — not this cluster's. **D20 rules that both are superseded and
must be rewritten**, not left standing:
- `src/app/shared/langgraph_layer/checkpointer.py:11-17` — "The lifespan wiring stays commented, by
  decision (D17)."
- `tests/unit/features/ingestion/test_unprovisioned_graph_fails_closed.py:3-8` — records it as a
  **non-goal**: "Nothing here provisions the shared graph, and nothing should … ingestion runs in the
  queue worker process, which never executes the application lifespan."

The reconciliation is per *process*, not per file: the ingestion graph's consumer is the Celery worker
(which never runs the lifespan, hence the `worker_process_init` hook), while the Saul graph and
checkpointer serve the API process (which does). The behavioural tests at that path keep passing
untouched — they hand-build a bare `FastAPI` app and never invoke the real lifespan. That is the seam
that makes this safe.

**Landmine.** `tests/unit/shared/langgraph_layer/test_checkpointer_lifecycle.py` includes proofs that
grep `checkpointer.py` for *absent* literals (the saver's connection-string classmethod name; the old
`hasattr(..., "pool")` guard). Any docstring edit there must not introduce those strings.

### Requirements

**### Requirement: Compiled graphs are process-scoped, not per-invocation**
- `#### Scenario:` WHEN two document-ingestion jobs execute in the same worker process THEN the graph
  SHALL be compiled at most once.
- `#### Scenario:` WHEN the API process completes startup THEN each provisioned graph SHALL have been
  compiled exactly once.
- `#### Scenario:` WHEN a graph module is imported THEN no graph SHALL be compiled as a side effect of
  that import.

**### Requirement: Job-scoped collaborators are supplied per invocation**
- `#### Scenario:` WHEN a document-ingestion run is invoked THEN the job-scoped repository SHALL be
  supplied through the invocation configuration.
- `#### Scenario:` WHEN a graph is compiled THEN it SHALL NOT capture a database session or repository,
  and graph state SHALL carry neither.
- `#### Scenario:` WHEN an invocation omits the required job-scoped collaborator THEN a typed exception
  naming the missing collaborator SHALL be raised, not a key or attribute error.

**### Requirement: Graph provisioning degrades rather than failing startup**
- `#### Scenario:` WHEN a graph fails to build during startup THEN startup SHALL continue and that
  capability's state attribute SHALL be absent or none.
- `#### Scenario:` WHEN a dependency reads an unprovisioned graph THEN it SHALL raise a typed
  service-unavailable naming the missing capability rather than producing a server error.

**### Requirement: Agent Saul is provisioned in the serving process**
- `#### Scenario:` WHEN the API process has started successfully THEN the Saul graph and the LangGraph
  checkpointer SHALL both be readable from application state.
- `#### Scenario:` WHEN a request reaches the Agent Saul dependency bundle on a healthy process THEN it
  SHALL NOT report the capability as unavailable.

**### Requirement: Checkpointer ownership is per process and released at shutdown**
- `#### Scenario:` WHEN the serving process shuts down after provisioning a checkpointer THEN its
  connection pool SHALL be closed and the outcome reported.
- `#### Scenario:` WHEN a checkpointer setup fails THEN no credential SHALL appear in any emitted log
  line.

### Tasks

Ordered by the forcing constraint above, not by file.

**## 1 Checkpointer provisioning (first — everything downstream needs it)**
- [ ] 1.1 `src/app/lifecycle/graphs.py` — provider functions only, no I/O at build time (`db_engine` is
  a handle assigned at `lifespan.py:464`, before the commented block at `:522`).
  **Proof:** `uv run python -c "import app.lifecycle.graphs"; echo $?` → `0`; `uv run ty check src/`
  count ≤ `baseline-ty.txt`.
- [ ] 1.2 Checkpointer policy calling `setup_langgraph_checkpointer(get_database_url(flavour="plain"))`;
  delete the commented block at `:549-565`. Teardown already exists behind the `hasattr` guard at `:365`.
  **Proof:** `uv run pytest tests/unit/shared/langgraph_layer/test_checkpointer_lifecycle.py -q` passes
  in full, **and** the forbidden-literal greps still find nothing
  (`rg -n 'from_conn_string|hasattr\(.*"pool"' src/app/shared/langgraph_layer/checkpointer.py; test $? -eq 1`).
  Cites restored `langgraph-checkpointing`; adds no delta to it.

**## 2 Agent Saul provisioning (blocked by 1.2 — `checkpointer` is a required parameter)**
- [ ] 2.1 Construct `AgentMemoryService` in the same policy. **D21 makes this cheap:** its
  `__init__` (`agent_memory_service.py:136`) takes only `partition_prefix` plus optional callables
  defaulting to `cognee.remember` / `.recall` / `.improve` — no client, no engine, no session. Do not
  invent a third construction site; the only existing one is `src/tasks/agent_memory_tasks.py:119`.
  **Proof:** a unit test constructs the policy's provider with no arguments beyond settings and
  asserts the returned service's partition prefix.
- [ ] 2.2 Saul graph policy assigning `app.state.saul_graph`.
  **Correction to the earlier draft:** `src/app/shared/rag/graphiti/registry.py:11-34` is **not a
  recipe to follow** — it is a module docstring, and a stale one. It names `build_tool_registry`
  (the real symbol is `build_tool_bundle`, `:92`) and `app.state.saul_checkpointer` (the actual
  reader, `agent_saul/dependencies.py:49`, uses `app.state.langgraph_checkpointer`). Wire against the
  reader, not the docstring, and fix the docstring in passing.
  **Proof:** a unit test asserts `get_saul_graph` returns the graph when state is populated and still
  raises the typed 503 when it is not; `rg -n 'build_tool_registry|saul_checkpointer' src/; test $? -eq 1`.
- [ ] 2.3 **Memory semantics are out of scope (D21).** Saul memory is cognee's, already specified by
  the live `saul-memory-prefetch-and-retrieval`. This change constructs and passes the service and
  specifies nothing about what it does.
  **Proof:** the change's spec delta contains no requirement mentioning recall, remember, or memory
  partitioning.

**## 3 The `configurable` seam (D7)**
- [ ] 3.1 Resolve the repository from the invocation config inside the node instead of from a closure
  parameter; add a typed accessor that raises a project exception when the key is absent.
  **Proof:** `uv run pytest tests/unit/documents tests/unit/shared/langgraph_layer/test_ingestion_checkpoint_plumbing.py -q`
  passes, plus a new test asserting the typed exception (not `KeyError`) on a missing key.

**## 4 Celery worker process wiring (blocked by 3.1 — nothing to pass until the seam exists)**
- [ ] 4.1 Point `run_document_ingestion_task` (`src/app/features/documents/service.py:904-947`) at a
  process-cached compiled graph and pass the repository through the config.
  **Proof:** new unit test patches the builder with a counting spy and asserts one compile across two
  invocations.
- [ ] 4.2 **Highest-risk task in this change.** Hoist the per-task construction at
  `service.py:912-935` into `worker_process_init`, and move its `close_graphiti()` and
  `engine.dispose()` teardown into `worker_process_shutdown`. Today those are built and torn down
  inside every job; hoisting them changes the lifetime of a database engine and a Graphiti client
  from per-task to per-process. Get the shutdown hook wrong and connections leak for the worker's
  whole life; get the init hook wrong and every task in the process shares a half-built client.
  **Proof:** a unit test drives `worker_process_init` then `worker_process_shutdown` against spies and
  asserts one construction, one `close_graphiti`, one `engine.dispose()`; and a second test asserts
  two simulated task invocations between them construct nothing.
- [ ] 4.3 Worker-process wiring registered (prior art: `src/tasks/crawler_tasks.py:72`).
  **Proof:** `rg -n "worker_process_init|worker_process_shutdown" src/tasks/` shows both ingestion
  hooks; a unit test asserts the init hook populates the process cache.

**## 5 The ingestion (KB) graph as a startup policy**
- [ ] 5.1 Register the ingestion graph as a `StartupPolicy`; delete the commented block at `:522-537`
  while **preserving** its `embedding_fn=` prohibition as a comment on the provider.
  **Proof:** `rg -n "embedding_fn" src/app/` still returns the note;
  `uv run pytest tests/unit/features/ingestion -q` passes unchanged (all seven fail-closed tests).
- [ ] 5.2 Update the two prose sites that forbid this wiring (`checkpointer.py:11-17`,
  `test_unprovisioned_graph_fails_closed.py:3-8`) to cite D20 as the superseding decision — without
  introducing the forbidden literals from 1.2.
  **Proof:** `rg -n "stays commented" src/ tests/; test $? -eq 1`; full-suite failure count ≤
  `baseline-pytest.txt`.

**## 6 Remove import-time compilation (last — no consumer is waiting on it)**
- [ ] 6.1 `src/app/shared/langgraph_layer/open_deep_search/graph.py:278,478,555` — convert the three
  module-global compiles to lazily built, process-cached providers.
  **Proof:** a unit test imports the module with a spy on the compile call and asserts zero invocations
  at import time.
- [ ] 6.2 `app.state.pageindex_client` (`lifespan.py:538`) stays unwired — D15 drops the PageIndex
  dependency and `knowledge-stack` owns the disposition.
  **Proof:** `rg -n "pageindex_client" src/app/lifecycle/lifespan.py` → the single commented line,
  unchanged.

**## 7 Re-verify against the captured baselines**
- [ ] 7.1 **Proof:** `uv run ruff check --no-cache src/` count ≤ `docs/relay/baseline-ruff-after.txt`;
  `uv run ty check src/` count ≤ `baseline-ty.txt`; `uv run pytest -q` failure count ≤
  `baseline-pytest.txt`.

**Dependencies / seams.** After `rag-tree-repair`. Must not touch: chunking, the embedder, retrieval
SQL, fusion, the reranker, or `_build_chat_model` (all other groups). It may *read* `app.state.graphiti`
and `app.state.db_engine`, which lifespan already assigns.

---

## Blast radius to re-verify (cross-change)

- `tests/unit/features/ingestion/test_unprovisioned_graph_fails_closed.py` — 7 tests; prose contradicts
  change 3, behaviour unaffected.
- `tests/unit/shared/langgraph_layer/test_checkpointer_lifecycle.py` — forbidden-literal greps and
  credential-scrub assertions.
- `tests/unit/shared/langgraph_layer/test_ingestion_checkpoint_plumbing.py` — guards D7's rejected
  repository-in-state alternative.
- `tests/unit/test_state_hydration.py`, `test_saul_step_budget.py`, `test_saul_role_output_schemas.py`,
  `test_agent_saul_persist_memory.py` — Saul graph surface.
- `tests/unit/documents/test_hybrid_search_failure.py` and `src/app/examples/policy_examples.py` —
  callers of `build_document_ingestion_graph`.
- `src/tasks/document_tasks.py` — two callers of `run_document_ingestion_task`.
- `tests/unit/shared/rag/test_chunker_tokenizer_cache.py`, `test_embedder_no_substitution.py`,
  `test_rag_agent_embedder_import.py`, `tests/unit/test_auth_documents_feature_errors.py` — rename
  importers (change 1).

## Risks

| Risk | Earliest step it shows |
|---|---|
| **Hoisting `service.py:912-935` changes an engine's and a Graphiti client's lifetime from per-task to per-process; a wrong shutdown hook leaks connections for the worker's whole life** | graph-lifecycle 4.2 — the highest-risk task in the cluster |
| The prior D17 prose forbids exactly what change 3 does; reviewers may read it as authoritative | graph-lifecycle 1.2 / 5.2 (D20 settles it) |
| `registry.py:11-34` is a stale docstring, not a recipe — following it verbatim wires a function that does not exist | graph-lifecycle 2.2 |
| `build_saul_graph`'s `checkpointer` is non-optional, so a "seam first" ordering deadlocks | graph-lifecycle group order (why 1 precedes 2 and 3) |
| The ten restored specs may not validate strictly once outside their archive | tree-repair 5.1 |
| pytest cost: the scout's single import probe exceeded 120 s | tree-repair 4.2 |
| `alembic/env.py` repair may be deeper than an import fix, and D12's migration is blocked on it | tree-repair 4.5 |
| The eval harness can be fully green against a fake retriever while wired to nothing | eval 3.4 (the liveness probe exists for exactly this) |
| Restoring `features/__init__.py` reintroduces import-time coupling | tree-repair 3.2 (fallback: scoped ignore, per D8) |

---

## Shapes (the compact return)

**1. `rag-tree-repair` → capability `source-tree-integrity`.** Track the untracked `docling/` tree,
finish D1's repoint so the tree imports, drop stale per-file-ignores, add the empty
`features/__init__.py`, restore D22's ten capabilities into `openspec/specs/`, repair `alembic/env.py`,
then measure ruff/ty/pytest/alembic into baseline files.
Task groups: `## 1 Confirm the toolchain without mutating it` · `## 2 Repoint the rename` ·
`## 3 The features package` · `## 4 Measure and record the cluster baseline` ·
`## 5 Restore the specification baseline`.
O7 ruling (revised): **repair in scope.** D19 makes the database fully available and D12's migration
needs a real `down_revision`, so `alembic current` must be made to run, not merely attempted.
Authoring migrations stays out of scope.

**2. `rag-eval-harness` → capability `retrieval-evaluation`.** Pure tier-1 metric core + validated
golden set + injected-retriever runner + a liveness probe, so every later retrieval change ships with
a before/after number instead of an assertion. D18's RAGAS tier is deferred but its seam is specified.
Task groups: `## 1 Metric core (pure, no I/O)` · `## 2 Golden-set schema and loader` ·
`## 3 Runner and report` · `## 4 Baseline capture`.

**3. `graph-lifecycle` → capability `compiled-graph-lifecycle`** (+ a citation of restored
`langgraph-checkpointing`, not a restatement). Checkpointer policy, Saul provisioning with cognee-owned
memory passed through, repository-free ingestion graph via `configurable`, Celery `worker_process_init`
/ `worker_process_shutdown`, KB graph as a `StartupPolicy`, then the import-time globals.
Task groups, in forced order: `## 1 Checkpointer provisioning` · `## 2 Agent Saul provisioning` ·
`## 3 The configurable seam` · `## 4 Celery worker process wiring` ·
`## 5 The ingestion (KB) graph as a startup policy` · `## 6 Remove import-time compilation` ·
`## 7 Re-verify against the captured baselines`.

Three findings the orchestrator carries into the proposals: D20 supersedes the prose in
`src/app/shared/langgraph_layer/checkpointer.py:11-17` and
`tests/unit/features/ingestion/test_unprovisioned_graph_fails_closed.py:3-8`, which must be rewritten
rather than worked around; `src/app/shared/rag/graphiti/registry.py:11-34` is a **stale docstring**
naming `build_tool_registry` and `app.state.saul_checkpointer`, neither of which exists, so wire
against `agent_saul/dependencies.py:49` instead; and `build_saul_graph`'s non-optional `checkpointer`
is what forces group 1 before group 2.
