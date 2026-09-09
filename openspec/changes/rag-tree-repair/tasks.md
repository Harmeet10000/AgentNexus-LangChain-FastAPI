# Tasks — rag-tree-repair

## How to read the Proofs

Every task carries a **Proof**: a command whose output settles whether the task is done. Five rules
govern them across this whole cluster, four inherited from the archived
`2026-09-07-ingestion-pipeline-unification` tasks and two of those superseded here. They are recorded
at the top of every change in this cluster so a reader who opens one change in isolation still has
them.

1. **Never use a test-process exit code as a Proof of test outcome.** The configured coverage floor
   is far above actual coverage, so a fully green suite still exits non-zero. Compare the **summary
   pass and failure counts** against the recorded baseline, and re-measure the baseline immediately
   before the task rather than trusting a number from an earlier session.
2. **Never prove a schema fact by rendering migrations offline.** `alembic upgrade head --sql`
   renders from base every time, so it cannot show what one revision adds. Schema facts are proven
   against the live catalogue.
3. **Never make a Proof depend on a durable outbound event firing.** The outbox tables do not exist.
4. **Superseded by D19.** The archived rule said database-touching proofs must bring up a local
   scratch Postgres and never touch the managed instance. The user has since ruled that the live
   database holds zero data and serves zero users and may be used freely. Database Proofs in this
   cluster may target it. This does not extend to any other system.
5. **Superseded by D20.** The archived rule forbade writing an upload-to-chunks acceptance check,
   because the shared graph wiring was to stay commented by decision. `graph-lifecycle` reverses that
   decision, so the check becomes writable — but it belongs to that change, not this one.

Baseline files this change produces live in `docs/relay/` and are cited by name by every later change.

## 1 Confirm the toolchain without mutating it

- [ ] 1.1 Check lock freshness. Do **not** run a bare `uv sync`: the test dependencies sit outside
  `default-groups = ["dev"]` (`pyproject.toml:240,252`), so a bare sync uninstalls `pytest-asyncio`
  and silently breaks collection.
  **Proof:** `uv lock --check; echo $?` → `0`.
- [ ] 1.2 Capture the pre-repair lint baseline, before any file is touched.
  **Proof:** `uv run ruff check --no-cache --output-format concise src/ > docs/relay/baseline-ruff-before.txt; wc -l < docs/relay/baseline-ruff-before.txt`
  → a recorded count. Do not assert a specific number here; the point of the file is that the number
  is measured rather than remembered.

## 2 Repoint the rename

- [ ] 2.0 **Track the replacement tree first.** `src/app/shared/rag/docling/` is untracked while
  `src/app/shared/rag/document_processing/` is deleted-and-staged. Until this lands, every Proof below
  runs against files version control cannot see, and the commit at the end of this cluster would land
  a repository that does not import.
  **Proof:** `git status --porcelain src/app/shared/rag/ | rg '^\?\?'; test $? -eq 1` → exit `0`.
- [ ] 2.1 `src/app/shared/rag/docling/__init__.py:3,9,21,27,37` — the replacement package imports the
  package it replaces.
  **Proof:** `uv run python -c "import app.shared.rag.docling"; echo $?` → `0`.
- [ ] 2.2 `src/app/features/documents/classification.py:13,14` and
  `src/app/features/documents/parser.py:11,12`.
  **Proof:** `uv run python -c "import app.features.documents.parser, app.features.documents.classification"; echo $?` → `0`.
- [ ] 2.3 `src/app/shared/langgraph_layer/ingestion_kb/nodes.py:31` (the `table_markdown` import) and
  `src/app/examples/policy_examples.py:36,40`.
  **Proof:** `uv run python -c "import app.shared.langgraph_layer.ingestion_kb.nodes, app.examples.policy_examples"; echo $?` → `0`.
- [ ] 2.4 Whole-application import.
  **Proof:** `uv run python -c "import app.main"; echo $?` → `0`.
- [ ] 2.5 Tests: `tests/unit/test_auth_documents_feature_errors.py:19`,
  `tests/unit/shared/rag/test_chunker_tokenizer_cache.py:33,34,40`,
  `tests/unit/shared/rag/test_embedder_no_substitution.py:26,27,34`, and
  `tests/unit/shared/rag/test_rag_agent_embedder_import.py:38` — the last is a **patch-target string**,
  not an import, so no symbol-import probe can see it.
  **Proof:** `uv run pytest -q --collect-only > /dev/null; echo $?` → `0`. A string grep alone is an
  insufficient probe here.
- [ ] 2.6 `pyproject.toml:536-553` — repoint five per-file-ignore entries to
  `src/app/shared/rag/docling/*`, and **delete** the `document_processing/ingest.py` entry. The
  replacement package has seven modules (`__init__`, `chunker`, `docling_enhanced`, `embedder`,
  `entity_extractor`, `ingest_v2`, `models`) and `ingest.py` is not among them, so that exemption
  names a file that will never exist.
  **Proof:** `rg -n "document_processing" pyproject.toml; test $? -eq 1` → exit `0`.
- [ ] 2.7 Non-load-bearing references: `src/app/utils/embedding.py:23`,
  `src/alembic/versions/0013_*.py:101` (a comment), and
  `src/app/shared/rag/langextract/langextract_to_graph.py:1` (a docstring).
  **Proof:** `rg -n "rag\.document_processing|rag/document_processing" src/ tests/ pyproject.toml; test $? -eq 1`
  → exit `0`.

## 3 The features package

- [ ] 3.1 Create `src/app/features/__init__.py` **empty**. It was deleted in `8e25352` to sever
  model-import and router-import coupling; restoring content would restore the coupling.
  **Proof:** `test ! -s src/app/features/__init__.py; echo $?` → `0`.
- [ ] 3.2 Confirm the eager coupling did not return with the file.
  **Proof:** `uv run python -c "import app.features, sys; assert not [m for m in sys.modules if m.startswith('app.features.') and m.endswith('.router')]"; echo $?`
  → `0`. If this is red, fall back to a scoped `INP001` per-file-ignore and record in `review.md` why
  the empty-package route was not available.

## 4 Measure and record the cluster baseline

- [ ] 4.1 Lint after the repair.
  **Proof:** `uv run ruff check --no-cache --output-format concise src/ > docs/relay/baseline-ruff-after.txt; rg -o "[A-Z]+[0-9]+" docs/relay/baseline-ruff-after.txt | sort -u`
  → `PLC2701` only, which `ingestion-chunking` owns; and the line count is strictly less than
  `baseline-ruff-before.txt`.
- [ ] 4.2 **First real pytest measurement.** Budget wall-clock for it: a single import probe during
  scouting exceeded 120 s. `addopts` already carries `--timeout=60` and deselects `integration` and
  `requires_db`.
  **Proof:** `uv run pytest -q > docs/relay/baseline-pytest.txt 2>&1; tail -1 docs/relay/baseline-pytest.txt`
  → a summary line. This file is the baseline every later change cites, under Proof rule 1.
- [ ] 4.3 Type baseline. Measure it; do not carry a prior count forward — a previously recorded ty
  count of 2 was actually 46, and repairing a shadowed import made the number go *up* because thirteen
  suppression comments turned dead.
  **Proof:** `uv run ty check src/ > docs/relay/baseline-ty.txt 2>&1; tail -1 docs/relay/baseline-ty.txt`.
- [ ] 4.4 **First attempt at the live revision.** Expected to fail: `src/alembic/env.py` is known to
  break migration commands, and it needs the now-repaired tree to import at all.
  **Proof:** `uv run alembic current > docs/relay/baseline-alembic.txt 2>&1 || true; cat docs/relay/baseline-alembic.txt`
  → the file holds either a revision identifier or the verbatim failure.
- [ ] 4.5 **Repair `src/alembic/env.py` until 4.4 succeeds.** Scope is strictly the environment module.
  No revision may be authored, edited, or applied in this change.
  **Proof:** `uv run alembic current > docs/relay/baseline-alembic.txt 2>&1; echo $?` → `0`, and the
  recorded identifier matches a `revision` literal present under `src/alembic/versions/`.
- [ ] 4.6 Reconcile the three disagreeing claims about the head in `review.md`: disk carries
  `0001`–`0017`; the handover claims `b3e7c41d92af`, which matches no revision on disk; project memory
  says the database is stamped at `0004`. The measured value wins.
  **Proof:** `review.md` names the measured identifier and states which of the three claims it
  contradicts.

## 5 Restore the specification baseline

- [ ] 5.1 Copy ten archived capability specs into `openspec/specs/<capability>/spec.md`. Reproduce
  requirement and scenario text verbatim; replace the `## ADDED Requirements` header with
  `## Requirements`; carry the delta's `## Purpose` body across unchanged. Format authority is the
  "Main Spec Format Reference" in `.opencode/skills/openspec-sync-specs/SKILL.md` — a main spec never
  contains a delta operation header.

  | Capability | Reqs | Scenarios | Source archive |
  |---|---|---|---|
  | `celery-worker-deployment` | 8 | 19 | `2026-09-07-ingestion-pipeline-unification` |
  | `document-ingestion-pipeline` | 10 | 27 | `2026-09-07-ingestion-pipeline-unification` |
  | `graph-entity-canonicalisation` | 4 | 12 | `2026-09-07-ingestion-pipeline-unification` |
  | `hierarchical-document-chunking` | 7 | 16 | `2026-09-07-ingestion-pipeline-unification` |
  | `hybrid-retrieval-ranking` | 4 | 12 | `2026-09-07-ingestion-pipeline-unification` |
  | `langgraph-checkpointing` | 8 | 24 | `2026-09-07-ingestion-pipeline-unification` |
  | `unified-embedding` | 10 | 24 | `2026-09-07-ingestion-pipeline-unification` |
  | `document-retrieval-schema` | 10 | 27 | `2026-09-07-documents-unified-schema` |
  | `legal-corpus-retrieval` | 4 | 11 | `2026-09-07-agent-tools-unification` |
  | `agent-tool-registry` | 5 | 15 | `2026-09-07-agent-tools-unification` |

  **Proof:** for each restored file, `rg -c '^### Requirement:'` and `rg -c '^#### Scenario:'` equal
  the counts above; and `openspec validate --specs` exits `0`.
- [ ] 5.2 Record in `review.md` the two capabilities deliberately **not** restored, and why.
  `graphiti-init-order` and `embedding-dimension-config` predate the requirement grammar — they use
  `## Scope` / `## Problem` / `## Solution` / `## Verification` and contain zero requirement blocks, so
  there is nothing to restore. Their content is absorbed as ordinary source facts.
  **Proof:** `rg -c '^### Requirement:' openspec/changes/archive/2026-06-22-quality-fixes-batch-2/specs/graphiti-init-order/spec.md; test $? -eq 1`
  → exit `0`.
- [ ] 5.3 Record the four measurements behind the restore in `review.md`, so it is auditable rather
  than asserted: the `typed-exception-handling` comparison, the six archive tick counts, the
  `src/app/shared/rag/docling/embedder.py:44-48` deferral quoted verbatim, and the `AgentToolBundle`
  docstring's "empty tool lists" admission.
  **Proof:** `review.md` contains all four, each with a file path or an archive directory name.
- [ ] 5.4 Confirm this change writes **no** delta against any restored capability.
  **Proof:** `ls openspec/changes/rag-tree-repair/specs/` → `source-tree-integrity` only.

## 6 Close out

- [ ] 6.1 Re-run the whole gate set and confirm the recorded baselines are the ones a reader would
  reproduce today.
  **Proof:** `openspec validate rag-tree-repair --strict` exits `0`; `uv run ruff check --no-cache src/`
  line count equals `docs/relay/baseline-ruff-after.txt`; `uv run pytest -q 2>&1 | tail -1` summary
  counts equal `docs/relay/baseline-pytest.txt`.
