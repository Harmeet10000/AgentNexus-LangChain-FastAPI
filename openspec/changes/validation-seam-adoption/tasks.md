> Change class: **L**. Tasks ordered by dependency; each has a concrete verification outcome. Strict order: a seam lands only with its parity suite green before the next begins.

## 1. Relationship pass (step 2 — lowest risk, highest fit)

- [x] 1.1 Wrap pass-2 validation in the pipeline: existing validators as `MechanicalEvaluator` predicates; retryable set = `span_mismatch`/`unknown_clause`-class findings; budget = the existing relationship share. Verify: drop-reason regression suite green — non-retryable outcomes byte-identical to pre-pipeline behaviour. (Proof: `relationships.py` routes through `ValidationPipeline`; drop-reason + truth-fidelity suites pass unmodified — 80/80 in `test_relationship_extraction_pass.py::` + `test_ingestion_truth_fidelity.py`.)
- [x] 1.2 Feedback-carrying retry test: a retryable-flawed output is retried with structured feedback and either recovers or drops with the existing reason. Verify: new test green; exhaustion maps to the existing drop/failed state with a `budget_exhausted` decision event. (Proof: 5 new tests in `test_relationship_extraction_pass.py` — retry-with-feedback recovery, retry-exhaustion drop, non-retryable no-retry, share-exhaustion `budget_exhausted` event, seam-named accept event.)

## 2. Legal metadata extraction (step 3)

- [x] 2.1 Mechanical gate: parties/jurisdiction occurrence in boundary text; exactly one feedback-carrying retry on rejection. Verify: fabricated-string fixture → retry → recover-or-fail. (Proof: 15 tests in `tests/unit/features/documents/ingestion/test_legal_metadata_validation.py` — fabricated party retried once with structured feedback then recovers; second rejection → honest absence.)
- [x] 2.2 Persisted `metadata_validation_failed` state; heuristic merge removed. Verify: fixture asserts the persisted state, its consumer-visible distinction from not-extracted, and that no unvalidated metadata is merged anywhere in the path. (Proof: `_metadata_validation_state` tests pin the persisted flag + per-contract truth; `extract_legal_metadata` returns `validation_failed=False` for seam-absent vs `True` for rejected; `_merge_metadata`/`_heuristic_metadata` deleted, `enrich_legal_chunks` unreachable for failures, column fallbacks gated.)

## 3. Ask grade loop (step 4 — parity refactor)

- [x] 3.1 `_grade_context` behind `SemanticEvaluator[RetrievedContext]`; loop behind the pipeline; two-iteration bound unchanged. Verify: implementation compiles; decision events emit per outcome (seam-named). (Proof: `_ContextGradeEvaluator` wraps `_grade_context` verbatim; `_ask` produces one plan→retrieve→rerank iteration per `validate` attempt with `max_attempts=2`; parity suite asserts seam-named `ask_context_grader` accept/escalate events; `ty check src/` green.)
- [x] 3.2 Parity suite: replay every existing grade-loop test case; assert byte-identical responses (shapes, fallback, warnings, cache keys). Verify: parity suite green; the hand-rolled loop's bespoke retry logic is deleted, not duplicated. (Proof: `tests/unit/documents/test_ask_grade_loop_parity.py` (6 tests) + the pre-existing `test_query_grader_pilot.py`/`test_ask_edge_expansion.py` cases all pass unmodified; the `for iteration in range(2)` break-logic loop is deleted — the pipeline owns the retry.)

## 4. Gates

- [x] 4.1 lint-imports 5/5; `uv run ruff check --fix src/` · `uv run ty check src/` · `uv run pytest tests/unit/ -q` · `ast-grep scan src/` — green; all three parity suites runnable in CI. (Proof: lint-imports 5 kept/0 broken; `ty check src/` "All checks passed"; `pytest tests/unit/` 2048 passed with only the 2 pre-existing `test_feature_result_migrations.py` failures from the concurrent legacy_service.py edit; ast-grep 34 findings all pre-existing baseline, none in touched files; ruff clean on all touched files. Parity suites: `test_relationship_extraction_pass.py`, `test_legal_metadata_validation.py`, `test_ask_grade_loop_parity.py` — all under `tests/unit/`.)

## 5. Archive

- [ ] 5.1 Sync deltas into `openspec/specs/document-ingestion-pipeline/spec.md` and `openspec/specs/agentic-retrieval-loop/spec.md`.
