> Change class: **M**. **Gated:** tasks 1.x require `post-llm-validation-core` to have landed; the change as a whole is sequenced after the qna grade-loop refactor (`mount-retrieval-graph` + seam adoption). Each task has a concrete verification outcome.

## 1. Type-level adoption (gated on the core)

- [x] 1.1 Map the verdict path's mechanical gates onto the predicate/report shapes and its exits onto shared-vocabulary decisions; every finding code enters the shared vocabulary. Verify: the existing verdict-path suite passes **unmodified**; type-level tests reach both ACCEPT and ESCALATE. — `verdict_issue_finding`/`verdict_mechanical_report`/`verdict_outcome` in nodes.py reuse `GROUNDING_GATE_REJECTED`/`INVALID_GROUNDING_OUTPUT`/`GROUNDING_VERIFIER_UNAVAILABLE`; test_analysis_safety.py unmodified in the diff; tests/unit/features/agent_saul/graph/test_verdict_validation_vocabulary.py reaches ACCEPT and ESCALATE through the real node.
- [x] 1.2 Finding-locator mapping test: a mechanical gate finding renders as a semantic-report finding with its locator. Verify: new test green. — `test_mechanical_gate_finding_renders_with_its_locator` + `test_quote_span_issue_renders_with_the_evidence_handle_locator` (locator from the real `_finding_evidence_issues` output).

## 2. Sink wiring

- [x] 2.1 Implement `HumanReviewSink` as an adapter over the existing human-review node and wire it in the graph factory. Verify: interrupt-through-sink test — an ESCALATE outcome with a review-class reason fires the existing HITL flow exactly once, through the sink. — `make_human_review_entry()` (nodes.py) returns the sink+node pair; factory `_build_graph_nodes` wires both (`human_review_sink` on SaulGraphNodes); tests/unit/features/agent_saul/graph/test_human_review_sink.py drives the real interrupt+resume through the sink.

## 3. Budget

- [x] 3.1 Conform Saul's ledger to the pipeline's budget port; charge per re-invocation; exhaustion ⇒ ESCALATE `budget_exhausted`. Verify: charge-arithmetic test (recorded charges == actual re-invocations); exhaustion test; no-fresh-allowance assertion. — `PipelineLedgerAdapter` in domain/budget.py satisfies the `BudgetLedger` port; tests/unit/features/agent_saul/domain/test_pipeline_ledger_adapter.py: 3 produce calls ⇒ 2 charges, `model_calls=0` ⇒ ESCALATE/`budget_exhausted`, post-exhaustion charge stays False.

## 4. Boundary assertion

- [x] 4.1 Middleware-stack assertion test for the tool-calling seams (`ModelRetryMiddleware` / `ToolRetryMiddleware` configuration equality pre/post adoption). Verify: test green and wired into CI. — tests/unit/features/agent_saul/graph/test_middleware_boundary.py pins both create_agent seams to the pre-adoption `middleware=[]` configuration (runs in the default unit suite).

## 5. Gates and archive

- [x] 5.1 lint-imports 5/5; `uv run ruff check --fix src/` · `uv run ty check src/` · `uv run pytest tests/unit/ -q` · `ast-grep scan src/` — green; no verdict-path test modified in the diff. — lint-imports 5/5 kept; ruff clean in scope (1 pre-existing `no-self-use` hit in documents/ingestion/legacy_service.py, a concurrent session's file outside this change's scope); ty: all checks passed; pytest tests/unit/: 2077 passed with only the 2 known pre-existing test_feature_result_migrations.py failures (not this change's); ast-grep: only the 5 pre-existing domain/errors.py findings, none new; `git status` shows no modification to any existing agent_saul test.
- [x] 5.2  (archived 2026-11-03; deltas synced)Sync the delta into `openspec/specs/agent-saul-delegation/spec.md`; record the gating dependencies in the archived entry.
