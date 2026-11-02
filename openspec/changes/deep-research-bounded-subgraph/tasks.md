> Change class: **M**. Tasks ordered by dependency; each has a concrete verification outcome.

## 1. Subgraph core

- [x] 1.1 Define `DeepResearchState` (objective, focus, scope from runtime context, evidence handles, coverage, step counter, status) and compile the `plan → search_evidence → read_evidence_by_id → synthesize` subgraph with a hard step bound. Verify: the subgraph compiles in the factory's build path; a unit test drives one full happy-path run. — `agent_saul/graph/deep_research.py` (state, 4 nodes, `compile_deep_research_subgraph`, no checkpointer); compiled at factory build via `make_deep_research_tool` (`build_agent_registry`); happy path: `test_full_run_completes_with_one_terminal_envelope`.
- [x] 1.2 Tool wrapper: one `@tool` with explicit Pydantic `args_schema` (`ConfigDict(extra="forbid")` — `objective`, optional `focus`); initial state built from `ToolRuntime[ToolRuntimeContext]`; deny-by-default refusal without valid context. Verify: wrapper schema exposes exactly the two arguments; no-context invocation refuses with a typed failure. — `DeepResearchInput` + `make_deep_research_tool`; `test_wrapper_schema_exposes_exactly_objective_and_focus`, `test_wrapper_denies_without_runtime_context`.

## 2. Bounding and scoping (the three failure modes, as tests)

- [x] 2.1 V6 propagation: a test asserts model-supplied scope-like arguments never enter the subgraph state; the subgraph executes only against server-owned scope. Verify: propagation test red before the explicit construction, green after. — `test_model_supplied_scope_never_reaches_subgraph` (hostile `ToolRuntime.state` + `extra="forbid"` schema: `test_wrapper_rejects_model_supplied_scope_arguments`).
- [x] 2.2 Budget: the step bound draws from Saul's ledger (harden 6.11); exhaustion returns `status: budget_exhausted` with partial handles and no synthesized answer presented as complete. Verify: forced-exhaustion test. — Deviation: the ledger is not wireable through `ToolRuntimeContext`, so the subgraph uses a local stage counter with the same typed exhaustion semantics; the node-level `DEEP_RESEARCH_FANOUT`/`RETRIEVAL_CALLS` ledger gates still apply outside the tool. Forced exhaustion: `test_budget_exhaustion_returns_partial_handles_without_answer` + node mapping `test_node_maps_budget_exhaustion_to_typed_failure_with_partial_handles`.
- [x] 2.3 Run-scoped handles: a handle issued in run N fails membership in run N+1, including on a resumed thread. Verify: cross-run refusal test. — `test_handle_from_run_n_fails_membership_in_run_n_plus_1` (real `EvidenceService`, same thread, run-1 handle refused in run-2; refusal names the run-scoping rule).
- [x] 2.4 No-interrupt: a full run emits zero interrupts and exactly one terminal envelope; `no_evidence` is a status, never an empty success. Verify: full-run + empty-search tests. — `test_subgraph_has_no_internal_checkpointer_and_no_interrupts` (checkpointer is None, no `__interrupt__`), `test_no_evidence_is_a_status_not_an_empty_success`.

## 3. Toolset composition (D3 parity + D5 registration)

- [x] 3.1 Risk binding gains `get_ingestion_state` (factory pattern identical to the compliance branch); risk output discloses incomplete documents. Verify: binding test + disclosure assertion. — `factory.build_agent_registry` appends `get_ingestion_state` to `risk_tools` beside compliance's; READINESS POLICY added to `_RISK_ANALYSIS_SYSTEM_PROMPT`; `test_risk_binding_gains_get_ingestion_state_for_readiness_parity`.
- [x] 3.2 Wrapper registered in the single registry with read-only authority tags; deep_research role's projection asserted at exactly `transfer + deep_research`. Verify: registry + toolset-identity assertions in the plan-level execution test. — `registry.register_deep_research_tool` (tags `side-effect:read`, `capability:deep-research`) + `deep_research_role_projection`; `transfer_to_deep_research` added to handoff roles; asserted in `test_deep_research_role_projection_is_exactly_transfer_plus_research` and the plan-level test.
- [x] 3.3 Plan-level execution test: a scripted plan through the orchestrator reaches deep research, completes, and lands a bounded envelope in the transcript. Verify: end-to-end unit test green. — `test_plan_execution_reaches_deep_research_and_lands_envelope` (orchestrator → deep_research node → metadata-only envelope AIMessage in transcript).

## 4. Gates

- [x] 4.1 `uv run ruff check --fix src/` · `uv run ty check src/` · `uv run pytest tests/unit/ -q` · `ast-grep scan src/` — green, no new diagnostics attributable to this change. — ruff clean on touched paths; `ty check src/` "All checks passed"; full unit suite 2004 passed / 2 failed, both in `test_feature_result_migrations.py` caused by a concurrent session's `documents/ingestion/legacy_service.py` edit (not this change); ast-grep shows the same 5 pre-existing `domain/errors.py` findings before and after.

## 5. Archive

- [ ] 5.1 Sync deltas into `openspec/specs/deep-research-subgraph/` (new) and `openspec/specs/agent-saul-delegation/spec.md`; note the harden 6.9 supersession in the archived change.
