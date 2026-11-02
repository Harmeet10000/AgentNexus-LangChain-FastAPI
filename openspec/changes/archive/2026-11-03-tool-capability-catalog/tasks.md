> Change class: **M**. Tasks ordered by dependency; each has a concrete verification outcome.

## 1. Structured authority metadata

- [x] 1.1 Add typed authority fields to the registry's registration surface (capability identifier, side-effect class, egress classification); register the five system/filesystem tools and web search with values matching their interim tags. Verify: tag/metadata agreement test green; disagreement raises a registration error. — `ToolRegistry.register(capability=, side_effect=, egress=)` + `ToolAuthority` in `base.py`; registrations migrated in `registry.py`; agreement/disagreement covered by `tests/unit/ai/test_tool_capability_policy.py::test_registrations_carry_structured_authority_matching_tags` / `::test_authority_tag_disagreement_is_a_registration_error` (green).
- [x] 1.2 Migration test: the enforcement suite for the five tools passes unmodified after the fields land. Verify: `pytest tests/unit/ai/test_system_tool_capabilities.py -q` green with no edits. — 18 passed, zero edits to the file.

## 2. Policy source

- [x] 2.1 Add the role → grants policy file/module; load fail-fast at startup (unknown role tolerated as empty, unknown capability identifier is an error); populate `ToolRuntimeContext.capabilities` from it at every context-build site. Verify: policy tests — context set == policy entry per role; unknown role → empty; unknown capability → startup error. — `src/app/ai/tools/policy.py`; `_analysis_tool_context(state, role=...)` in `nodes.py` feeds risk/compliance/deep_research contexts from `grants_for(role)`; `validate_policy` runs at the final startup registration (`register_deep_research_tool`); tests in `test_tool_capability_policy.py` + `test_policy_consistency.py::test_context_capability_set_equals_policy_entry_per_role` / `::test_unknown_role_context_grants_nothing` (green).
- [x] 2.2 Initial policy encodes each role's *current effective* grants exactly. Verify: consistency test (task 3.2) green before any narrowing. — rows: orchestrator/risk/compliance/qna/unbound = ∅, deep_research = {deep-research}; consistency test green.

## 3. Selector scoping and egress

- [x] 3.1 Selector candidate set = policy-authorized ∩ candidates, computed before ranking. Verify: high-relevance unauthorized tool is absent from the projection; subset invariant holds for every role at every turn. — `PolicyToolScopeMiddleware` in `guardrails.py` installed before `LLMToolSelectorMiddleware` in `build_default_middleware_stack`; `::test_high_relevance_unauthorized_tool_is_absent_from_projection` / `::test_scoping_denies_by_default_without_grants` (green).
- [x] 3.2 Projection-consistency test over all roles: model-visible toolset ⊆ policy-authorized, and the ≤7 projection cap holds. Verify: test walks every role and fails naming the policy row on violation. — `tests/unit/features/agent_saul/graph/test_policy_consistency.py::test_every_role_projection_is_authorized_and_within_cap` walks risk/compliance/deep_research/orchestrator/qna via the real factory bindings; assertion messages name the policy row (green).
- [x] 3.3 Web search migrates to the standard `ToolRuntime` injection seam and requires `web.search` at execution; schema stays model-clean. Verify: no-grant refusal names the capability; granted-path parity test; schema assertion. — `web_search.py` converted to `@tool(args_schema=WebSearchInput)` with injected runtime + `require_capability` (shell refusal idiom via shared `base.require_capability`); refusal/granted/schema tests green (`::test_web_search_refuses_without_grant`, `::test_web_search_granted_path_unchanged`, `::test_web_search_schema_stays_model_clean`).

## 4. Gates

- [x] 4.1 `uv run ruff check --fix src/` · `uv run ty check src/` · `uv run pytest tests/unit/ -q` · `ast-grep scan src/` — green, no new diagnostics attributable to this change.

## 5. Archive

- [x] 5.1  (archived 2026-11-03; deltas synced)Sync deltas into `openspec/specs/tool-capability-catalog/` (new) and `openspec/specs/agent-tool-registry/spec.md`; record in the archived change that the enforcement change's interim tags are now derived views.
