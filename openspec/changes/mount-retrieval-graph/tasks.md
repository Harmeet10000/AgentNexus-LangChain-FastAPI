> Change class: **L**. Tasks are ordered by dependency: parity port → parity suite → qna engine wiring (behind the parity gate) → HTTP cutover → inline deletion → docs. No production implementation is included in this proposal.

## 1. Parity port — inline-only capabilities into the graph nodes (no serving change)

- [ ] 1.1 Add the `edge_grounding` node (between `post_process` and `context_grader`) porting `fetch_relationship_edges` + `expand_along_edges` + span resolution + `_edge_citations` provenance discipline from `documents.py` (`_edge_grounding_for_reranked`/`_resolved_edge_blocks`/`_edge_citations`), fail-closed and additive; verify an edge-free query produces byte-identical behaviour to the edge-free contract and update the node/edge shape assertion in the same change.
- [ ] 1.2 Replace the graph's `_answer_cache_key` with the version-aware derivation (tenant, authorized scope, resolved document versions, pipeline version, evidence policy — the inline `_build_answer_cache_key` contract); verify a document-version or pipeline-version change produces a different key and the inline derivation matches exactly.
- [ ] 1.3 Thread the request evidence policy (`require_graphiti_verified` → `graph_indexed_only`/`mechanically_verified`) through retrieval state into the hybrid retrieval node's filter params and the empty-scope fallback envelope; verify `/legal/ask`-shaped requests filter on graph-indexed evidence and the empty-scope fallback carries the request's policy.
- [ ] 1.4 Compute the truthful coverage envelope (documents requested/ready, graph-indexed, graph-unavailable) from hydrated rows and scope-unavailable documents, and port `_warnings_from_rows`/`_merge_warning_lists` into the graph's finalisation; verify the scope-unavailable warning and deduplicated warning list match the inline contract.
- [ ] 1.5 Port the injected grader seam into the `context_grader` node (optional grader via invocation context, temperature 0.1, seam-absent degrade preserved verbatim); verify both seam-present and seam-absent grades match `_grade_context` behaviour.
- [ ] 1.6 Surface per-stage stage-usage entries (plan, source-identifier, retrieve/fusion, rerank, grade, generate, edge-grounding) in the graph's result state, without importing anything from `features/agent_saul`; verify each executed stage appears with its counts.
- [ ] 1.7 Emit a span per executed stage and a structured grade decision-log event (outcome, iteration, escaped-to-fallback flag) from the grader; verify spans and events exist for a two-iteration run including the cap-exhaustion escape.

## 2. Parity suite — the cutover gate and permanent drift killer

- [ ] 2.1 Build the parity suite: record the inline contract's responses for a fixed query set (sufficient grade, insufficient-then-rewritten grade, cap-exhausted fallback, empty scope, cache hit, edge-grounded, warnings, unavailable scope) against fakes, then run the same queries through the compiled graph and assert identical response shapes, citation provenance, fallback behaviour, and cache-key derivation; verify it passes only when every ported capability (1.1–1.5) matches.
- [ ] 2.2 Add the drift-killer guarantee: intentionally mutate one ported behaviour in a scratch commit and verify the suite fails naming the divergent behaviour, then revert; record the suite as a permanent test that gates any future change to either side.
- [ ] 2.3 Add the access-recording parity check: the service wrapper around the graph invocation records `document.read` for the same scope/outcome combinations the inline `ask` wrapper did (denied empty scope included); verify with the existing audit-recording fakes.

## 3. qna engine wiring — Agent Saul exposure (behind the parity gate being green)

- [ ] 3.1 Provision the compiled retrieval graph once per process in lifespan with the context schema for per-invocation collaborators (repository, cache client, language models, graphiti client, reranker, optional grader seam); verify compile-once, that no session/repository is captured at compile, and that provisioning failure degrades rather than aborting startup.
- [ ] 3.2 Rewrite `make_qna_node` (`agent_saul/graph/nodes.py`) to invoke the provisioned graph node-internally: explicit `LegalAgentState` → `RetrievalState` mapping on entry (query from transcript; tenant, `allowed_doc_ids`, version scope from server-owned runtime context) and `GeneratedAnswer` + citations → answer message, evidence handles, coverage/errors on exit; verify the V6 construction test asserts the initial retrieval state's contents and that model-supplied identity/scope never appears in it.
- [ ] 3.3 Assert the boundary is not a tool: the retrieval graph and any wrapper are absent from every role's toolset and the registry; verify risk/compliance/orchestrator toolsets are byte-identical to their pre-engine definitions.
- [ ] 3.4 Add the state-mapping test: no shared state keys between the two schemas, and only the answer, evidence-handle citations, and typed coverage/errors enter the agent checkpoint (no hydrated chunk text or retrieval working state); verify against the deep-research precedent's checkpoint fixtures.
- [ ] 3.5 Wire the budget draw: pre-check the ledger before invocation, charge per-stage spend from the graph's stage-usage entries after it, and return the typed `BUDGET_EXHAUSTED` outcome without invoking when the budget is gone; verify per-stage attribution lands in the ledger and the grade loop stays capped at two iterations.
- [ ] 3.6 Deliver the answer through the existing WS event vocabulary (node-status mapping / astream_events loop) with evidence-handle citations resolvable through the evidence registry; verify the transcript frames are the existing vocabulary and the handles resolve.

## 4. HTTP cutover — the routers move to the compiled graph

- [ ] 4.1 Retarget `/search/ask` and `/legal/ask` in `documents/api/http/router.py` to the graph-served service path with their evidence policies (`mechanically_verified` / `graph_indexed_only`) and access-audit recording intact; verify the full parity suite still passes with the routers serving the graph and the manual smoke (`/search/ask`, `/legal/ask`) returns the recorded shapes.
- [ ] 4.2 Add the fail-closed path: when the graph is not provisioned in the serving process, both endpoints return a typed unavailability rather than falling back to any other answering path; verify with an unprovisioned app fixture.

## 5. Inline deletion — the one deletion the retired plan permits

- [ ] 5.1 Delete the inline `ask`/`_ask` bodies and their now-orphaned helpers from `documents/application/queries/documents.py` once every prior section is green; verify `rg` finds no second inline plan→retrieve→grade→generate implementation on the served path and the full gate (`uv run ruff check --fix src/`, `uv run ty check src/`, `uv run pytest`, `ast-grep scan src/`) is green.
- [ ] 5.2 Correct the false prose in `documents/api/http/legacy_ingest_dependencies.py` (lines 8–13 claim a commented-out lifespan assignment by "decision (D17)" while `lifecycle/lifespan.py` provisions `app.state.ingestion_graph`); rewrite the docstring to state the actual provisioning truth; verify the docstring matches runtime reality and no behavioural change ships with it.

## 6. Docs — retired-plan residue and tracking

- [ ] 6.1 Update `openspec/changes/archive/2026-10-31-plan-bulletproof-ingestion.md`'s retirement header: mark step 6 (retrieval unification) DONE with the cutover decision and the surviving residue pointers (G4/G5/G6/G7 remain on issue #107; Phase E quarantine ledger is its own item); verify the header matches the shipped state.
- [ ] 6.2 If the `gh` CLI is authenticated, add a comment to issue #107 noting step 6 is delivered by this change and pointing at the change directory; otherwise record the intended note in the change directory so the next session can post it; verify the issue comment exists or the recorded note does.
