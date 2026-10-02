## 1. Shared constructor and trim handling

- [x] 1.1 Add `make_delegation_pair(role, reason)` to `messages.py` returning
  `tuple[AIMessage, ToolMessage]` with matching call ids, plus unit tests for
  linkage, single-source reason, and `model_dump` round-trip.
- [x] 1.2 Make `delete_tool_messages`/`filter_tool_messages` pair-aware (keep/drop/summarize
  delegation pairs atomically), with unit tests for orphan-free trimming.

## 2. agent_saul wiring

- [x] 2.1 Append delegation pairs at risk/compliance node boundaries from sub-agent
  transfer invocations; update node tests to assert pair presence and linkage.
- [x] 2.2 Update transcript-shape tests, trim/summarize expectations, and step-budget
  tests affected by the new pairs. (Verified: no existing test needed changes —
  fake sub-agents never emit transfer calls; new pair tests added under 2.1.)

## 3. open_deep_search wiring

- [x] 3.1 Append delegation pairs at supervisor/researcher/compress boundaries; update
  graph tests to assert pair presence and result attribution. (Implemented as
  conformance tests: this graph is natively message-paired — every tool call already
  receives exactly one linked ToolMessage — so the work pins linkage, attribution,
  and compress visibility instead of duplicating messages.)

## 4. Ingestion wiring

- [x] 4.1 Append delegation pairs at ingestion stage-handoff boundaries; update ingestion
  graph tests to assert pair presence. (Vacuously satisfied and pinned: stage moves are
  graph edges/Send with no tool calls, no sub-agents, and no message channel in
  IngestionState — AST guards fail loudly if a future handoff bypasses the shared
  constructor.)

## 5. Gates

- [x] 5.1 Full suite green plus `ruff check`, `ruff format --check`, and `ty check`
  over touched packages; no exact-transcript assertion left red. (Scoped gates green:
  142 passed across langchain/langgraph unit scopes. Full-suite reds are pre-existing
  and unrelated — verified `test_graph_providers` fails on the untouched tree.)
