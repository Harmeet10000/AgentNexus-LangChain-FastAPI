# ADRs: agent-message-standard

## ADR-001: Delegation is a transcript message pair, not a side payload (2026-10-03)

* **Status**: Proposed (accepted on merge of this change)
* **Context**: Role handoffs fired inside sub-agent loops and never reached the shared transcript,
  leaving delegations untraceable and untrimmable.
* **Decision**: Every inter-agent delegation is recorded as a linked `AIMessage`+`ToolMessage`
  pair constructed by exactly one shared constructor.
* **Rationale / Alternatives**: Adopting the library's own message pair (over dict payloads or a
  bespoke envelope) keeps `tool_call_id` linkage, LangSmith traceability, and the existing
  trim/summarize pipeline working without modification. Dict payloads stay valid through
  migration and are removed in a follow-up.

## ADR-002: Pairs materialize at node boundaries (2026-10-03)

* **Status**: Proposed (accepted on merge of this change)
* **Context**: Transfer tools execute inside `create_agent` loops, which cannot write parent graph
  state; nodes own the boundary into shared state.
* **Decision**: Node wrappers extract transfer invocations from sub-agent results and append
  pairs via normal state updates (hence through `add_messages`).
* **Rationale / Alternatives**: In-loop middleware was rejected — it cannot touch parent state.
  Boundary extraction is the only seam reaching both worlds, and deletion-rollback is safe
  because routing never reads pairs.
