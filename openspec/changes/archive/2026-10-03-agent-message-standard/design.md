# Design: agent-message-standard

## Context

See proposal.md (Why). Current state, relevant to the approach:

* Saul role handoffs fire *inside* `create_agent` sub-agent loops (`risk_agent`, `compliance_agent`
  in `factory.py`). Only the structured output (`RiskAnalysisOutput`, `ComplianceOutput`) flows
  back into `LegalAgentState`; transfer tool invocations never reach the shared transcript.
* Main-graph routing reads typed state (`orchestrator_action`, conditional edges in `graph.py`),
  never the transfer dicts — so making delegations visible changes observability, not routing.
* `messages.py` already owns transcript helpers (`trim_by_token_count`, `delete_tool_messages`,
  `filter_tool_messages`, `manage_context`) operating on `BaseMessage` lists.

## Goals / Non-Goals

* Goals: one pair constructor used by all three graphs; delegations visible in transcript and
  traces; trim pipeline pair-aware; dict contract intact through migration.
* Non-Goals: changing routing decisions; changing structured-output contracts (`LegalAgentState`
  fields stay the source of truth for data); removing `TransferPayload` (follow-up change).

## Decisions

### 1. Constructor lives in `messages.py`, not `handoff.py`

`make_delegation_pair(role, reason) -> tuple[AIMessage, ToolMessage]`: assistant message carries
the reason as content plus one `transfer_to_<role>` tool call; tool message carries the same
reason with the matching call id. Rationale: `messages.py` already owns transcript-shape helpers
and is imported by all three graphs; `handoff.py` owns tool definitions. Alternative (method on
`TransferPayload`) rejected: payload is the transitional dict contract, and attaching the new
standard to the artifact being phased out invites permanent dual ownership.

### 2. Pairs materialize at node boundaries, not inside sub-agent loops

Node wrappers (`risk_analysis_node`, `compliance_node`, deep-research supervisor/researcher nodes,
ingestion stage nodes) scan the sub-agent result's message list for transfer invocations and
append the constructed pair to shared state via normal state update (hence through
`add_messages`). Alternative (middleware inside `create_agent`) rejected: middleware runs inside
the loop and cannot write parent graph state; post-hoc extraction at the boundary is the only
seam that touches both worlds.

### 3. Dual-write through migration

Handoff tools keep returning the `TransferPayload` dict (routing/tests unchanged); nodes
additionally append pairs. Rollback is deletion of the append lines — routing never reads pairs,
so rollback cannot misroute. Removal of the dict contract is explicitly a follow-up change with
its own spec delta.

### 4. Trim treats pairs atomically

`delete_tool_messages`/`filter_tool_messages` gain pair-awareness: a delegation assistant message
and its linked tool message are kept together, dropped together, or replaced together by one
`HumanMessage` summary naming the delegation. Rationale: half-pairs are worse than no pairs —
an unlinked `ToolMessage` is uninterpretable and an unanswered tool call looks like a failure.

## Risks / Trade-offs

* [Risk] Transcript growth: pairs add 2 messages per delegation → Mitigation: pair-aware
  summarization already replaces dropped pairs with one summary entry.
* [Risk] `tool_call_id` collision across retries → Mitigation: constructor generates a fresh id
  per invocation; reducer upserts by id so retries replace rather than duplicate.
* [Risk] Tests asserting exact transcript shapes break → Mitigation: expected, marked BREAKING;
  updated in the same change, not left red.

## Migration Plan

1. Add constructor + pair-aware trim handling (no callers yet; all existing tests green).
2. Wire Saul nodes (risk, compliance), then deep-research nodes, then ingestion nodes — one graph
   per task, tests updated per graph.
3. Full suite + `ty` + `ruff` gates green before merge. Rollback: revert node-boundary appends.

## Open Questions

None — the dict-removal follow-up is deliberately out of scope, not an open question.
