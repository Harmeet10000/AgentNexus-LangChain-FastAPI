# Proposal: agent-message-standard

## Why

Agent-to-agent delegation in this repo is invisible: handoffs travel as plain-dict tool payloads outside the shared transcript, so delegations lose `tool_call_id` linkage, LangSmith traceability, and context-management handling. Standardizing delegation as real `AIMessage`+`ToolMessage` pairs fixes that in one rule.

## What Changes

* Every inter-agent delegation in `agent_saul`, `open_deep_search`, and the ingestion graph is recorded in the shared transcript as an `AIMessage` (with `tool_calls`) followed by its linked `ToolMessage` — the exact pair shape `langchain_core` 1.6.2 and `langgraph` 1.2.11 already implement and merge via `add_messages`.
* One shared delegation-message constructor is defined once and used by all three graphs; per-graph specs state only what differs (which roles, which transcript field).
* **BREAKING**: transcripts gain delegation message pairs where dict payloads previously flowed off-transcript. Anything asserting exact transcript contents (tests, trim/summarize expectations, LangSmith queries) must be updated to the new shape. The `TransferPayload` dict contract at the tool boundary stays valid during migration and is removed in a follow-up.
* `manage_context` strategies (summarize/trim/delete) explicitly cover delegation pairs: a delegation `AIMessage` without its linked `ToolMessage` (or vice versa) must never survive trimming.

## Capabilities

New capabilities (one spec each):
* `agent-saul-delegation` — delegation pairs for Saul role handoffs (`transfer_to_<role>`), plus the shared constructor definition all graphs use.
* `open-deep-search-delegation` — delegation pairs for supervisor/researcher/compress message flow.
* `ingestion-delegation` — delegation pairs for ingestion-graph stage handoffs.

No existing capability requirements change; no modified capabilities.

## Impact

* Code: `tools/handoff.py`, Saul `factory.py`/`nodes.py`/`graph.py` routing reads, `open_deep_search` graph/nodes, ingestion graph nodes, `messages.py` context pipeline.
* Tests: transcript-shape assertions, trim/summarize/filter tests, handoff/registry tests, step-budget tests.
* Observability: delegation events become visible in LangSmith traces and OTLP transcript handling; trim behavior changes for transcripts containing pairs.
* Dependencies: none new — uses pinned `langchain_core`/`langgraph` message and reducer APIs only.
