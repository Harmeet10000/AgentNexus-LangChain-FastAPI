## Purpose

Defines how Agent Saul's `qna` node uses the compiled retrieval graph as its grounded-answer engine:
the graph is invoked inside the node — never exposed as a tool — with retrieval context constructed
from server-owned state, explicit state mapping across the boundary, budget drawn from Agent Saul's
ledger, and answers delivered through the existing transcript event vocabulary.

## ADDED Requirements

### Requirement: The qna node invokes the compiled retrieval graph as its engine

Agent Saul's `qna` node SHALL answer grounded queries by invoking the compiled retrieval graph inside
the node (node-internal subgraph invocation, the same structural pattern as the deep-research node).
The retrieval graph SHALL NOT be registered as an agent tool, SHALL NOT appear in any role's toolset,
and no role's toolset SHALL change because of this engine.

#### Scenario: A grounded answer flows through a node-internal graph invocation

- **WHEN** the qna node handles a grounded-answer intent
- **THEN** the answer SHALL have been produced by an invocation of the compiled retrieval graph made
  inside the node

#### Scenario: The graph is not a tool

- **WHEN** any role's toolset is resolved from the tool registry
- **THEN** it SHALL NOT contain the retrieval graph or any wrapper exposing it, and the role toolsets
  SHALL be unchanged from their pre-engine definition

#### Scenario: Precedent consistency with deep research

- **WHEN** the qna engine wiring is reviewed
- **THEN** it SHALL follow the same node-wraps-bounded-subgraph pattern as the deep-research node,
  with no tool-calling surface added to the schema-locked qna node

### Requirement: Retrieval context is constructed from server-owned state

The qna node SHALL construct the retrieval graph's initial state from server-owned values — the
tenant, the allowed document scope, and the document version scope — carried by the agent's
server-owned runtime context. Identity or scope supplied by the model SHALL NOT reach the retrieval
graph, and an explicitly empty authorized scope SHALL yield the grounded fallback envelope rather
than an unscoped search. Context propagation SHALL be verified by a construction test that asserts
the initial retrieval state's contents.

#### Scenario: Server-owned scope reaches the graph

- **WHEN** the qna node invokes the graph
- **THEN** the initial retrieval state SHALL carry the tenant, allowed document scope, and version
  scope resolved from the server-owned runtime context

#### Scenario: Model-supplied identity is forbidden

- **WHEN** a model response or message payload names a tenant, document scope, or version
- **THEN** that value SHALL NOT appear in the retrieval state; only server-owned values do

#### Scenario: Empty authorized scope stays bounded

- **WHEN** the server-owned allowed document scope is empty
- **THEN** the graph SHALL be driven to the grounded fallback envelope and no unscoped retrieval
  SHALL run

### Requirement: State crosses the boundary by explicit mapping

The qna node SHALL map agent state to retrieval state on entry and retrieval results to agent state
on exit through an explicit mapping; the two state schemas SHALL share no keys. Only the answer text,
its citations as evidence handles, and typed coverage or error metadata SHALL enter the agent
checkpoint — no hydrated chunk payload, retrieval state, or graph internals SHALL be persisted into
agent state.

#### Scenario: The mapping is explicit in both directions

- **WHEN** the qna node enters and exits the graph invocation
- **THEN** every field crossing the boundary SHALL be named in the mapping, and no retrieval state
  key SHALL be read from or written into agent state by key sharing

#### Scenario: Only answer metadata is checkpointed

- **WHEN** a graph invocation completes
- **THEN** the agent state delta SHALL contain the answer, evidence-handle citations, and typed
  coverage/errors — and no hydrated chunk text or retrieval working state

### Requirement: The qna engine draws from the budget ledger

The graph invocation SHALL draw from Agent Saul's budget ledger: the node SHALL check the remaining
budget before invoking the graph, SHALL charge the ledger per stage of spend the graph reports, and
SHALL attribute that spend per stage. The retrieval loop's grade phase SHALL remain bounded at two
iterations, and an exhausted budget SHALL block the invocation with the existing typed
budget-exhausted outcome.

#### Scenario: An exhausted budget blocks the engine

- **WHEN** the ledger reports no remaining budget for the engine's dimensions
- **THEN** the node SHALL return the typed budget-exhausted outcome and no graph invocation SHALL run

#### Scenario: Spend is attributed per stage

- **WHEN** a graph invocation completes
- **THEN** the ledger SHALL have been charged with spend attributed to the stages that executed, in a
  form that downstream tracing can consume per stage

#### Scenario: The grade loop stays bounded

- **WHEN** grading repeatedly reports insufficient context inside the engine
- **THEN** the loop SHALL stop at two grade iterations and return the grounded fallback

### Requirement: The qna engine is a process-scoped compiled graph

The retrieval graph the qna engine uses SHALL be compiled at most once per serving process and SHALL
receive its request-scoped collaborators — the repository, cache client, language models, and
relationship-graph client — at invocation time rather than capturing them at compile time, per the
`compiled-graph-lifecycle` capability. When the graph is not provisioned in the serving process, the
qna node SHALL fail closed with a typed unavailable outcome rather than falling back to an unscoped
or single-shot answering path.

#### Scenario: The engine graph compiles once

- **WHEN** the serving process handles multiple sessions that reach the qna engine
- **THEN** the retrieval graph SHALL have been compiled once, before the first invocation

#### Scenario: Collaborators arrive per invocation

- **WHEN** the qna node invokes the graph
- **THEN** the request-scoped collaborators SHALL have been supplied through the invocation context,
  and the compiled graph SHALL capture neither a session nor a repository

#### Scenario: An unprovisioned engine fails closed

- **WHEN** the serving process has no provisioned retrieval graph
- **THEN** the qna node SHALL return a typed unavailable outcome and SHALL NOT substitute its own
  answering path

### Requirement: The qna answer reaches the transcript through the existing event vocabulary

The qna node SHALL deliver the engine's answer to the transcript through the existing WebSocket
event vocabulary — the same node-status event stream every other node uses — and the answer's
citations SHALL carry evidence handles resolvable through the agent's evidence registry.

#### Scenario: The answer streams through existing events

- **WHEN** the engine returns an answer
- **THEN** the transcript SHALL receive it through the same event frames other nodes emit, with no
  new event vocabulary introduced for the engine

#### Scenario: Citations carry evidence handles

- **WHEN** the engine returns citations
- **THEN** each citation SHALL be registered as an evidence handle resolvable through the evidence
  registry, and the transcript message SHALL reference the handles rather than raw chunk payloads
