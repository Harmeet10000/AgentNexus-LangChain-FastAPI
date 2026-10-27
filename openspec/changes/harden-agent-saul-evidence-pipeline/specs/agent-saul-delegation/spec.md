## ADDED Requirements

### Requirement: Existing WebSocket/HITL handoffs preserve authorization context

The existing transcript-visible WebSocket/HITL handoff contract MUST remain intact while every dispatched role receives
the same server-derived run authorization context. A handoff MUST NOT copy authority from model-visible arguments.

#### Scenario: Role handoff retains transcript shape
- **WHEN** the orchestrator delegates to risk or compliance
- **THEN** the existing linked assistant/tool handoff pair MUST be emitted and the target MUST receive scoped context

#### Scenario: HITL pause does not widen scope
- **WHEN** a run resumes after a human-review pause
- **THEN** its handoff and evidence reads MUST use revalidated context for the current thread and tenant

