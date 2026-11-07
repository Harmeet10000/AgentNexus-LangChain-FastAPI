# ADRs: ag-ui-copilotkit-console

## ADR-001: Adopt AG-UI for Agent Clients

- **Status:** accepted
- **Date:** 2026-11-07
- **Supersedes:** none

### Context

Agent Saul exposes a bespoke WebSocket frame protocol for text, progress, and human review. Every new client would need to reproduce that adapter, and an approval is coupled to the lifetime of one socket. A maintained, versioned protocol is needed for resumable user-facing clients.

### Decision

Adopt AG-UI 1.0 as the long-lived agent-to-client contract for Agent Saul. Serve it over HTTP and server-sent events. Represent a human gate as a terminal interrupt outcome and resume it with a new authorized run request. Keep the existing WebSocket protocol as a deprecated compatibility surface until a separate retirement change.

### Alternatives

- Keep the bespoke WebSocket contract: rejected because every client needs a custom adapter and human pauses remain connection-bound.
- Bind AG-UI to WebSocket: rejected because it requires a custom client transport and preserves the socket-lifetime failure mode.
- Use bespoke JSON over SSE: rejected because it recreates a protocol without its ecosystem or compatibility contract.

### Consequences

- Interrupt and resume payloads become published contracts and require compatibility discipline.
- Every resume re-resolves current authority.
- WebSocket and AG-UI projections remain behaviorally equivalent while both exist.
- Future clients can consume the same protocol without adopting legacy frames.

## ADR-002: Make the Agent Run Service Transport-Neutral

- **Status:** accepted
- **Date:** 2026-11-07
- **Supersedes:** none

### Context

`AgentSaulService` currently owns WebSocket security state and waits for resume commands inside a connection-scoped loop. An HTTP/SSE request cannot use that lifecycle without duplicating lock, checkpoint, budget, transcript, and authorization logic.

### Decision

Make the application service request-scoped and transport-neutral. It emits a closed projection event union through a `RunSink` port and exposes distinct start and resume phases. Transport adapters encode those facts for WebSocket or AG-UI; the WebSocket adapter composes the phases to preserve its existing wire behavior.

### Alternatives

- Translate only at the encoder: rejected because the service would remain tied to an interactive socket loop.
- Fork the service for AG-UI: rejected because locking, budget, transcript, and authorization would drift.
- Move execution to a Redis-backed background stream: rejected because it changes request ownership and recreates connection state indirectly.

### Consequences

- Locking, checkpointing, budget accounting, transcript recording, and resume authorization have one implementation.
- The active WebSocket path requires recorded golden-frame parity before enablement.
- Adding another transport does not require another graph orchestration service.
- Phase boundaries guarantee checkpoint persistence, lock release, and an explicit terminal or interrupt outcome.

## ADR Review Verdict

- [x] Every durable decision in design.md is captured here
- [x] No decision supersedes an earlier change-local ADR
- [x] No repository-level ADR file is used
