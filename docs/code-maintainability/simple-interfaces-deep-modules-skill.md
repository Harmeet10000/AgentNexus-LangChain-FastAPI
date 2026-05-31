---
name: simple-interfaces-deep-modules
description: Use when a subsystem feels hard to change because callers know too much about internals, configuration is duplicated, helper sprawl is growing, or you need to hide complexity behind a small stable interface in a large codebase.
---

# Simple Interfaces, Deep Modules

## Overview

Use this skill when code is technically modular but still hard to change. The goal is to concentrate complexity inside a small number of deep modules so callers stay obvious, stable, and hard to misuse.

## When to Use

- Callers know internal storage, transport, or event-shape details
- The same config object or branching logic appears in multiple call sites
- A route, handler, or component is doing policy work
- A module exports many helpers but has no obvious main entrypoint
- Changing internals forces edits in many callers

Do not use this for one-off cleanup with no reuse pressure.

## Core Pattern

Bad:

```python
graph = deps.graph
redis = deps.redis
checkpointer = deps.checkpointer

if checkpointer is None:
    raise ServiceUnavailableException("Persistence unavailable")

frame = map_event(event)
if event.get("output", {}).get("status"):
    ...
```

Good:

```python
runtime = deps.websocket_runtime
frame = runtime.map_event_to_frame(event)
```

The caller expresses intent. The module owns the ugly branching.

## Quick Reference

Use this sequence:

1. Find repeated caller knowledge.
2. Name the capability, not the mechanism.
3. Move branching inward.
4. Replace multi-step calling protocols with one entrypoint.
5. Test the seam, not the internals.

## Implementation Rules

- Expose capability, not representation.
- Prefer one canonical builder or factory over repeated config blobs.
- Put reusable policy in service methods, dependency builders, base procedures, or option factories.
- Keep leaf call sites stupid.
- Let modules be deep internally, but narrow publicly.

## Preferred Shapes

### Canonical builder

```python
def build_agent_saul_websocket_runtime(...) -> AgentSaulWebsocketRuntime:
    ...
```

### Base dependency alias

```python
AgentSaulWebsocketRuntimeDep = Annotated[
    AgentSaulWebsocketRuntime,
    Depends(get_agent_saul_websocket_runtime),
]
```

### Single mapping seam

```python
def map_graph_event_to_ws_frame(event: GraphEvent) -> WSFrame | None:
    ...
```

## Smells

- DTO contract lies about runtime behavior
- Optional infrastructure leaks into flows that do not use it
- Callers manually stitch helpers together in the right order
- Internal prefixes, keys, or event quirks escape the owner module
- The public API grows by adding helpers instead of deepening one entrypoint

## Common Mistakes

- Mistake: splitting code into many helpers and calling that abstraction
Fix: measure abstraction quality by how little the caller must know.

- Mistake: exporting raw internals for convenience
Fix: add a delegating public method or typed result instead.

- Mistake: testing private branch behavior instead of contract
Fix: test inputs, outputs, and invariants at the seam.

## Repo Notes

In this repo, prefer applying this skill to:

- websocket runtime/dependency bundles
- graph-event to transport-frame mapping
- LangChain/LangGraph factories
- auth and session policy boundaries

Reference: `docs/code-maintainability/simple-interfaces-deep-modules.md`

## Chosen Ones

Shallow modules do not remove complexity. They export it. The signature stays short, but the caller pays the real cost in hidden prerequisite knowledge. Deep modules are the opposite: they look almost suspiciously simple from the outside because the complexity has been forced to stay where it belongs.
