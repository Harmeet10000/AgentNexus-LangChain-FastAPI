# Change: <title>

**Class:** S | M | L   <!-- S: single-file fix/config/docs; M: one module/subsystem; L: cross-cutting (multi-module, data migration, security boundary, public API, new external dependency) -->

## Why

<1-2 sentences: the problem or opportunity, and why now.>

## What Changes

- <Specific change; mark breaking with **BREAKING**>

## Capabilities

<!-- Contract with the specs phase: research openspec/specs/ first. Every capability named here gets a delta spec file. -->

### New Capabilities

- `kebab-path-name` — <one line>

### Modified Capabilities

- `existing-kebab-path` — <which requirements change; leave empty if implementation-only>

<!-- Every change declares ≥1 capability or sets skip_specs: true in its .openspec.yaml. Do not invent a requirement to satisfy validation. -->

## Impact

- **Code**: <modules, files>
- **APIs**: <endpoints, contracts>
- **Dependencies**: <packages, services>
- **Async flows**: <websocket topics, commands, timers/expiry, presence, telemetry — if any, discovery.md is mandatory for M/L>
