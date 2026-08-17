> Change class (pick one): **S** single-file fix / config / bump / docs · **M** feature in one module · **L** cross-cutting (multi-module, migration, security, public API)

## Why

<!-- Explain the motivation for this change. What problem does this solve? Why now?
     S: 1-2 sentences. M/L: state the problem and the opportunity. -->

## What Changes

<!-- Describe what will change. Be specific about new capabilities, modifications, or removals.
     Mark breaking changes with **BREAKING**. S: a short bullet list. -->

## Scope / Non-Goals

<!-- S: optional, one line if anything is deliberately excluded.
     M/L: state explicitly what is OUT of scope so reviewers know the boundary. -->

## Capabilities

### New Capabilities
<!-- Capabilities being introduced. Use kebab-case for path segments you introduce
     (e.g., user-auth or identity/user-auth) that follow the project's existing
     spec organization. Each creates specs/<capability-path>/spec.md.
     S with no behavior change: leave empty and set skip_specs: true in .openspec.yaml. -->
- `<capability-path>`: <brief description of what this capability covers>

### Modified Capabilities
<!-- Existing capabilities whose REQUIREMENTS are changing (not just implementation).
     Only list here if spec-level behavior changes. Each needs a delta spec file.
     Use the exact existing path under openspec/specs/. Leave empty if no requirement
     changes. A change with no capabilities at all (pure refactor, tooling, docs)
     must set `skip_specs: true` in its .openspec.yaml - openspec validate rejects
     a zero-delta change without that marker. Do not invent a requirement just to
     satisfy validation. -->
- `<existing-capability-path>`: <what requirement is changing>

## Impact

<!-- Affected code, APIs, dependencies, systems. S: keep to one or two lines. -->

## Risks

<!-- What could go wrong, and the mitigation. Omit for S unless risk is unusual. -->
