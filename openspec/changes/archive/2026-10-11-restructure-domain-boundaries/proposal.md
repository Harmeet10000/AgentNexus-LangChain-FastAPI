> Change class: **L** — follow-up architecture migration; planning only.

## Why

After producer/worker separation, reusable infrastructure still reaches into domain code and eager package initializers enlarge import cycles. The supplied code-review notes identify useful ownership improvements, but several proposed merges would erase intentional boundaries unless their behavior is checked first.

## What Changes

- Make feature package initializers lightweight; use explicit contracts instead of eager router/service exports.
- Extract reusable retrieval and token contracts; place Agent Saul and document-specific graphs/tools with their owning features.
- Clarify ownership of auth/admin user writes and the two ingestion surfaces, retaining their existing behavior.
- Consolidate only behaviorally equivalent implementations, retaining distinct resource factories, adapters and policies when their responsibilities differ.
- Keep billing as one context, extract pure money primitives, and simplify credits where it reduces complexity.
- Put explicit model registration in a dedicated database composition module and verify metadata completeness before considering shim removal.
- Audit first-party RAG type coverage, stale tooling globs and packaging; move the separate Go tool out of Python's package root after verifying its callers.
- **BREAKING** internal Python paths change; HTTP, MCP, database, Celery wire contracts and checkpoint state remain compatible.

## Scope / Non-Goals

This change is separate from `consolidate-worker-feature-layout`, per the user's explicit selection. It is planned now and must not be applied during that migration. Preserve that change's producer-safe contracts and feature-owned thin task adapters. No speculative broker/event-bus switch, wholesale error hierarchy replacement, automatic database drops, separate repository creation, or package deletion chosen by importer counts.

## Capabilities

### New Capabilities

None. This is a behavior-preserving ownership/tooling refactor; `skip_specs: true` is intentional.

### Modified Capabilities

None. Existing feature-error-contract, source-tree-integrity, unified-embedding, ingestion-delegation, document-ingestion-pipeline, agent-saul-delegation, mcp-directory-restructure, transactional-outbox and worker-runtime-isolation requirements constrain implementation. Changes to their observable behavior require a separate delta before implementation.

## Impact

Feature packages, reusable retrieval/security modules, domain graphs/tools, billing, Alembic registry, Python packaging, Go tool location and verification configuration. See assessment.md for the claim-by-claim review of the second supplied attachment, and design.md for ownership and migration sequence.

## Risks

- Import-only moves can alter Pydantic/FastAPI runtime annotation resolution and task registration; test real imports and route schemas.
- Moving LangGraph code can break persisted state or node names; preserve serialized state and graph identifiers.
- Missing models can be hidden by Alembic's safety filter; assert the expected metadata set explicitly.
- Ingestion APIs currently expose different contracts; retain both until adapters prove equivalent behavior.
- Existing deployment references may call Lynk outside the Python tree; preserve CLI behavior and verify build paths before relocation.
