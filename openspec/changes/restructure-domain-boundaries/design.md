> Change class: **L**. Planned follow-up; do not apply with the worker migration.

## Context

See proposal.md and assessment.md. This design starts after `consolidate-worker-feature-layout`. Existing public endpoints, task names, MCP protocols, SQL schema and LangGraph checkpoint state are compatibility constraints. The attachment's proposed directory names are suggestions, not evidence that every module should move.

## Goals / Non-Goals

**Goals:** Domain graphs live with domain owners; reusable code has no domain imports; package imports are cheap and predictable; data and wire contracts survive relocation; automated checks cover the first-party RAG integrations.

**Non-Goals:** Rewriting business logic, unifying all errors, creating empty architectural directories, moving MCP just to change its prefix, or deleting deployment-dependent model shims without evidence.

## Decisions

### 1. Enforce an ownership graph with explicit composition roots

Use three categories rather than a single total layer order:

- Domain features depend on pure contracts and injected adapter interfaces.
- Reusable policy/runtime modules depend on shared contracts and configuration; they do not import domain implementations.
- Composition roots (API router assembly, lifespan, worker composition, database registry/seeders, MCP tool assembly) can import features and adapters to wire them together.

Concrete adapters can depend on pure contracts; business services should receive their minimal collaborator via a Protocol rather than import an adapter factory. Import-linter contracts must encode explicit exceptions for composition roots and fail on new violations. Do not exempt entire shared packages.

### 2. Extract only demonstrably reusable primitives

Create `app.retrieval` for shared context/result types, fusion and chunk-policy primitives, and an embedding contract holding the existing persisted vector dimension. Keep database queries, tenant filtering, legal classification and orchestration in documents. Move token claim types and cryptographic encode/decode to `app.security`; keep account policy, revocation and HTTP dependencies in auth. Change every importer and keep environment names, claim fields, token validation and key behavior stable.

If a pure `app.kernel` is introduced, move only the existing ErrorKind/FeatureError/Result typing spine and other already-shared primitives. Leave render_result, APIResponse, status mapping, logging adapters and APIException outside the pure kernel. Preserve every feature error union, code and retryability value. Do not invent Clock/ID abstractions without a caller that needs them.

### 3. Move domain implementations; keep generic runtime reusable

Move `shared/langgraph_layer/agent_saul` to `features/agent_saul/graph`. Move document retrieval graph code to `features/documents/retrieval_graph`. Place legal tools with their data owner and register them through the existing tool registry. Keep generic model factories, embeddings, tracing callbacks, graph execution helpers, memory contracts, shell and web tools reusable. Do not require a simultaneous rename of every generic module to an `ai` umbrella.

Preserve LangGraph node names, state schema/version, checkpointer identifiers, runtime configurable keys, tool names and public service methods. Build compatibility tests from the existing serialized state fixtures before changing imports.

### 4. Make package surfaces intentional

Feature initializers may contain docstrings and pure type exports; no router/service/repository imports. Migrate internal callers to explicit modules. Composition roots import routers directly. Preserve model initialization order until cycles are removed and mapper configuration tests pass. Existing billing subpackages remain one bounded context. Keep feature-owned Celery entry points from the current migration; they are allowed worker adapters, and business orchestration is already in jobs/services.

### 5. Consolidation requires behavioral equivalence

| Choice | Benefit | Cost / Decision |
|---|---|---|
| Merge modules by import count | Quick mechanical choice | Reject: popularity says nothing about correctness, key semantics or lifecycle. |
| One implementation behind preserved adapters | Removes proven duplication while retaining contracts | Adopt after comparing behavior and tests. |
| Keep distinct implementations with shared policy | Preserves different latency, storage or error semantics | Adopt for HTTP/provider rate limits and source-specific chunk engines when semantics differ. |

Outbox factories versus relay/writer logic, Tavily client construction versus service adaptation, and object-store lifecycle versus storage operations can legitimately remain separate. Record their owner and remove only redundant implementation. Keep ingestion's synchronous graph response and documents' durable ingestion API as separate adapters over shared use cases where possible; no endpoint disappears.

For identity, auth owns the User document and credential lifecycle. Move reusable persistence operations behind its explicit identity repository contract; users keeps admin use cases/authorization and maps shared failures to its own error union. Do not merge role/activation/deletion authorization into token primitives.

### 6. Billing and schema safety

Extract pure tax/currency conversion into billing/money.py with preserved rounding and currency behavior. Credits may retain multiple repositories/models/routers; do not replace a coherent package with a large monolithic file merely for visual symmetry.

Move `_MODEL_MODULE_NAMES` into `database.registry` and expose an explicit registration function. Alembic and metadata verification call it. Preserve the full table/constraint identity snapshot. Keep document_vectors and historical compatibility imports until deployment evidence permits removal. A zero-import grep is not evidence that a historical migration or external deployment no longer needs a name. The current reflected-table filter prevents automatic drops but can hide a missing model, so test the exact expected table set.

### 7. Tooling and multi-language layout

First collect a ty diagnostic baseline with first-party RAG included. Resolve diagnostics at adapter boundaries; remove the broad exclusion only when those modules pass. Keep existing passing checks blocking throughout. Audit per-file-ignore globs using file discovery and an explicit rationale for any intentionally future/generated pattern; fail unexplained unmatched paths.

Move Lynk to `tools/lynk` in this repository, preserving `go.mod`, fixtures and CLI behavior. Check Python subprocess calls, deployment scripts, documentation and CI references first. The current scan did not identify a Python caller, so no new adapter is planned. If an existing caller is found, update its path and preserve its existing contract; introducing a new integration is outside scope.

Remove only confirmed unreferenced tombstones and generated output. Keep generated files out of distributions and version control; do not treat local bytecode or regenerated egg-info as production code.

## Risks / Trade-offs

- [Eager re-exports hide dependency cycles] → remove by context, run mapper configuration and fresh-process imports after each batch.
- [Different implementations appear redundant] → behavior matrix before consolidation; retain adapters for differing contracts.
- [Graph relocation affects persisted runs] → stable state/node/tool identifiers and checkpoint compatibility fixtures.
- [Moving security weakens validation] → preserve negative JWT, revocation, tenant and authorization tests.
- [Model shims have deployment dependencies] → keep them unless evidence supports removal; never generate destructive migrations for a path refactor.
- [A massive directory rename obscures functional regressions] → dependency-ordered commits, mechanical moves separate from logic changes, checked boundaries per batch.

## Migration Plan

1. Require the current worker migration to finish; capture route/tool/task/metadata/graph baselines and the existing unrelated verification failures.
2. Audit first-party type coverage and tooling globs; move Lynk and remove only proven disposable artifacts.
3. Remove eager feature initializers and extract pure retrieval/security contracts; add targeted blocking import rules.
4. Relocate domain graphs/tools without changing serialized identifiers; verify domain service and checkpoint compatibility.
5. Consolidate proven duplicates, formalize identity ownership, and keep ingestion adapters compatible.
6. Extract billing money primitives and simplify credits where cohesive.
7. Centralize the model registry and verify identical metadata; retain deployment-dependent shims.
8. Run full Python checks, ast-grep fixture tests, Go tests, installed-package imports and runtime smoke tests. Update docs and rollback instructions.

Deploy the completed source refactor as one coordinated revision after the checks pass. No planned schema or queue change requires data migration or draining. Roll back the code/images together. If implementation reveals a wire or storage change, stop that portion and create the corresponding behavior delta before proceeding.
