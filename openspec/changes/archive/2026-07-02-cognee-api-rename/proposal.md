## Why

Cognee 1.0 renames the core API methods. The V1 names (`add`, `cognify`, `search`) still work but emit `AuthlibDeprecationWarning` and will be removed before 2.0. Migrating now avoids accumulating deprecation debt and ensures the codebase stays on a supported API surface.

## What Changes

- Rename `cognee.add()` → `cognee.remember()` (store content in episodic memory)
- Rename `cognee.cognify()` → `cognee.improve()` (process and enrich stored content)
- Rename `cognee.search()` → `cognee.recall()` (query stored memories)
- Update `SearchType.INSIGHTS` import path if changed in 1.0
- Remove the `# type: ignore[attr-defined]` suppression on `cognee.search` (now `cognee.recall` should be properly typed)
- No behavioral changes — the new names are 1:1 aliases

## Capabilities

### New Capabilities
- `cognee-v1-api`: Cognee 1.0 API surface (`remember`, `improve`, `recall`, `forget`) with correct imports and type annotations.

### Modified Capabilities
None. This is a pure rename — no spec-level behavior changes.

## Impact

- `src/app/shared/langchain_layer/agents/memory/cognee_client.py` — 3 call sites (`store_final_report`, `store_relationships`, `search_episodic_memory`)
- `src/app/shared/rag/graphiti/write_final_report.py` — calls `cognee_service.store_final_report()` and `cognee_service.store_relationships()` (indirect, no direct cognee import)
- No API contract changes, no migration scripts, no DB changes
