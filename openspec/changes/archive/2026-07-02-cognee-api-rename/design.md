## Context

Agent Saul uses Cognee for episodic and procedural memory. The current code calls the V1 API methods (`add`, `cognify`, `search`) which are deprecated in Cognee 1.0. The new API uses `remember`, `improve`, `recall`. The existing `cognee-saul-memory-migration` change (0/25 tasks) handles memory architecture restructuring — this change is a prerequisite rename that should land first.

## Goals / Non-Goals

**Goals:**
- Replace all V1 API calls with 1.0 equivalents in `cognee_client.py`
- Fix type annotations (remove `# type: ignore` suppressions that existed because V1 was untyped)
- Verify the import path for `SearchType` is still valid in 1.0
- Zero behavioral change — same inputs, same outputs, same error handling

**Non-Goals:**
- Memory architecture restructuring (handled by `cognee-saul-memory-migration`)
- Adding new Cognee features (e.g., `forget`, session memory, access control)
- Updating the `CogneeStore` placeholder class (stub, not wired)
- Changing Cognee config setup (`set_llm_config`, `set_graph_db_config`, `set_relational_db_config`)

## Decisions

### D1: Rename at call sites, not wrapper layer

**Choice:** Rename directly in `cognee_client.py` — the only file with direct `cognee.*` calls.

**Why not a compatibility shim:** Only 3 call sites. A wrapper adds indirection for no benefit. The upstream API is a clean 1:1 rename.

### D2: Keep `SearchType.INSIGHTS` import

**Choice:** Keep `from cognee import SearchType` and use `SearchType.INSIGHTS` in `recall()`.

**Why:** The `SearchType` enum is still the documented way to specify search mode in 1.0. No rename detected for this enum.

### D3: Remove stale `type: ignore` comments

**Choice:** Remove `# type: ignore[attr-defined]` on `cognee.search` (now `cognee.recall`) and `# type: ignore[call-arg]` on keyword args.

**Why:** The 1.0 API has proper type stubs. If the stubs are still incomplete, we'll see it in `ty check` and can re-add targeted suppressions.

## Risks / Trade-offs

- [Risk] Cognee 1.0 may have changed `SearchType.INSIGHTS` → something else → Verify with import check before merging. **Mitigation:** Run `python -c "from cognee import SearchType; print(SearchType.INSIGHTS)"` as a gate.
- [Risk] `cognee.remember()` / `cognee.improve()` may have slightly different signatures → **Mitigation:** Check function signatures with `inspect.signature()` before committing. The deprecation warning says "still work", implying same signatures.
- [Risk] Existing `cognee-saul-memory-migration` change may conflict → **Mitigation:** This rename is a clean, isolated diff. Land it first; the architecture change rebases on top.

## Migration Plan

1. Rename methods in `cognee_client.py`
2. Run `uv run ruff check src/app/shared/langchain_layer/agents/memory/cognee_client.py`
3. Run `uv run ty check src/app/shared/langchain_layer/agents/memory/cognee_client.py`
4. Verify imports: `uv run python -c "import cognee; print(dir(cognee))"`
5. No rollback needed — V1 names still work, this is forward-looking cleanup
