## 1. Verify Cognee 1.0 API Surface

- [x] 1.1 Run `uv run python -c "import cognee; print([m for m in dir(cognee) if not m.startswith('_')])"` to confirm `remember`, `improve`, `recall` exist
- [x] 1.2 Run `uv run python -c "from cognee import SearchType; print(SearchType.INSIGHTS)"` — confirmed REMOVED in 1.0, dropped import
- [x] 1.3 Run `uv run python -c "import inspect; import cognee; print(inspect.signature(cognee.remember))"` to confirm signature matches old `add()`

## 2. Rename API Calls in cognee_client.py

- [x] 2.1 In `store_final_report()`: replace `cognee.add(...)` with `cognee.remember(...)`
- [x] 2.2 In `store_final_report()`: replace `cognee.cognify(...)` with `cognee.improve(...)`
- [x] 2.3 In `store_relationships()`: replace `cognee.add(...)` with `cognee.remember(...)`
- [x] 2.4 In `store_relationships()`: replace `cognee.cognify(...)` with `cognee.improve(...)`
- [x] 2.5 In `search_episodic_memory()`: replace `cognee.search(...)` with `cognee.recall(...)`
- [x] 2.6 Remove `# type: ignore[attr-defined]` from the recall call
- [x] 2.7 Remove `# type: ignore[call-arg]` from keyword arguments on recall

## 3. Update Docstrings

- [x] 3.1 Update `store_final_report()` docstring: `add() + cognify()` → `remember() + improve()`
- [x] 3.2 Update `store_relationships()` docstring if it mentions `add/cognify`
- [x] 3.3 Update `search_episodic_memory()` docstring: `search()` → `recall()`

## 4. Lint and Type Check

- [x] 4.1 Run `uv run ruff check src/app/shared/langchain_layer/agents/memory/cognee_client.py` — clean (no new errors)
- [x] 4.2 Run `uv run ty check src/app/shared/langchain_layer/agents/memory/cognee_client.py` — clean (no new errors)
- [x] 4.3 Fix any new lint/type errors from the rename — none found

## 5. Verify No Other Cognee Call Sites

- [x] 5.1 Run `grep -rn "cognee\.\(add\|cognify\|search\)" src/` — zero V1 calls remain
- [x] 5.2 Confirm `write_final_report.py` uses the wrapper methods (no direct cognee import)
