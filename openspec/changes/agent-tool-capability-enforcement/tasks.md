> Change class: **M**. Tasks ordered by dependency; each has a concrete verification outcome. D0 is already implemented in the working tree — tasks marked done record what landed and where; the unchecked items close the review's named gaps so archive never syncs an untested requirement.

## 1. Executor enforcement (landed 2026-10-27)

- [x] 1.1 `capabilities: tuple[str, ...]` on `ToolRuntimeContext` (`features/documents/tools/authorization.py`) — default-deny shape; the field is absent unless policy grants it.
- [x] 1.2 Grant check in all five system/filesystem tools (`app/ai/tools/shell.py`): no runtime / missing grant → typed failure naming the capability. Verify: `pytest tests/unit/ai/test_system_tool_capabilities.py -q` → 12 passed.
- [x] 1.3 Workspace jail: `TOOL_WORKSPACE_ROOTS` setting (`config/settings.py`), `_resolve_in_jail` with resolve-before-containment (symlink-safe), unset-jail refusal naming the setting.
- [x] 1.4 Shell explicit-cwd rule: missing `cwd` → refusal naming the jailed-working-directory requirement.
- [x] 1.5 `WebSearchInput` → plain `BaseModel` (`app/ai/tools/web_search.py`).

## 2. Registry authority metadata (landed 2026-10-27)

- [x] 2.1 `side-effect:*` / `capability:*` tags on the five registrations (`app/ai/tools/registry.py`); legacy category tags unchanged.

## 3. Gaps to close before archive (review §gaps)

- [x] 3.1 Named test: authority metadata discoverable per tool (capability + side-effect tags) and legacy tag selection unchanged. Verify: new assertions in the enforcement suite pass.
- [x] 3.2 Named test: web-search argument schema is a plain schema model exposing exactly its declared fields. Verify: assertion passes against the registered tool's schema.
- [x] 3.3 Apply-time gate: `uv run ruff check src/app/ai/tools/ src/app/config/settings.py src/app/features/documents/tools/authorization.py`, `uv run pytest tests/unit/ai/ -q`, `ast-grep scan src/app/ai/` — all green, no new diagnostics attributable to this change. (Note: a concurrent session moved authorization.py to `src/app/features/documents/application/tools/`; the ruff gate passes against the new path.)

## 4. Archive

- [ ] 4.1 Sync the delta into `openspec/specs/agent-tool-registry/spec.md` via the archive workflow; confirm the five requirements appear with their scenarios.
