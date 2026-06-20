## Context

6 files contain `from X import *` wildcard imports. `ruff` rule `F403` catches these. The project already runs `ruff check` as part of CI. Wildcard imports are a legacy pattern from early development.

## Goals / Non-Goals

**Goals:**
- Replace all 6 wildcard imports with explicit imports
- Enforce `F403` (no wildcard imports) as a hard lint rule going forward
- Ensure no runtime import errors after cleanup

**Non-Goals:**
- Refactoring module structure or splitting large `__init__.py` files
- Adding import ordering rules beyond what ruff already enforces (`isort`)

## Decisions

1. **Use `ruff check --fix F403` to auto-detect used names, then manually verify**
   - Rationale: ruff knows which names are actually referenced in the file. Manual review catches edge cases (e.g., names used in `eval()`, `getattr()`, or string references).
   - Alternative considered: Manual grep per file — slower and error-prone.

2. **Add `F403` to `pyproject.toml` `[tool.ruff.lint] select` if not already present**
   - Rationale: Prevents regression. The project's `pyproject.toml` already has ruff configured; adding `F403` to the selected rules is the smallest-enforcement change.

3. **No `# noqa: F403` exemptions unless explicitly justified**
   - Rationale: Wildcard imports are never required in this codebase. If a module's `__init__.py` re-exports names, explicit imports from the specific sub-module are preferred.

## Risks / Trade-offs

- **Risk**: Missing an import that's only used dynamically (e.g., `globals()`). **Mitigation**: Run full test suite after each file change. `ruff check` + `ty check` catches most issues.
- **Risk**: Merge conflicts with in-flight PRs touching the same files. **Mitigation**: Land this early in a sprint, communicate to team.
