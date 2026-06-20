## 1. Identify Wildcard Imports

- [ ] 1.1 Run `rg 'from .* import \*' src/` and list all 6 files
- [ ] 1.2 For each file, run `ruff check --select F403 <file>` to confirm detection

## 2. Fix Each File

- [ ] 2.1 For each file, run `ruff check --fix --select F403 <file>` to auto-generate explicit imports
- [ ] 2.2 Manually verify the generated imports — check for edge cases (names used in `eval()`, `getattr()`, string references)
- [ ] 2.3 Run `python -c "import <module>"` for each affected module to verify no ImportError

## 3. Enforce F403 in pyproject.toml

- [ ] 3.1 Check if `F403` is already in `[tool.ruff.lint] select` in `pyproject.toml`
- [ ] 3.2 If not, add `F403` to the select list
- [ ] 3.3 Run `uv run ruff check src/` and verify zero F403 violations

## 4. Verify

- [ ] 4.1 Run `uv run ruff check src/` — zero F403 errors
- [ ] 4.2 Run `uv run ruff format src/`
- [ ] 4.3 Run `uv run pytest` to confirm no regressions
