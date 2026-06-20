## 1. Audit str() Calls

- [ ] 1.1 Extract `SECRET_FIELDS` list from `src/app/config/settings.py`
- [ ] 1.2 Run `rg 'str\(settings\.' src/` and catalog matches against SECRET_FIELDS
- [ ] 1.3 Run `rg 'str\(.*\)' src/ | grep -E 'token|key|secret|password|api'` for broader secret-related str() calls
- [ ] 1.4 Create a list of files and line numbers needing fix

## 2. Fix Each File

- [ ] 2.1 Replace `str(settings.FIELD)` with `settings.FIELD.get_secret_value()` for each confirmed SecretStr coercion
- [ ] 2.2 Replace f-string interpolations `f"...{settings.SECRET_FIELD}..."` with `settings.SECRET_FIELD.get_secret_value()`
- [ ] 2.3 Run `uv run ty check <file>` on each modified file to verify type correctness

## 3. CI Check

- [ ] 3.1 Create a shell script (e.g., `scripts/check_secretstr_usage.sh`) that greps for `str()` on SECRET_FIELDS names
- [ ] 3.2 Add the script to CI pipeline (e.g., GitHub Actions or pre-commit hook)
- [ ] 3.3 Test the check: intentionally add a bad `str()` call and verify CI fails

## 4. Documentation

- [ ] 4.1 Add `### SecretStr` section to `.opencode/instructions/PYTHON-TYPING-RULES.md`
- [ ] 4.2 Document `.get_secret_value()` convention and禁止 `str()` coercion
- [ ] 4.3 Include example of correct vs incorrect pattern

## 5. Verify

- [ ] 5.1 Run `uv run ruff check src/` — no new errors
- [ ] 5.2 Run `uv run ty check src/` — no type regressions
- [ ] 5.3 Run `uv run pytest` — full test suite passes
- [ ] 5.4 Run the new CI check script and verify it passes on current code
