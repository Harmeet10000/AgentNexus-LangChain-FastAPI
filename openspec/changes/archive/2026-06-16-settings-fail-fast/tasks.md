## 1. Settings Validation

- [x] 1.1 Add `PRODUCTION_SECRET_FIELDS` dict to `src/app/config/settings.py`
- [x] 1.2 Add `model_post_init(self, __context: object)` to `Settings` class
- [x] 1.3 In `model_post_init`: check `self.ENVIRONMENT == "production"`
- [x] 1.4 If production: iterate `PRODUCTION_SECRET_FIELDS`, compare `.get_secret_value()` against bad defaults
- [x] 1.5 If bad defaults found: raise `ValueError` with formatted message listing all bad fields
- [x] 1.6 If non-production: log `logger.warning()` with the list of bad fields (allow startup)
- [x] 1.7 Never log or expose actual secret values in error/warning messages

## 2. Testing

- [x] 2.1 Add unit test: `Settings(ENVIRONMENT="production")` raises `ValueError` with bad secrets
- [x] 2.2 Add unit test: `Settings(ENVIRONMENT="production")` succeeds with proper secrets
- [x] 2.3 Add unit test: `Settings(ENVIRONMENT="development")` succeeds with bad secrets (no error)
- [x] 2.4 Add unit test: error message lists all bad field names
- [x] 2.5 Add unit test: error message does not contain actual secret values
- [x] 2.6 Run `uv run pytest tests/unit/test_settings.py -v`

## 3. Lint & Type Check

- [x] 3.1 Run `uv run ruff check src/app/config/settings.py`
- [x] 3.2 Run `uv run ty check src/app/config/settings.py`
