## 1. Unused Code Cleanup

- [x] 1.1 Remove `AppFutureResult` type alias from `src/app/shared/result/types.py`
- [x] 1.2 Remove `AppFutureResult` re-export from `src/app/shared/result/__init__.py`
- [x] 1.3 Verify `ty check` and `ruff check` pass with zero references to `AppFutureResult`

## 2. Documents Repository — Result Variants

- [x] 2.1 Add `from returns.result import Failure, Success` and `AppResult`, `NotFoundAppError`, `InfrastructureAppError`, `ConflictAppError` imports to `documents/repository.py`
- [x] 2.2 Add `_result` variant and thin wrapper for `get_document_by_user_hash`
- [x] 2.3 Add `_result` variant and thin wrapper for `get_document_by_id`
- [x] 2.4 Add `_result` variant and thin wrapper for `fetch_status`
- [x] 2.5 Add `_result` variant and thin wrapper for `create_document`
- [x] 2.6 Add `_result` variant and thin wrapper for `upsert_chunks`

## 3. Search Repository — Result Variants

- [x] 3.1 Add imports (`AppResult`, `NotFoundAppError`, `InfrastructureAppError`, `ConflictAppError`, `Success`, `Failure`) to `search/repository.py`
- [x] 3.2 Add `_result` variant and thin wrapper for `get_document_by_content_hash`
- [x] 3.3 Add `_result` variant and thin wrapper for `get_document_by_id`
- [x] 3.4 Add `_result` variant and thin wrapper for `bm25_search`
- [x] 3.5 Add `_result` variant and thin wrapper for `vector_search`
- [x] 3.6 Add `_result` variant and thin wrapper for `trigram_search`

## 4. Auth Repository — Complete Conversion

- [x] 4.1 Add `_result` variant for `UserRepository.find_by_email` + thin wrapper
- [x] 4.2 Add `_result` variant for `UserRepository.find_by_verification_token_hash` + thin wrapper
- [x] 4.3 Add `_result` variant for `UserRepository.find_by_reset_token_hash` + thin wrapper
- [x] 4.4 Add `_result` variant for `UserRepository.create` + thin wrapper
- [x] 4.5 Add `_result` variant for `UserRepository.save` + thin wrapper
- [x] 4.6 Add `_result` variant for `UserRepository.email_exists` + thin wrapper
- [x] 4.7 Add `_result` variant for `UserRepository.find_or_create_oauth_user` + thin wrapper
- [x] 4.8 Add `_result` variant for `RefreshTokenRepository.store_session` + thin wrapper
- [x] 4.9 Add `_result` variant for `RefreshTokenRepository.get_session` + thin wrapper
- [x] 4.10 Add `_result` variant for `RefreshTokenRepository.revoke_session` + thin wrapper
- [x] 4.11 Add `_result` variant for `RefreshTokenRepository.get_user_sessions` + thin wrapper
- [x] 4.12 Add `_result` variant for `RefreshTokenRepository.revoke_all_user_sessions` + thin wrapper
- [x] 4.13 Update `auth/service.py` with `match/case` unwrapping for new Result variants at the service boundary

## 5. Ingestion LangGraph Nodes — Result Helpers

- [x] 5.1 Add `log_expected_failure` import to `ingestion_kb/nodes.py`
- [x] 5.2 Convert `_ingestion_failure` helper to return `Failure(AppError)` instead of plain value
- [x] 5.3 Convert sync validation helpers to return `AppResult` with `Success`/`Failure`
- [x] 5.4 Update node entrypoints to unwrap helper Results and call `log_expected_failure` at boundary

## 6. Reconciliation LangGraph Nodes — Result Helpers

- [x] 6.1 Add `log_expected_failure` import to `reconciliation/nodes.py`
- [x] 6.2 Convert reconciliation sync helpers to return `AppResult` with `Success`/`Failure`
- [x] 6.3 Update node entrypoints to map `Failure` to `failures` list in state and call `log_expected_failure`

## 7. Verify

- [x] 7.1 Run `uv run ruff check src/` — zero new violations
- [x] 7.2 Run `uv run ty check src/` — zero new errors
- [x] 7.3 Run `uv run ruff format src/` — formatting clean
