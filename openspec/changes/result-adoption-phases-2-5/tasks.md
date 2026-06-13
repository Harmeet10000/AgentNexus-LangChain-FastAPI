## 1. Unused Code Cleanup

- [ ] 1.1 Remove `AppFutureResult` type alias from `src/app/shared/result/types.py`
- [ ] 1.2 Remove `AppFutureResult` re-export from `src/app/shared/result/__init__.py`
- [ ] 1.3 Verify `ty check` and `ruff check` pass with zero references to `AppFutureResult`

## 2. Documents Repository — Result Variants

- [ ] 2.1 Add `from returns.result import Failure, Success` and `AppResult`, `NotFoundAppError`, `InfrastructureAppError`, `ConflictAppError` imports to `documents/repository.py`
- [ ] 2.2 Add `_result` variant and thin wrapper for `get_document_by_user_hash`
- [ ] 2.3 Add `_result` variant and thin wrapper for `get_document_by_id`
- [ ] 2.4 Add `_result` variant and thin wrapper for `fetch_status`
- [ ] 2.5 Add `_result` variant and thin wrapper for `create_document`
- [ ] 2.6 Add `_result` variant and thin wrapper for `upsert_chunks`

## 3. Search Repository — Result Variants

- [ ] 3.1 Add imports (`AppResult`, `NotFoundAppError`, `InfrastructureAppError`, `ConflictAppError`, `Success`, `Failure`) to `search/repository.py`
- [ ] 3.2 Add `_result` variant and thin wrapper for `get_document_by_content_hash`
- [ ] 3.3 Add `_result` variant and thin wrapper for `get_document_by_id`
- [ ] 3.4 Add `_result` variant and thin wrapper for `bm25_search`
- [ ] 3.5 Add `_result` variant and thin wrapper for `vector_search`
- [ ] 3.6 Add `_result` variant and thin wrapper for `trigram_search`

## 4. Auth Repository — Complete Conversion

- [ ] 4.1 Add `_result` variant for `UserRepository.find_by_email` + thin wrapper
- [ ] 4.2 Add `_result` variant for `UserRepository.find_by_verification_token_hash` + thin wrapper
- [ ] 4.3 Add `_result` variant for `UserRepository.find_by_reset_token_hash` + thin wrapper
- [ ] 4.4 Add `_result` variant for `UserRepository.create` + thin wrapper
- [ ] 4.5 Add `_result` variant for `UserRepository.save` + thin wrapper
- [ ] 4.6 Add `_result` variant for `UserRepository.email_exists` + thin wrapper
- [ ] 4.7 Add `_result` variant for `UserRepository.find_or_create_oauth_user` + thin wrapper
- [ ] 4.8 Add `_result` variant for `RefreshTokenRepository.store_session` + thin wrapper
- [ ] 4.9 Add `_result` variant for `RefreshTokenRepository.get_session` + thin wrapper
- [ ] 4.10 Add `_result` variant for `RefreshTokenRepository.revoke_session` + thin wrapper
- [ ] 4.11 Add `_result` variant for `RefreshTokenRepository.get_user_sessions` + thin wrapper
- [ ] 4.12 Add `_result` variant for `RefreshTokenRepository.revoke_all_user_sessions` + thin wrapper
- [ ] 4.13 Update `auth/service.py` with `match/case` unwrapping for new Result variants at the service boundary

## 5. Ingestion LangGraph Nodes — Result Helpers

- [ ] 5.1 Add `log_expected_failure` import to `ingestion_kb/nodes.py`
- [ ] 5.2 Convert `_ingestion_failure` helper to return `Failure(AppError)` instead of plain value
- [ ] 5.3 Convert sync validation helpers to return `AppResult` with `Success`/`Failure`
- [ ] 5.4 Update node entrypoints to unwrap helper Results and call `log_expected_failure` at boundary

## 6. Reconciliation LangGraph Nodes — Result Helpers

- [ ] 6.1 Add `log_expected_failure` import to `reconciliation/nodes.py`
- [ ] 6.2 Convert reconciliation sync helpers to return `AppResult` with `Success`/`Failure`
- [ ] 6.3 Update node entrypoints to map `Failure` to `failures` list in state and call `log_expected_failure`

## 7. Verify

- [ ] 7.1 Run `uv run ruff check src/` — zero new violations
- [ ] 7.2 Run `uv run ty check src/` — zero new errors
- [ ] 7.3 Run `uv run ruff format src/` — formatting clean
