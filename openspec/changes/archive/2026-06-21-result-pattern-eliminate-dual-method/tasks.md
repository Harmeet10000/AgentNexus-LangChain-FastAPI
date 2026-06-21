## 1. Repositories — auth

- [ ] 1.1 Delete `UserRepository.find_by_id` wrapper, rename `find_by_id_result` → `find_by_id`
- [ ] 1.2 Delete `UserRepository.find_by_email` wrapper, rename `find_by_email_result` → `find_by_email`
- [ ] 1.3 Delete `UserRepository.find_by_verification_token_hash` wrapper, rename `_result` → primary
- [ ] 1.4 Delete `UserRepository.find_by_reset_token_hash` wrapper, rename `_result` → primary
- [ ] 1.5 Delete `UserRepository.email_exists` wrapper, rename `_result` → `email_exists`
- [ ] 1.6 Delete `UserRepository.create` wrapper, rename `create_result` → `create`
- [ ] 1.7 Delete `UserRepository.save` wrapper, rename `save_result` → `save`
- [ ] 1.8 Delete `UserRepository.find_or_create_oauth_user` wrapper, rename `_result` → primary
- [ ] 1.9 Delete `RefreshTokenRepository.store_session` wrapper, rename `_result` → `store_session`
- [ ] 1.10 Delete `RefreshTokenRepository.get_session` wrapper, rename `_result` → `get_session`
- [ ] 1.11 Delete `RefreshTokenRepository.revoke_session` wrapper, rename `_result` → `revoke_session`
- [ ] 1.12 Delete `RefreshTokenRepository.get_user_sessions` wrapper, rename `_result` → primary
- [ ] 1.13 Delete `RefreshTokenRepository.revoke_all_user_sessions` wrapper, rename `_result` → primary
- [ ] 1.14 Remove unused `app_error_to_exception` import

## 2. Repositories — users

- [ ] 2.1 Delete `UserAdminRepository.find_by_id` wrapper, rename `find_by_id_result` → `find_by_id`

## 3. Repositories — documents

- [ ] 3.1 Delete `DocumentRepository.get_document_by_user_hash` wrapper, rename `_result` → primary
- [ ] 3.2 Delete `DocumentRepository.get_document_by_id` wrapper, rename `_result` → primary
- [ ] 3.3 Delete `DocumentRepository.create_document` wrapper, rename `_result` → primary
- [ ] 3.4 Delete `DocumentRepository.upsert_chunks` wrapper, rename `_result` → `upsert_chunks`
- [ ] 3.5 Delete `DocumentRepository.fetch_status` wrapper, rename `_result` → `fetch_status`
- [ ] 3.6 Remove unused `app_error_to_exception` import

## 4. Repositories — search

- [ ] 4.1 Delete `SearchRepository.get_document_by_content_hash` wrapper, rename `_result` → primary
- [ ] 4.2 Delete `SearchRepository.get_document_by_id` wrapper, rename `_result` → primary
- [ ] 4.3 Delete `SearchRepository.bm25_search` wrapper, rename `_result` → `bm25_search`
- [ ] 4.4 Delete `SearchRepository.vector_search` wrapper, rename `_result` → `vector_search`
- [ ] 4.5 Delete `SearchRepository.trigram_search` wrapper, rename `_result` → `trigram_search`
- [ ] 4.6 Remove unused `app_error_to_exception` import

## 5. Services — auth

- [ ] 5.1 Update `AuthService.login`: match on `find_by_email` returning `AppResult`
- [ ] 5.2 Update `AuthService.refresh`: `find_by_id` instead of `find_by_id_result` (pattern match already exists)
- [ ] 5.3 Update `AuthService.verify_email`: match on `find_by_verification_token_hash`
- [ ] 5.4 Update `AuthService.resend_verification`: match on `find_by_email`
- [ ] 5.5 Update `AuthService.forgot_password`: match on `find_by_email`
- [ ] 5.6 Update `AuthService.reset_password`: match on `find_by_reset_token_hash`
- [ ] 5.7 Update `AuthService.logout`: match on `revoke_session`
- [ ] 5.8 Update `AuthService.oauth_callback`: match on `find_or_create_oauth_user`

## 6. Dependencies — auth

- [ ] 6.1 Update `get_current_user`: match on `find_by_id` returning `AppResult`

## 7. Services — documents

- [ ] 7.1 Update `upload_document`: match on `get_document_by_user_hash`, `create_document`
- [ ] 7.2 Update `get_status`: match on `fetch_status`
- [ ] 7.3 Update `process_document_ingestion`: match on `upsert_chunks`
- [ ] 7.4 Update `_verify_legal_chunks`: match on `upsert_chunks`

## 8. Services — search

- [ ] 8.1 Update `ingest_document`: match on `get_document_by_content_hash`, `create_document`
- [ ] 8.2 Update `get_ingest_status`: match on `get_document_by_id`
- [ ] 8.3 Update `_run_parallel_search`: wrap `asyncio.gather` results in failure matching

## 9. Verification

- [ ] 9.1 Run `ruff check` on all changed files
- [ ] 9.2 Run `ruff format` on all changed files
- [ ] 9.3 Run `ty check` on all changed files
