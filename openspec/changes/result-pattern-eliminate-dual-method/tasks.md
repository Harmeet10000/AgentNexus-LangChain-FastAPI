## 1. Repositories — auth

✅ Already refactored (13 wrappers deleted, 13 `_result` renamed).

## 2. Repositories — users (1 method)

- [x] 2.1 Delete `UserAdminRepository.find_by_id` wrapper (line 18-22), rename `find_by_id_result` → `find_by_id` (line 24)

## 3. Repositories — documents (5 rename + 3 new)

- [x] 3.1 Delete wrapper (lines 57-69), rename `get_document_by_user_hash_result` → `get_document_by_user_hash`
- [x] 3.2 Delete wrapper (lines 104-111), rename `get_document_by_id_result` → `get_document_by_id`
- [x] 3.3 Delete wrapper (lines 146-176), rename `create_document_result` → `create_document`
- [x] 3.4 Delete wrapper (lines 271-275), rename `upsert_chunks_result` → `upsert_chunks`
- [x] 3.5 Delete wrapper (lines 322-326), rename `fetch_status_result` → `fetch_status`
- [x] 3.6 Remove unused `app_error_to_exception` import (line 24)
- [x] 3.7 Add `_result` variant + wrapper to `bm25_search` (line 377) — wraps `SQLAlchemyError`
- [x] 3.8 Add `_result` variant + wrapper to `vector_search` (line 412)
- [x] 3.9 Add `_result` variant + wrapper to `trigram_search` (line 453)

## 4. Repositories — search (5 rename + 3 new)

- [x] 4.1 Delete wrapper (lines 46-50), rename `get_document_by_content_hash_result` → `get_document_by_content_hash`
- [x] 4.2 Delete wrapper (lines 80-84), rename `get_document_by_id_result` → `get_document_by_id`
- [x] 4.3 Delete wrapper (lines 152-166), rename `bm25_search_result` → `bm25_search`
- [x] 4.4 Delete wrapper (lines 194-208), rename `vector_search_result` → `vector_search`
- [x] 4.5 Delete wrapper (lines 243-257), rename `trigram_search_result` → `trigram_search`
- [x] 4.6 Remove unused `app_error_to_exception` import (line 18)
- [x] 4.7 Add `_result` variant + wrapper to `create_document` (line 114) — wraps `SQLAlchemyError`
- [x] 4.8 Add `_result` variant + wrapper to `upsert_chunks` (line 132)
- [x] 4.9 Add `_result` variant + wrapper to `fetch_chunks_by_ids` (line 286)

## 5. Services — auth (21 call sites)

- [ ] 5.1 `register` (line 69, 79): match `email_exists` + `create`
- [ ] 5.2 `login` (line 95): match `find_by_email` with `case _` (constant-time)
- [ ] 5.3 `login` (line 115): match `save`
- [ ] 5.4 `logout` (line 128): match `revoke_session`
- [ ] 5.5 `refresh` (line 141-152): match `get_session` + `find_by_id` (rename from `find_by_id_result`)
- [ ] 5.6 `verify_email` (line 172): match `find_by_verification_token_hash`
- [ ] 5.7 `verify_email` (line 177): match `save`
- [ ] 5.8 `resend_verification` (line 181): match `find_by_email` with `case _`
- [ ] 5.9 `resend_verification` (line 190): match `save`
- [ ] 5.10 `forgot_password` (line 199): match `find_by_email` with `case _`
- [ ] 5.11 `forgot_password` (line 209): match `save`
- [ ] 5.12 `reset_password` (line 217): match `find_by_reset_token_hash`
- [ ] 5.13 `reset_password` (line 227): match `save`
- [ ] 5.14 `reset_password` (line 230): match `revoke_all_user_sessions`
- [ ] 5.15 `oauth_callback` (line 267): match `find_or_create_oauth_user`
- [ ] 5.16 `list_sessions` (line 286): match `get_user_sessions`
- [ ] 5.17 `revoke_session` (line 305-310): match `get_session` + `revoke_session`
- [ ] 5.18 `revoke_all_sessions` (line 321): match `revoke_all_user_sessions`
- [ ] 5.19 `_create_session` (line 350): match `store_session`
- [ ] 5.20 Remove unused `Failure`/`Success` imports after refactor
- [ ] 5.21 Ensure `log_expected_failure` is imported (line 9, already present)

## 6. Dependencies — auth (1 call site)

- [ ] 6.1 `get_current_user` (deps.py:106): match `find_by_id` with `case _` wildcard

## 7. Services — documents (8 call sites)

- [ ] 7.1 `upload_document` (line 118): match `get_document_by_user_hash`
- [ ] 7.2 `upload_document` (line 142): match `create_document` + add `log_expected_failure`
- [ ] 7.3 `get_status` (line 184): match `fetch_status` with `case _`
- [ ] 7.4 `search` (lines 233-245): unwrap `bm25_search`/`vector_search`/`trigram_search` from `asyncio.gather`
- [ ] 7.5 `fetch_chunks_by_ids` (line 259): match result
- [ ] 7.6 `process_document_ingestion` (line 471): match `upsert_chunks`
- [ ] 7.7 `_verify_legal_chunks` (line 634): match `upsert_chunks`
- [ ] 7.8 Add `log_expected_failure` to all `Failure` branches

## 8. Services — search (4 call sites)

- [ ] 8.1 `ingest_document` (line 73): match `get_document_by_content_hash`
- [ ] 8.2 `ingest_document` (line 82): match `create_document`
- [ ] 8.3 `get_ingest_status` (line 129): match `get_document_by_id`
- [ ] 8.4 `_run_parallel_search` (lines 350-360): unwrap `AppResult` from each gathered result + `log_expected_failure`

## 9. LangGraph nodes — ingestion_kb (10 fixes)

- [ ] 9.1 Line 98: `result.failure` → `result.failure()`
- [ ] 9.2 Line 99: `result.failure` → `result.failure()`
- [ ] 9.3 Line 121: `result.failure` → `result.failure()`
- [ ] 9.4 Line 122: `result.failure` → `result.failure()`
- [ ] 9.5 Line 160: `result.failure` → `result.failure()`
- [ ] 9.6 Line 161: `result.failure` → `result.failure()`
- [ ] 9.7 Line 262: `result.failure` → `result.failure()`
- [ ] 9.8 Line 263: `result.failure` → `result.failure()`
- [ ] 9.9 Line 308: `result.failure` → `result.failure()`
- [ ] 9.10 Line 309: `result.failure` → `result.failure()`

## 10. LangGraph nodes — reconciliation (10 fixes)

- [ ] 10.1 Line 118: `error_result.failure` → `error_result.failure()`
- [ ] 10.2 Line 120: `error_result.failure` → `error_result.failure()`
- [ ] 10.3 Line 179: `error_result.failure` → `error_result.failure()`
- [ ] 10.4 Line 182: `error_result.failure` → `error_result.failure()`
- [ ] 10.5 Line 191: `error_result.failure` → `error_result.failure()`
- [ ] 10.6 Line 194: `error_result.failure` → `error_result.failure()`
- [ ] 10.7 Line 258: `error_result.failure` → `error_result.failure()`
- [ ] 10.8 Line 260: `error_result.failure` → `error_result.failure()`
- [ ] 10.9 Line 342: `error_result.failure` → `error_result.failure()`
- [ ] 10.10 Line 344: `error_result.failure` → `error_result.failure()`

## 11. Verification

- [ ] 11.1 Run `ruff check` on all changed files
- [ ] 11.2 Run `ruff format` on all changed files
- [ ] 11.3 Run `ty check` on all changed files
