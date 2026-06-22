# Capability: repo-service-call-sites

## Purpose

Update 5 service-layer call sites to correctly pattern-match on `AppResult` after the repo methods switch to single-method returns. Fix 1 stale `find_by_id_result` reference.

## MODIFIED Requirements

### Requirement: Fix stale `find_by_id_result()` reference in users/service.py

The call SHALL change `self._user_repo.find_by_id_result(user_id)` to `self._user_repo.find_by_id(user_id)`. This fixes a runtime bug: `find_by_id_result` was removed in the previous refactor.

#### Scenario: Admin user lookup succeeds
Given `_get_user_or_raise("valid-user-id")` is called, when the repo method is invoked via `find_by_id()` (not `find_by_id_result()`), then the existing `match` arms handle `AppResult[User | None]` correctly.

### Requirement: Rename `create_document_result()` to `create_document()` in search/service.py

The call SHALL change `match await self.repo.create_document_result(...)` to `match await self.repo.create_document(...)`.

#### Scenario: Ingestion flow unchanged
Given `ingest_document()` processes a new document, when the call site uses `create_document()` (the renamed method), then the match arms `case Success(document): ... / case Failure(error): ...` work identically.

### Requirement: Pattern-match `fetch_chunks_by_ids()` result in search/service.py

The call SHALL replace `chunk_lookup = await self.repo.fetch_chunks_by_ids(...)` with `match await ...: case Success(chunk_lookup): ... case Failure(error): chunk_lookup = {}`.

#### Scenario: Chunk lookup fails gracefully
Given `hybrid_search()` fetches chunks for fused results, when the chunk lookup fails, then the search response is built with an empty lookup dict and partial results.

### Requirement: Pattern-match `upsert_chunks()` result in search/service.py

The call SHALL replace `await repo.upsert_chunks(...)` with `match await repo.upsert_chunks(...): case Success(): pass case Failure(error): raise ...`.

#### Scenario: Upsert failure raises exception
Given `process_ingestion_document()` upserts chunks during ingestion, when the upsert fails, then the error is logged and an exception is raised.

### Requirement: Pattern-match each search in asyncio.gather in documents/service.py

The gather block SHALL replace the triple-assignment `asyncio.gather` with per-result pattern matching so each of the 3 parallel searches can fail independently.

#### Scenario: One index fails, others still contribute
Given `search()` runs 3 parallel searches (bm25, vector, trigram), when the vector index is unreachable, then `Failure` is logged for vector search but bm25 and trigram results still contribute to the fused output.
