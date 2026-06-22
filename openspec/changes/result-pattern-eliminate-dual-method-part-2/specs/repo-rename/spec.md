# Capability: repo-dual-method-rename

## Purpose

Eliminate all remaining dual-method pairs in `documents/repository.py` and `search/repository.py`. Each pair has a thin public wrapper that calls `_result` variant and unwraps the `AppResult`. The fix: delete the wrapper, rename the `_result` method to drop the suffix.

## ADDED Requirements

### Requirement: bm25_search shall return AppResult in documents/repository.py

The `bm25_search()` method SHALL delete the public wrapper (lines 310-326) and rename `bm25_search_result()` (line 328) to `bm25_search()`. Return type changes from `list[dict[str, Any]]` to `AppResult[list[dict[str, Any]]]`.

#### Scenario: Caller adapts to AppResult return
Given `documents/service.py:244` calls `self.repo.bm25_search(...)`, when the method now returns `AppResult`, then the caller MUST pattern-match with `match ... case Success(v) / case Failure(error)` before passing data to `reciprocal_rank_fusion`.

### Requirement: vector_search shall return AppResult in documents/repository.py

The `vector_search()` method SHALL delete the public wrapper (lines 373-389) and rename `vector_search_result()` (line 391) to `vector_search()`. Return type changes from `list[dict[str, Any]]` to `AppResult[list[dict[str, Any]]]`.

#### Scenario: Same pattern as bm25
Given `documents/service.py:250` calls `self.repo.vector_search(...)`, when the method now returns `AppResult`, then the caller MUST pattern-match with `match ... case Success(v) / case Failure(error)`.

### Requirement: trigram_search shall return AppResult in documents/repository.py

The `trigram_search()` method SHALL delete the public wrapper (lines 442-458) and rename `trigram_search_result()` (line 460) to `trigram_search()`. Return type changes from `list[dict[str, Any]]` to `AppResult[list[dict[str, Any]]]`.

#### Scenario: Same pattern as bm25
Given `documents/service.py:256` calls `self.repo.trigram_search(...)`, when the method now returns `AppResult`, then the caller MUST pattern-match with `match ... case Success(v) / case Failure(error)`.

### Requirement: create_document shall return AppResult in search/repository.py

The `create_document()` method SHALL delete the public wrapper (lines 102-118) and rename `create_document_result()` (line 120) to `create_document()`. Return type changes from `SearchDocument` to `AppResult[SearchDocument]`.

#### Scenario: Caller already handles AppResult
Given `search/service.py:90` calls `match await self.repo.create_document_result(...)`, when the method is renamed to `create_document()`, then the call site MUST change to `match await self.repo.create_document(...)` with no match-arm changes.

### Requirement: upsert_chunks shall return AppResult in search/repository.py

The `upsert_chunks()` method SHALL delete the public wrapper (lines 148-155) and rename `upsert_chunks_result()` (line 157) to `upsert_chunks()`. Return type changes from `None` to `AppResult[None]`.

#### Scenario: Caller needs match block
Given `search/service.py:322` calls `await repo.upsert_chunks(...)`, when the method now returns `AppResult`, then the caller MUST add `match ... case Success(): pass / case Failure(error): raise ...`.

### Requirement: fetch_chunks_by_ids shall return AppResult in search/repository.py

The `fetch_chunks_by_ids()` method SHALL delete the public wrapper (lines 285-292) and rename `fetch_chunks_by_ids_result()` (line 294) to `fetch_chunks_by_ids()`. Return type changes from `dict[str, SearchChunkRecord]` to `AppResult[dict[str, SearchChunkRecord]]`.

#### Scenario: Caller degrades gracefully on Failure
Given `search/service.py:190` calls `chunk_lookup = await self.repo.fetch_chunks_by_ids(...)`, when the method returns `AppResult`, then the caller MUST pattern-match and use empty dict `{}` on Failure.
