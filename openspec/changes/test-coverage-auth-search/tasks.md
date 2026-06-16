## 1. Test Fixtures

- [ ] 1.1 Create `tests/conftest.py` with session-scoped PostgreSQL async engine fixture
- [ ] 1.2 Add `async_session` fixture with `begin()` + rollback per test
- [ ] 1.3 Add `redis` fixture using `fakeredis.aioredis.FakeRedis`
- [ ] 1.4 Add `client` fixture using `httpx.AsyncClient` with `app.main:app`
- [ ] 1.5 Add `auth_service` fixture injecting `UserRepository` + `RefreshTokenRepository`
- [ ] 1.6 Add `search_service` fixture injecting `SearchRepository` + mocked Redis + mocked embeddings
- [ ] 1.7 Verify fixtures work: `uv run pytest tests/conftest.py --co`

## 2. Auth Integration Tests

- [ ] 2.1 Create `tests/integration/test_auth.py`
- [ ] 2.2 Test `register`: creates user, returns `UserResponse`, password is hashed
- [ ] 2.3 Test `register`: duplicate email raises `ConflictException`
- [ ] 2.4 Test `login`: valid credentials returns `TokenResponse` with access + refresh tokens
- [ ] 2.5 Test `login`: wrong password raises `UnauthorizedException`
- [ ] 2.6 Test `login`: non-existent email raises `UnauthorizedException` (same message)
- [ ] 2.7 Test `login`: disabled account raises `UnauthorizedException`
- [ ] 2.8 Test `login`: unverified email raises `UnauthorizedException`
- [ ] 2.9 Test `refresh`: valid token returns new access token
- [ ] 2.10 Test `refresh`: expired/revoked token raises `UnauthorizedException`
- [ ] 2.11 Test `logout`: revokes refresh token in Redis
- [ ] 2.12 Test `list_sessions`: returns active sessions with metadata
- [ ] 2.13 Test `revoke_session`: removes specific session
- [ ] 2.14 Test `revoke_all_sessions`: removes all except current
- [ ] 2.15 Test `forgot_password`: generates reset token
- [ ] 2.16 Test `reset_password`: updates password, revokes all sessions
- [ ] 2.17 Run `uv run pytest tests/integration/test_auth.py -v`

## 3. Search Integration Tests

- [ ] 3.1 Create `tests/integration/test_search.py`
- [ ] 3.2 Mock `build_embedding_client` to return fixed 768-dim vectors
- [ ] 3.3 Test `process_ingestion_document`: chunks text, stores embeddings
- [ ] 3.4 Test ingestion: empty document returns 0 chunks
- [ ] 3.5 Test ingestion: content hash dedup prevents re-ingestion
- [ ] 3.6 Test `hybrid_search`: finds documents by keyword (BM25)
- [ ] 3.7 Test `hybrid_search`: finds documents by semantic similarity (vector)
- [ ] 3.8 Test `reciprocal_rank_fusion`: merges results from multiple sources
- [ ] 3.9 Test `assemble_rag_context`: groups chunks by document, merges adjacent
- [ ] 3.10 Test `assemble_rag_context`: respects `max_tokens` limit
- [ ] 3.11 Test cache: first search stores in Redis, second returns cached
- [ ] 3.12 Test cache: `bypass_cache=True` skips cache
- [ ] 3.13 Run `uv run pytest tests/integration/test_search.py -v`

## 4. Unit Tests (Fast Feedback)

- [ ] 4.1 Create `tests/unit/test_circuit_breaker.py`
- [ ] 4.2 Test `acquire`: returns `ALLOW` when state is closed
- [ ] 4.3 Test `failure`: increments failure count
- [ ] 4.4 Test `failure`: opens circuit after threshold breaches
- [ ] 4.5 Test `success`: resets failure count and closes circuit
- [ ] 4.6 Test `probe`: allows one request in half-open state
- [ ] 4.7 Expand `tests/unit/search/test_fusion.py`: test RRF with 3 sources
- [ ] 4.8 Expand `tests/unit/search/test_fusion.py`: test empty input
- [ ] 4.9 Expand `tests/unit/search/test_chunking.py`: test overlap behavior
- [ ] 4.10 Expand `tests/unit/search/test_chunking.py`: test unicode content
- [ ] 4.11 Run `uv run pytest tests/unit/ -v`

## 5. CI Integration

- [ ] 5.1 Verify `uv run pytest tests/ --cov=src --cov-fail-under=80` passes
- [ ] 5.2 Add test summary to CI output
