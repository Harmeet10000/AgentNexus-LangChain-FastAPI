## Context

The project has 10 test files (4 unit, 3 integration, 3 e2e placeholder) for 274 source files. Key gaps:

- **Auth**: 0 tests for register, login, refresh, logout, OAuth, session management
- **Search**: 0 tests for ingestion pipeline, hybrid search, RAG context assembly
- **Circuit breaker**: 0 tests for Redis-backed state transitions
- **Fusion**: existing `test_fusion.py` covers basic cases but not edge cases
- **Chunking**: existing `test_chunking.py` covers basic cases but not edge cases

The auth service at `src/app/features/auth/service.py` uses Beanie (MongoDB) for user model, asyncpg (PostgreSQL) for some queries, and Redis for session storage. The search service at `src/app/features/search/service.py` uses PostgreSQL with pgvector, Redis for caching, and Celery for async ingestion.

## Goals / Non-Goals

**Goals:**
- Integration tests for auth lifecycle (register → login → refresh → logout)
- Integration tests for search pipeline (ingest → search → RAG)
- Unit tests for circuit breaker, fusion, chunking
- Shared test fixtures (async DB session, test client, mocked Redis)
- No production code changes

**Non-Goals:**
- Full test rewrite of existing tests
- E2E tests with real browser
- Load/performance tests
- Test fixtures (deferred to follow-up)

## Decisions

### D1: Test strategy — integration tests with mocked external services

**Decision:** Use real PostgreSQL + Redis for integration tests (via testcontainers or local dev services). Mock LLM calls (Gemini) and external services (Tavily, S3). Beanie/MongoDB tests use a test database.

**Rationale:** Real DB tests catch query bugs (pgvector operators, Redis protocol). Mocking LLMs keeps tests fast and deterministic. This matches the existing pattern in `tests/unit/search/test_rag.py`.

**Alternatives considered:**
- *Mock everything*: misses real persistence bugs — rejected
- *Real LLM calls*: slow, non-deterministic, costs money — rejected
- *SQLite for tests*: pgvector not available — rejected

### D2: Fixture strategy — conftest.py with session-scoped DB

**Decision:** Create `tests/conftest.py` with session-scoped fixtures for PostgreSQL (async session), Redis (mocked with fakeredis), and test client (httpx.AsyncClient). Per-test cleanup via transaction rollback.

**Rationale:** Session-scoped DB fixtures avoid spinning up a new DB per test. Transaction rollback is faster than TRUNCATE. fakeredis avoids requiring a running Redis for unit tests.

**Alternatives considered:**
- *testcontainers per test*: too slow — rejected
- *Shared DB without cleanup*: test pollution — rejected
- *Real Redis for all tests*: requires running Redis — use fakeredis for unit tests

### D3: Auth test flow — end-to-end through service layer

**Decision:** Test `AuthService` methods directly (not HTTP endpoints). This catches service logic bugs without HTTP overhead. Use `httpx.AsyncClient` for endpoint-level tests where needed.

**Rationale:** Service-level tests are faster and more focused. The router layer is thin (just validation + delegation). Endpoint tests add coverage for request parsing but are lower priority.

**Alternatives considered:**
- *Endpoint-only tests*: slower, less focused — rejected for now
- *Repository-only tests*: misses service logic — insufficient
- *Both service + endpoint*: ideal but scope creep — defer to follow-up

### D4: Search test approach — mock embeddings, real DB

**Decision:** Mock `GoogleGenerativeAIEmbeddings` to return fixed vectors. Use real PostgreSQL with pgvector for vector search tests. Mock Celery for ingestion tests.

**Rationale:** Mocking embeddings makes tests deterministic and fast. Real pgvector catches similarity search bugs. Celery mock avoids requiring RabbitMQ for tests.

## Risks / Trade-offs

- **[Test DB availability]** Integration tests require PostgreSQL with pgvector extension. **Mitigation:** Skip integration tests if DB unavailable (`@pytest.mark.skipif`). Unit tests run without DB.
- **[fakeredis divergence]** fakeredis may not implement all Redis commands used by the app. **Mitigation:** Use `fakeredis[aioredis]` with `decode_responses=True`. Fall back to real Redis for integration tests.
- **[Beanie test database]** Beanie requires a running MongoDB for document model tests. **Mitigation:** Skip Beanie-specific tests if MongoDB unavailable. Focus on auth service logic that uses PostgreSQL/Redis.
