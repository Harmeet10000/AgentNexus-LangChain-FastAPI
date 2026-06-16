## Why

The codebase has 10 test files for 274 source files (3.6% ratio). The two highest-risk paths — auth flow and search ingestion — have zero integration tests. A regression in token refresh, session revocation, BM25/vector fusion, or chunk dedup would be invisible until production. This change closes the gap with targeted integration tests for the two paths where a regression causes the most user-visible damage.

## What Changes

### Auth Integration Tests
- Register → Login → Refresh → Logout flow
- Email verification and password reset flows
- OAuth callback flow (mocked userinfo)
- Session revocation (single + all)
- Timing-attack safe login (constant-time response regardless of email existence)
- Token expiry and refresh token rotation

### Search Integration Tests
- Document ingest → chunk → embed → store pipeline
- Hybrid search (BM25 + vector + trigram → RRF fusion)
- RAG search (search → context assembly)
- Duplicate document detection via content hash
- Cache hit/miss behavior

### Unit Tests (Fast Feedback)
- Circuit breaker state transitions (acquire, success, failure, probe)
- RRF fusion ranking logic
- Chunk text splitting

## Capabilities

### New Capabilities
- `auth-integration-tests`: Full auth lifecycle tests against real DB + Redis
- `search-integration-tests`: Ingestion + search pipeline tests with mocked embeddings

### Modified Capabilities
- (none)

## Impact

### Affected Code
- `tests/integration/test_auth.py` — new
- `tests/integration/test_search.py` — new
- `tests/unit/test_circuit_breaker.py` — new
- `tests/unit/test_fusion.py` — new (expands existing)
- `tests/unit/test_chunking.py` — new (expands existing)
- `tests/conftest.py` — shared fixtures (async DB session, test client, mocked Redis)
- `pyproject.toml` — add `pytest-asyncio`, `httpx` to test deps

### Affected APIs
- No production code changes
- No breaking changes

### Dependencies Added
- `pytest-asyncio` (already in dev deps)
- `httpx` (already in dev deps)

### Systems
- CI: `uv run pytest tests/` must pass with 80% coverage threshold
