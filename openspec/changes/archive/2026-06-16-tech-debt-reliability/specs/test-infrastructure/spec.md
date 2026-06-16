# Capability: test-infrastructure

## Purpose
Establish production-grade test infrastructure enabling reliable integration, contract, and chaos testing across the full stack.

## Requirements

### R1: Testcontainers Integration
- PostgreSQL testcontainer with pgvector extension
- Redis testcontainer with hiredis protocol
- Neo4j testcontainer with APOC plugin
- Session-scoped fixtures that share containers across test modules
- Container startup timeout: 30s
- Automatic cleanup after test session

### R2: Factory-Boy Data Factories
- `UserFactory` — generates auth users with configurable roles
- `DocumentFactory` — generates document records with realistic metadata
- `ChunkFactory` — generates chunk records with embeddings
- `TaskResultFactory` — generates Celery task results
- Factories use `factory.Factory` (not `factory-boy` BaseModel factories for Pydantic)

### R3: Integration Test Suite
- Document ingestion pipeline end-to-end: upload → parse → classify → embed → store → verify
- Hybrid search: BM25 + vector + trigram → RRF fusion
- RAG pipeline: search → context assembly → grading → generation
- Graphiti knowledge graph: add episode → search → verify chunk IDs
- Redis caching: set → get → TTL expiry

### R4: Contract Tests (Pact)
- Consumer-driven contracts for API endpoints
- Self-hosted Pact broker
- `can-i-deploy` gate in CI
- Initial contracts for: `/documents/upload`, `/search`, `/search/ask`, `/health`

### R5: Chaos Testing (Litmus)
- Redis-kill experiment: verify app degrades gracefully (uncached reads succeed)
- RabbitMQ-pause experiment: verify Celery tasks queue and resume
- DB-slow-query experiment: verify timeout handling and circuit breaker
- CPU-stress experiment: verify rate limiter and request queuing
- Weekly Monday 09:00 UTC schedule in `chaos-staging` environment

### R6: Coverage Gates
- Minimum 30% coverage on new code (enforced in CI)
- Minimum 50% coverage on Tier 1-3 reliability paths
- Coverage report as PR comment
- `--cov-fail-under=80` in pytest config (already set)

### R7: Test Data Management
- Fixtures return isolated data per test (no cross-test pollution)
- Database truncation between tests (not transaction rollback — pgvector needs it)
- Redis flush between tests
- Neo4j session clear between tests

## Acceptance Criteria
- [ ] `uv run pytest tests/integration/` passes with testcontainers
- [ ] `uv run pytest tests/contract/` passes with Pact broker
- [ ] Coverage report shows 30%+ on new code
- [ ] Factory-boy generates valid test data for all factories
- [ ] Chaos experiments run in staging without app crash

## Non-Goals
- Rewrite existing unit tests (keep them as-is)
- Add E2E browser tests (out of scope)
- Migrate from pytest to another framework
