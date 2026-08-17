# Security Review: langchain-fastapi-production

**Date**: 2026-08-15
**Reviewer**: Automated security review
**Scope**: Full codebase analysis

---

## Summary
- **Findings**: 2 (1 High, 1 Medium)
- **Risk Level**: High
- **Confidence**: High

---

## Findings

### [VULN-001] SSRF via Crawl Tool (High)
**Location**: 
- `src/app/shared/langchain_layer/agents/tools/crawl.py:41`
- `src/app/shared/crawler/crawler.py:199`
- `src/app/shared/crawler/validator.py`

**Confidence**: High

**Issue**: The `crawl_url` LangChain tool accepts a user-controlled `url` parameter. While `validate_url()` blocks private IPs, localhost, metadata endpoints, and blocked domains, the tool is registered in the global tool registry and exposed to LangChain agents. An attacker who can influence agent tool selection (via prompt injection or direct API access to agent endpoints) could trigger crawls to internal services, cloud metadata endpoints, or internal APIs.

**Impact**: Server-Side Request Forgery allowing access to:
- Internal services (databases, caches, internal APIs)
- Cloud metadata endpoints (AWS 169.254.169.254, GCP metadata.google.internal, Azure metadata)
- Kubernetes API server
- Internal HTTP services on non-standard ports

**Evidence**:
```python
# crawl.py:41 - User-controlled URL passed directly to crawler
async def _arun(self, url: str, ...) -> str:
    crawler = await get_crawler()
    result = await crawler.crawl(url=url)  # url from user/agent input
```

```python
# crawler.py:199 - Sanitization happens but validation is the only barrier
url = sanitize_url(url)
if not is_valid_url(url):
    return CrawlResult(success=False, error_message="Invalid or disallowed URL")
```

**Fix**: 
1. Add an allowlist of permitted domains for crawling (configurable via settings)
2. Require explicit user consent/confirmation for each crawl operation
3. Run crawler in a network-isolated environment (separate VPC/network namespace)
4. Consider removing the crawl tool from production agent configurations unless strictly needed

---

### [VULN-002] Missing Request Size Limits on File Upload (Medium)
**Location**: 
- `src/app/features/documents/router.py:28-46`
- `src/app/features/profile/router.py:101-116`

**Confidence**: High

**Issue**: File upload endpoints (`/documents/upload`, `/profile/avatar`) read entire file contents into memory with `await file.read()` without enforcing maximum file size limits. An attacker could upload arbitrarily large files causing memory exhaustion (DoS).

**Impact**: Denial of Service via memory exhaustion, potential storage cost abuse.

**Evidence**:
```python
# documents/router.py:34 - No size limit enforcement
raw_bytes = await file.read()
```

```python
# profile/router.py:110 - No size limit enforcement  
contents = await file.read()
```

**Fix**: Add `File(..., max_size=10*1024*1024)` or similar limit, and configure `uvicorn` with `limit_max_request_size` in `src/app/server.py`.

---

## Needs Verification

### [VERIFY-001] MCP Client Upstream Server Configuration
**Location**: `src/mcp_core/client/manager.py:214-227`

**Question**: Are `MCPClientServerConfig` URLs and commands sourced from server-controlled configuration only, or can they be influenced by user input? The `_build_client` method uses `config.url` and `config.command` directly.

---

### [VERIFY-002] Graphiti/Neo4j Query Parameterization
**Location**: `src/app/shared/rag/graphiti/subgraph.py:165-170`

**Question**: Verify that Cypher query parameters (`seed_uuids`, `group_ids`, `limit`) are properly parameterized and not string-interpolated. The code uses `session.run(cypher, seed_uuids=..., group_ids=...)` which appears parameterized, but confirm Neo4j driver behavior.

---

## Positive Security Observations

### Authentication & Authorization
- **Password Hashing**: Argon2id with configurable parameters, automatic rehash on parameter upgrades
- **JWT**: Short-lived access tokens, refresh token rotation, secure token storage (SHA-256 hashes only in DB)
- **OAuth**: State parameter verification (CSRF protection), PKCE support
- **Session Management**: Redis-backed with MongoDB audit trail, automatic TTL cleanup
- **Rate Limiting**: Applied to auth endpoints (login, register, verification), WebSocket connections

### Secrets Management
- **SecretStr**: Used throughout settings for all sensitive fields
- **Access Pattern**: `.get_secret_value()` enforced, no `str(settings.FIELD)` usage found
- **Production Validation**: Startup validation rejects default/insecure secret values in production
- **Environment Variables**: All secrets loaded from environment, no hardcoded secrets

### Database Security
- **MongoDB (Beanie)**: Parameterized queries via ODM, no raw query interpolation
- **PostgreSQL (SQLAlchemy)**: Parameterized queries via async ORM
- **Redis**: Connection pooling with authentication, no direct command interpolation

### Input Validation
- **Pydantic Models**: `extra="forbid"` on all request DTOs rejects unknown fields
- **Email Validation**: `EmailStr` type for all email inputs
- **File Upload**: Content-type validation, filename sanitization via S3 key generation
- **URL Validation**: SSRF protection in crawler with private IP/blocked domain blocking

### Transport Security
- **Cookies**: HttpOnly, Secure, SameSite=Lax on all auth cookies
- **CORS**: Configured via settings, credentials allowed only for specific origins
- **TLS**: Enforced in production via `secure=True` cookie flag

### Error Handling
- **Production**: Generic error messages, no stack traces leaked to clients
- **Development**: Detailed traces only in non-production environments
- **Global Handler**: Single exception handler with uniform error envelope

### Logging & Auditing
- **Structured Logging**: loguru with contextual binding
- **Audit Trail**: Token audit logs with TTL-based auto-purge
- **No Secrets in Logs**: SecretStr automatic redaction in `model_dump()`/`repr()`

---

## Recommendations

### Immediate (Before Production)
1. **Fix VULN-001**: Implement domain allowlist for crawler or remove crawl tool from production agents
2. **Fix VULN-002**: Add file size limits to upload endpoints and uvicorn config

### Short-term
3. Complete verification of VERIFY-001 and VERIFY-002
4. Add security headers middleware (CSP, HSTS, X-Frame-Options, etc.)
5. Implement request body size limits globally

### Ongoing
6. Regular dependency vulnerability scanning (Dependabot/Trivy)
7. Periodic penetration testing
8. Security training for team on prompt injection risks with LLM tools

---

## Appendix: Files Reviewed

### Core Security Files
- `src/app/config/settings.py` - Secrets management, production validation
- `src/app/features/auth/security.py` - Password hashing, token generation, JWT
- `src/app/features/auth/service.py` - Auth business logic
- `src/app/features/auth/repository.py` - User/session persistence
- `src/app/shared/crawler/validator.py` - SSRF protection
- `src/app/shared/langchain_layer/agents/tools/crawl.py` - Crawl tool
- `src/app/middleware/global_exception_handler.py` - Error handling
- `src/app/server.py` - Server configuration

### Agent & Tool Files
- `src/app/shared/langchain_layer/agents/tools/registry.py` - Tool registry
- `src/app/shared/langchain_layer/agents/tools/web_search.py` - Web search tool
- `src/app/shared/langchain_layer/agents/tools/get_obligation_chain.py` - Graphiti tool
- `src/app/shared/langgraph_layer/agent_saul/` - Legal agent implementation

### Infrastructure Files
- `src/app/connections/httpx_client.py` - HTTP client configuration
- `src/app/connections/neo4j.py` - Neo4j driver
- `src/app/connections/celery.py` - Celery configuration
- `src/mcp_core/client/manager.py` - MCP client manager

---

*Generated by automated security review. Manual verification recommended for all findings.*