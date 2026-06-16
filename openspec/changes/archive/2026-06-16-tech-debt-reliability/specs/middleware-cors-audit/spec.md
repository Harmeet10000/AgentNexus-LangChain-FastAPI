# Capability: middleware-cors-audit

## Purpose
Verify and fix potential duplicate CORS headers caused by Guard's `SecurityMiddleware.configure_cors()` and FastAPI's native CORS middleware.

## Requirements

### R1: Verification Test
- Manual `curl` test in CI (or pre-commit hook):
  ```bash
  curl -s -D - -H "Origin: https://example.com" http://localhost:5000/ | grep -i "access-control"
  ```
- Assert exactly ONE `Access-Control-Allow-Origin` header
- Assert exactly ONE `Access-Control-Allow-Methods` header
- If duplicates found, log warning with remediation steps

### R2: Middleware Audit
- Document current middleware execution order in `main.py`
- Verify `SecurityMiddleware.configure_cors()` doesn't add its own CORS headers
- If it does, disable Guard's CORS and use only FastAPI's native CORS
- If it doesn't, no code change needed — just document

### R3: CI Integration (Stretch Goal)
- Add CORS check to CI pipeline
- Run against dev server after startup
- Fail CI if duplicate headers detected
- Comment on PR with test results

### R4: Documentation
- Add middleware order diagram to ARCHITECTURE-RULES.md
- Document which middleware handles CORS
- Document Guard's role vs FastAPI's CORS middleware

## Acceptance Criteria
- [ ] `curl` test shows exactly one `Access-Control-Allow-Origin` header
- [ ] Middleware order documented in architecture docs
- [ ] No duplicate headers in any environment (dev, staging, prod)
- [ ] If Guard adds duplicate CORS: fix by disabling Guard's CORS helper
- [ ] If Guard doesn't add duplicates: document as verified

## Non-Goals
- Restructure middleware order (only verify/fix if broken)
- Add CORS testing framework (manual curl is sufficient)
- Implement CORS preflight caching
- Add CORS metrics to Prometheus
