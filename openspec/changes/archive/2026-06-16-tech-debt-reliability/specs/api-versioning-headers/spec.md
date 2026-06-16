# Capability: api-versioning-headers

## Purpose
Add deprecation headers to v1 API responses, providing clients a clear migration path to v2.

## Requirements

### R1: Deprecation Middleware
- Location: `src/app/middleware/api_versioning.py`
- Check: `request.url.path.startswith("/api/v1/")`
- Inject headers on response:
  - `Deprecation: true`
  - `Sunset: Sat, 01 Jan 2027 00:00:00 GMT`
  - `Link: </api/v2/>; rel="successor-version"`
- Apply to all HTTP methods (GET, POST, PUT, DELETE, PATCH)

### R2: Configuration
- `API_SUNSET_DATE: str = Field(default="Sat, 01 Jan 2027 00:00:00 GMT")`
- `API_V2_BASE_PATH: str = Field(default="/api/v2")`
- Headers only added when `API_SUNSET_DATE` is not empty
- Disable headers by setting `API_SUNSET_DATE=""`

### R3: Middleware Registration
- Register in `main.py` after CORS middleware, before routes
- Must not conflict with existing Guard/SecurityMiddleware
- Use `app.add_middleware()` pattern (not class-based)

### R4: Response Header Order
- Headers added to response, not request
- Must not overwrite existing headers
- Must not conflict with CORS `Access-Control-Expose-Headers`
- Add `Sunset` and `Deprecation` to exposed headers list

### R5: Documentation
- OpenAPI spec should document deprecated status
- Swagger UI should show deprecation badge
- Add `deprecated: true` to v1 router in OpenAPI config

### R6: Exempt Paths
- `/health` — no deprecation headers (infrastructure endpoint)
- `/metrics` — no deprecation headers (monitoring endpoint)
- `/api-docs`, `/api-redoc`, `/swagger.json` — no deprecation headers

## Acceptance Criteria
- [ ] `curl -v http://localhost:5000/api/v1/documents/upload` shows `Deprecation: true` header
- [ ] `curl -v http://localhost:5000/health` does NOT show deprecation headers
- [ ] `Sunset` header value matches `API_SUNSET_DATE` setting
- [ ] `Link` header points to `/api/v2/`
- [ ] No duplicate CORS headers introduced
- [ ] `API_SUNSET_DATE=""` disables all deprecation headers

## Non-Goals
- Implement v2 API routes (only add headers to v1)
- Automatic route migration
- Client SDK generation for v2
- Deprecation logging/analytics
