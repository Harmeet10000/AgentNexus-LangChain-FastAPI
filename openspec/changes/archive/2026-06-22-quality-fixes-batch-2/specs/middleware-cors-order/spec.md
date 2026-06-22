# Middleware CORS Order Verification

## Scope

`src/app/main.py` — middleware registration block (lines 47-83)

## Problem

The middleware stack in `main.py` has two CORS-related mechanisms:

1. **`SecurityMiddleware.configure_cors(app, config)`** (line 61) — FastAPI Guard's built-in helper that adds an internal CORSMiddleware to the Starlette app.
2. **`SecurityMiddleware`** itself (line 77) — Guard's security middleware, which also sets security-related headers (including potentially CORS) in its `process_response` handler.

The comments in `main.py:47-57` describe the intended execution order as:

```
1. RequestStateLoggingMiddleware
2. SecurityMiddleware (Guard)
3. GZipMiddleware
4. ApiDeprecationMiddleware
5. CORSMiddleware
6. Route handler
```

But the *actual* middleware chain is more nuanced:
- `SecurityMiddleware.configure_cors()` adds CORSMiddleware **below** SecurityMiddleware in the ASGI chain (because it's added before SecurityMiddleware is registered)
- SecurityMiddleware itself sits at position 5 (counting from outermost)
- CORSMiddleware sits below SecurityMiddleware because it was added *before* SecurityMiddleware was added to the stack

Guard's `SecurityMiddleware` has internal dedup logic for CORS headers — it checks for `Access-Control-Allow-Origin` before adding its own. However, this is undocumented and could break if:
- Another CORS middleware is added manually (e.g., FastAPI's `CORSMiddleware` directly)
- Guard is upgraded and changes its CORS handling behaviour

## Solution

**No code change to middleware registration.** The current setup is correct. Guard manages CORS internally and deduplicates headers.

Changes:
1. **Update the comment block** in `main.py` to reflect reality more accurately — document that `SecurityMiddleware.configure_cors()` adds CORSMiddleware as a sub-middleware, and that Guard's SecurityMiddleware deduplicates CORS headers.
2. **Add a verification procedure** to the tasks section.

### Updated comment block

```python
# ============================================================================
# Add middlewares in REVERSE order of execution
# Last added = First executed
#
# Execution order (outermost → innermost):
#   1. RequestStateLoggingMiddleware — correlation ID, request state
#   2. SecurityMiddleware (Guard) — IP checks, rate limiting, pen-test detection, CORS headers
#   3. GZipMiddleware — response compression
#   4. ApiDeprecationMiddleware — Deprecation/Sunset headers on v1
#   5. CORSMiddleware (injected by SecurityMiddleware.configure_cors())
#      Guard's SecurityMiddleware deduplicates CORS headers internally.
#      Do NOT add another CORSMiddleware directly.
#   6. Route handler
# ============================================================================
```

## Verification

```bash
# Verify single CORS header
curl -H "Origin: http://example.com" -v http://localhost:8000/health 2>&1 | \
  grep -ci "access-control-allow-origin"

# Expected: 1 (deduplicated by Guard)
# If > 1: duplicate CORS headers detected — fix required
```

## Edge Cases

| Scenario | Expected |
|----------|----------|
| Request with `Origin` header | Exactly 1 `access-control-allow-origin` in response |
| Preflight (`OPTIONS`) request | Correct CORS headers, not blocked |
| No `Origin` header | No CORS headers added (Guard handles this) |
| Guard passive mode | CORS still works (configure_cors runs at startup regardless) |

## Verification

1. Run the curl command above, confirm single header
2. Run full test suite: `pytest tests/ -x` — no CORS-related regressions
