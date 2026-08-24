# Smoke Test Report — 2026-08-24

App run against live infrastructure: Postgres (Timescale) ✅, MongoDB ✅,
Redis ✅, Neo4j (resumed mid-session) ✅, RabbitMQ ❌ (connection reset —
degrades with warning), S3/R2 ❌ (unconfigured — by design).

## Findings

### F1 — P0 — Startup blocked indefinitely on unreachable Neo4j — FIXED

**What happened.** With the Neo4j instance down, uvicorn printed
`Application starting`, logged `Neo4j startup failed, continuing without
graph features` — then hung in `Waiting for application startup` forever.
No endpoint ever answered (connection refused).

**Why it broke.** Two independent initialisers talk to Neo4j:

1. `setup_neo4j()` fails fast and correctly returns `None`.
2. `setup_graphiti()` is called **unconditionally afterwards**, even though it
   targets the same unreachable host. The neo4j driver's internal transaction
   retry loop (`Transaction failed and will be retried …`) backs off
   exponentially and effectively never gives up on a DNS failure, so the
   lifespan coroutine never returns and uvicorn never opens the socket.

The `except (ConnectionError, TimeoutError, OSError, ServiceUnavailable)`
around it cannot catch what never raises.

**Fix.** `src/app/lifecycle/lifespan.py`:

* Skip Graphiti entirely when `app.state.neo4j_driver is None` — same host,
  so initializing it is pointless.
* Belt: wrap the whole setup in `asyncio.wait_for(..., timeout=20)` and add
  `asyncio.TimeoutError` to the caught set.

**Verification.** With Neo4j still down the app now reaches
`Application startup complete` (~30 s); `/health` answers 200; Graphiti-dependent
routes answer 503 per the resilience contract.

### F2 — P1 — First-time document upload always returned 404 — FIXED

**What happened.** Uploading any new file produced
`404 {"code": "DOCUMENT_NOT_FOUND", "message": "Resource not found"}` instead of
queueing ingestion. Every upload was broken.

**Why it broke.** `upload_document` first checks for an existing copy via
`repo.get_document_by_user_hash(...)`. The repository signals *no row* as
`Failure(NotFoundAppError)` (its standard dual-method shape). The service
treated **any** `Failure` as fatal:

```python
elif isinstance(existing_result, Failure):
    log_expected_failure(...)
    raise app_error_to_exception(existing_result.failure())
```

So the normal first-upload path ("nothing exists yet") raised a 404 at the
caller.

**Fix.** `src/app/features/documents/service.py`: only non-`NotFoundAppError`
failures abort; a miss means "no duplicate — proceed".

**Verification.** Upload now passes the duplicate check and correctly reaches
the next gate: `503 Object storage is not configured` — which is the designed
answer while S3/R2 is unconfigured.

### F3 — Documented, not fixed — Access token survives logout (≤ 15 min)

Logout revokes the refresh token server-side, but access tokens are stateless
15-minute JWTs and `get_current_user` consults no revocation list, so
`/api/v1/auth/me` keeps answering 200 until expiry.

This is a coherent design (short-TTL stateless access tokens), consistent with
the RBAC guards' "claims-based — no DB hit" comment. If stricter semantics are
wanted: add a Redis jti denylist checked in `get_token_claims`
(O(1) per request; ~15 min TTL entries). Recorded here deliberately — auth
semantics should not change silently.

### Ops note — killing the server from this shell

`pkill -f "uvicorn app.main:app"` matches the *shell's own command line*
(which contains that literal) and kills the session before it can detach the
next process. Use the bracket idiom: `pkill -f "[u]vicorn app.main:app"`.
Start detached: `setsid nohup .venv/bin/uvicorn … & disown` — otherwise the
tool session's process-group teardown reaps uvicorn when the launching command
ends.

## Verified working (no action needed)

| Check | Result |
|---|---|
| `GET /health`, `/api/v1/health/`, `/api/v1/health/self`, `/` | 200 |
| `GET /swagger.json`, `/api-docs` | 200 |
| `POST /auth/register` | 201, envelope correct |
| Duplicate register | 409 |
| Login before email verification | 401 "Email not verified" |
| Wrong password | 401 |
| `GET /auth/me` without token | 401 |
| `GET /auth/me` bearer + cookie | 200 |
| `POST /auth/refresh` (`refresh_token`) | 200 |
| `POST /auth/logout` (`refresh_token`) | 200 |
| Rate limiting | 429 after ~5 bad logins |
| Error envelope shape | `{success, statusCode, request, message, data, error:{code,…}}` everywhere |
| Agent routes with unwired graph | **503** "Saul graph is not wired" (resilience spec scenario) |
| Upload without file | 422 validation |

## Environment notes

* RabbitMQ refused the connection (`Errno 104`) — Celery setup times out and
  the app continues without a task queue, as designed.
* Neo4j was resumed partway through the session; F1's fix additionally covers
  the down case permanently.

---

## Second pass — 2026-08-24 (later same day)

### F4 — P1 — `/health` returned 500 under `src.app.main:app` launches — FIXED

**What happened.** With the app launched as `uvicorn src.app.main:app`, `GET /health`
consistently returned 500: pydantic rejected the probe results with
`Input should be a valid dictionary or instance of DependencyHealth` even though
the probes *did* return `DependencyHealth` instances. The same code served 200
when launched as `app.main:app`.

**Why it broke.** Dual module identity. The launch string `src.app.main:app`
makes Python import every module twice under two names (`src.app.utils.response_type`
vs `app.utils.response_type`), because internal imports are absolute (`from app.…`).
`main.py` builds `HealthResponse` against one class; the middleware probes
returned instances of the other; pydantic's isinstance check failed.

**Fix.** `src/app/middleware/health_check.py` imports `DependencyHealth`
relatively (`from ..utils import …`) so probe instances always match the class
`main.py` validates against, whichever entrypoint spelling is used.
Verified: `/health` → 200 under **both** `app.main:app` and `src.app.main:app`.

### F5 — P2 — Postgres probe always reported `unknown/unhealthy` — FIXED

**What happened.** The deep-health dependency list showed postgres as
`unknown` + unhealthy on every request even while Postgres was connected and
serving traffic.

**Why it broke.** Two stacked bugs in `check_postgres`: it re-wrapped
`app.state.db_engine` (already an `AsyncEngine`) in another `AsyncEngine`,
which makes SQLAlchemy fall back to a sync driver and raise
*"required an async execution but none was detected"* (xd1r) on every probe;
and its `except (OSError, TimeoutError)` list did not include
`SQLAlchemyError`, so the error escaped as an unnamed failure instead of
`fail("postgres", …)`.

**Fix.** Use the engine directly; catch `SQLAlchemyError` too.
Verified live: all six dependencies report healthy.

### Startup-failure triage note

The user-reported startup exit (`PostgreSQL startup failed` at
`lifespan.py:181`) was caused by total DNS loss in that shell
(`gaierror: Temporary failure in name resolution` for Postgres, Redis and
LangSmith simultaneously). Postgres fail-fast is deliberate — it is the primary
store; Neo4j/Celery/Graphiti degrade to 503s instead. If graceful degradation
is preferred for Postgres too, that is a one-line policy change in lifespan.
