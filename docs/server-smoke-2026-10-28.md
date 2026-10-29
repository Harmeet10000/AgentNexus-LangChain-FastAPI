# Server smoke test — 2026-10-28

Environment: native process (no docker — daemon requires interactive sudo), cloud
infra from `.env.development` (Timescale Cloud Postgres, Redis Cloud, Railway
RabbitMQ, MongoDB Atlas, Cloudflare R2, Neo4j cloud, cognee local).
Server: `uv run uvicorn app.main:app --port 5000`. Worker: `uv run worker --role ingestion`.
Commit under test: `17d936e` (PR #106 merge).

## Boot

- Startup completes in ~60–80 s (verbose ORM `_configure_property` logging + cloud
  handshakes). Port 5000.
- First boot: `Agent Saul unavailable; continuing without capability` — a transient
  cloud timeout during `_setup_saul_graph_state` degraded the whole capability.
  Second boot (identical env): `Agent Saul graph initialized`. **Finding F6: Saul
  boot has no retry — one transient dependency timeout silently removes the
  product's core capability for the process lifetime.**
- Deep health `/health` (unauthenticated): all seven dependencies healthy
  (postgres 1896 ms — first-query cloud latency; redis 55 ms; mongodb 43 ms;
  neo4j 294 ms; neo4j-plugins 819 ms; graphiti; cognee).
  `/api/v1/health/{live,ready,self}` all 200.

## Endpoint results

| Probe | Result |
|---|---|
| `POST /api/v1/auth/register` | 201; 422 with typed field errors for reserved-TLD email (validation works) |
| `POST /api/v1/auth/login` | 200; JWT `accessToken` (900 s), refresh token family, permissions in claims |
| `GET /api/v1/documents/{id}/status` | 200; truthful terminal state, but see F2 |
| `POST /api/v1/search` | 200; empty corpus → `items: []`; 20 s cold (embedding + cache-miss path) |
| `POST /api/v1/legal/ask` | 401 unauth; field is `query` (not `question`) |
| `POST /api/v1/agent-saul/sessions` | 422 requires ≥1 document (ingestion-readiness validation present) |
| Upload size limit (12 MB vs 10 MB) | 422 `FILE_TOO_LARGE` **before storage/parse** (bounded buffering verified) |
| Upload extension (.exe) | 422 `EXTENSION_NOT_ALLOWED` with allowlist echoed |
| Re-upload identical file | returns the SAME `doc_id` — content-hash idempotency works |
| OpenAPI `/openapi.json` | 404 (docs disabled in this env) |
| Global error envelope | typed `{success, statusCode, correlationId}` on every failure incl. 404s |

## Defects found (ranked)

- **F1 (P0 — fixed in this session, `app/worker/celery/app.py`): `uv run worker --role
  <role>` crashes on boot.** Celery's click path constructs `app.Worker(...)` without an
  `app=` kwarg; `Worker.app` defaults to None → `AttributeError: 'NoneType' object has no
  attribute 'loader'` in `CatalogValidatingWorker.__init__`. Deterministic via the console
  script; healthy when the app is built in-process. Fix: bind `_apps.active` when celery
  passes no app. No unit test existed that boots a worker via the CLI path — CI never
  exercises it.
- **F2 (P1): terminal ingestion failure carries no surfaced reason.** Status API returns
  `status: "failed"`, `chunk_count: 0`, `warnings: []` — no error field. The actual reason
  (`INGESTION_UNEXPECTED_FAILURE: "Document ingestion retry budget exhausted"`) is buried
  in `documents.metadata_.ingestion_error` in Postgres and nowhere else. This violates the
  bulletproof-ingestion "truthful terminal state" property: the DB knows, the API doesn't.
- **F3 (P1): the worker log contains zero task-receipt or failure lines.** The document
  went `received → retried → failed` with nothing in `/tmp/worker-smoke2.log` (only
  ETA/exchange warnings). Celery std-logging is swallowed by the loguru capture for task
  events; a failed ingest is invisible in the worker's own log.
- **F4 (P1): fresh signups cannot reach the product.** Registration creates no personal
  tenant; every tenant-scoped route (documents/search/saul) returns 403 "Tenant
  authorization is unavailable". Recovery is `scripts/migrate_personal_tenants.py`
  (default mode; `--reconcile` is mirror-sync only and scans 0). Either registration
  should provision the personal tenant, or the login response must tell the client to
  run onboarding.
- **F5 (P2): login says "Check your inbox" but no SMTP is configured** — a fresh
  environment user can never self-verify (`is_verified` stays false in Mongo);
  verification is the login gate. Dev needs a documented bypass (dev-only verify
  endpoint or seeded account).
- **F6 (P2): Saul boot has no retry on transient dependency failure** (see Boot above).
- **F7 (P2): `documents.status` exposes `document_version: null, pipeline_version: null`
  for a failed document** — version identity should be assigned at receive time, not on
  success, per the provenance design.
- **F8 (P3): search cold latency 20 s** on an empty corpus — embedding path dominates;
  fine for dev, worth a budget note (C4 of the ingestion plan).

## Verified working

Auth (register/login/JWT/claims), tenant resolution after backfill, bounded upload
validation ordering, content-hash idempotency, deep health with live latency probes,
typed error envelopes with correlation IDs, queue → worker → terminal status pipeline
(once F1 fixed), RabbitMQ/Timescale/Redis/Mongo/R2/Neo4j connectivity, graceful
degradation reporting, Saul graph compilation.

## Cleanup

Smoke user `smoke-test-20261028@example.com` (verified for testing) and one failed
document remain in the infra DBs. Ingestion root cause of the retry exhaustion was not
chased to its origin (worker logs carry no error — F3); first suspect is the
extraction/LLM step for a `.txt` file.
