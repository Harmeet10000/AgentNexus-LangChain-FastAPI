# Server smoke test — 2026-11-15

Environment: native processes under user systemd (`systemd-run` transient units with
memory scopes; no docker). Cloud infra from `.env.development` (Timescale Postgres,
Redis Cloud, Railway RabbitMQ, MongoDB Atlas, Cloudflare R2, Neo4j cloud). Razorpay
test-mode keys present in the running API; `RAZORPAY_WEBHOOK_SECRET` added to
`.env.development` **after** first boot (missing from the running process until the
mid-run restart). ngrok tunnel `https://reverend-mousiness-comment.ngrok-free.dev` →
localhost:5000 up for Razorpay dashboard test-webhook delivery.

- Commits under test (see the provenance table below — this run spanned three
  boots): boot 1 (02:14:15) **failed**; boot 2 (~02:24, tester restart) served
  **`dd3abfe`** (a temp review branch checked out underneath the running
  service; `git log -1` at 02:21 had shown `f36bf5d`); boot 3 (02:41:33,
  orchestrator restart on main) served **`ef28bec`** (main) with
  `RAZORPAY_WEBHOOK_SECRET` loaded. Probes are labeled per boot; every critical
  finding was re-verified on main.
- API unit: `agentnexus-uvicorn` (MemoryMax=1800M, MemoryHigh=1500M), port 5000.
- Workers: `agentnexus-celery-default` (role default, concurrency 2, MemoryMax 2.3G /
  MemoryHigh 1.9G per unit properties) — running across all boots.
  `agentnexus-celery-ingestion` (role ingestion, concurrency 1, MemoryMax 1400M) —
  **repeatedly exits gracefully ~2 min after boot (finding F5)**; started four
  times during the run by tester and orchestrator. `agentnexus-celery-crawler` —
  **started by tester at 02:40:53** with the exact sanctioned systemd-run command
  (MemoryMax 1400M / MemoryHigh 1100M); remained active. No other worker started;
  concurrency never raised.
- OpenAPI: `/openapi.json` and `/swagger.json` both checked live; route inventory
  cross-checked against `src/app/api/v1/__init__.py`, `src/app/api/v2/__init__.py`
  and feature routers.

## Boot

- **First boot attempt failed**: `agentnexus-uvicorn` started 02:14:15, exited
  02:15:47 with `status=3/NOTIMPLEMENTED` after 1m32s (uvicorn exit code 3 =
  startup failure). The unit journal contains **zero application output lines** —
  the app's stdout/stderr never reached journald, so the traceback is unrecoverable
  from the unit (finding F3 below). `uv run python -c "import app.main"` succeeds
  standalone, so the failure was in startup/lifespan, not import.
- API restarted per the test plan (`systemctl --user restart agentnexus-uvicorn`);
  boot took ~6–7 min (verbose ORM `_configure_property` logging + cloud handshakes)
  before `/health` returned 200.
- `agentnexus-celery-ingestion` started 02:14:23 and **exited silently** (unit no
  longer listed, journal shows only the start line); re-started with the same
  command (`systemd-run --user --unit=agentnexus-celery-ingestion --property=MemoryMax=1400M
  --property=MemoryHigh=1100M bash -c 'cd /home/harmeet/Desktop/Projects/langchain-fastapi-production
  && exec make celery-ingestion'`).

## Boot provenance (read this first)

This smoke ran while parallel review agents were switching git branches underneath
the running service. The app's root route reports `git_sha: "unknown"`, so code
provenance must be reconstructed from tree state at boot time:

| Boot | Started | Outcome | Code served |
|---|---|---|---|
| 02:14:15 | systemd transient unit | **Failed** — uvicorn exit 3 at 02:15:47, no traceback captured anywhere (journal has zero app output for that invocation) | unknown (mid-switch tree) |
| ~02:24:00 | `systemctl --user restart agentnexus-uvicorn` (tester, sanctioned) | Healthy, serving all probes | **`dd3abfe`** (temp review branch checked out at restart; 15 commits behind main `ef28bec` — lacks rerank flag + D-3 quota work). `git log -1` at 02:21 showed `f36bf5d`; the tree was switched again to `greptile-chunk-bc79fab` by ~02:33 while the process kept serving dd3abfe |
| (pending) | orchestrator-only clean restart on main with webhook secret | — | `ef28bec` (main) — after-signal re-verification recorded at the end |

All probes below were therefore executed against **dd3abfe** unless explicitly
marked `[main re-check]`. The stale-boot episode itself is finding **F3** below.

### Probes executed so far (dd3abfe process)

Evidence format: `curl` status + typed envelope. Code: `error.code` from
`{"success":false,"statusCode":N,"error":{"code":...}}`.

| Method + path | Status | Observed behavior | Required backend piece |
|---|---|---|---|
| GET / | 200 | envelope `{success,statusCode,request.correlationId,data:{status:"healthy",version,git_sha:"unknown",build_date:"unknown"}}` | none — in-process |
| GET /health (unauth) | 200 | all 7 deps healthy: postgres 1886ms, redis 167ms, mongodb 167ms, neo4j 167ms, neo4j-plugins 263ms, graphiti 0.002ms, cognee 161ms | all cloud deps |
| GET /api/v2/health/live | 200 | server info envelope | none — in-process |
| GET /api/v2/health/ready | 200 | `status:"healthy"`, uptime/memory/cpu system block | none — in-process |
| GET /api/v2/health/self | 200 | server identity block | none — in-process |
| GET /metrics | 200 | `text/plain; version=1.0.0` Prometheus text | none — in-process |
| GET /openapi.json | 404 | docs disabled (prior smoke consistent) | none |
| GET /swagger.json | 200 | 138 operations across v1+v2 | none — in-process |
| POST /api/v1/health/live (v1 compat header check) | 200 | `deprecation: true`, `sunset: 2027-06-15`, `link: </api/v2/health/live>; rel="successor-version"` | none — in-process |
| POST /api/v2/auth/register (fresh) | 201 | `isVerified:false`, message "Check your email"; personal tenant `personal:6af8cacc…` auto-created in `app_tenant_memberships` (DB check) — **F4 from 10-28 verified fixed** | none — in-process |
| POST /api/v2/auth/register (duplicate) | 409 | `error.code=USER_CONFLICT` "Email already registered" | none — in-process |
| POST /api/v2/auth/register (weak pw) | 422 | `VALIDATION_ERROR` with field-level `body → password` detail | none — in-process |
| POST /api/v2/auth/login (principal) | 429 then 200 | **429 on first login of the window after 3 registers** — shared unscoped GCRA bucket (F6); 429 body typed `TOO_MANY_REQUESTS` | Redis Cloud |
| POST /api/v2/auth/login (fresh unverified) | 401 | `INVALID_CREDENTIALS` "Email not verified. Check your inbox." — F7 gate still live | SMTP (absent → blocked self-verify) |
| GET /api/v2/auth/me | 200 | same user id 6af208ec…, role user, isVerified true | none — in-process |
| GET /api/v2/auth/me (no token) | 401 | `UNAUTHORIZED` typed envelope | none |
| GET /api/v2/auth/sessions | 200 | `data: []` — empty list even though a session was just created (JWT carries sid) — see F9 note | Mongo |
| POST /api/v2/auth/refresh | 200 | new access+refresh issued | none |
| POST /api/v2/auth/refresh (REUSED old token) | 200 | **old refresh token still accepted after rotation — no reuse detection (F2)** | none |
| POST /api/v2/auth/refresh (garbage) | 401 | `UNAUTHORIZED` | none |
| POST /api/v2/auth/resend-verification | 200 | canned "If that email is registered…" (no SMTP configured; unverifiable) | SMTP |
| POST /api/v2/auth/forgot-password | 200 | canned message, same caveat | SMTP |
| GET/PATCH /api/v2/profile/ | 200/200 | PATCH with `full_name` persists (fresh read confirms); `extra=forbid` 422 on unknown field; response DTO camelCase | none — in-process |
| POST /api/v2/profile/change-password (wrong current pw) | 401 | `CURRENT_CREDENTIAL_INVALID` | none — in-process |
| POST /api/v2/profile/change-password (missing field) | 422 | field-level `body → new_password` "Field required" | none |
| GET /api/v2/users/ (role=user) | 403 | `FORBIDDEN` "Missing required permission: users:read" | none |
| GET /api/v2/users/{id} (role=user) | 403 | `FORBIDDEN` users:read | none |
| PATCH /api/v2/tenants/{tid}/members/{uid}/legal-roles (analyst, own tenant) | 403 | `FORBIDDEN` "Corpus-admin authority is required" | none |
| PATCH /api/v2/tenants/{tid}/settings/self-review (analyst, own tenant) | 403 | `FORBIDDEN` "Corpus-admin authority is required" | none |
| PATCH /api/v2/tenants/{other-tid}/settings/self-review | 403 | distinct message "Tenant membership does not grant access to this tenant" (no cross-tenant leak) | none |
| POST /api/v2/documents/upload (.txt) | 201 | `doc_id c268e6d5-a0d5-47c2-b499-89fb19b20cb7`, `status:queued`, `task_id:null` (outbox/JobBus path; "Celery task published tasks.documents_ingest routing_key=ingestion" in journal), object_uri s3://agent-saul-1/... | celery ingestion worker |
| POST /api/v2/documents/upload (identical bytes) | 201 | `duplicate:true`, SAME doc_id, `status:received` — content-hash idempotency intact | none — in-process |
| POST /api/v2/documents/upload (.exe) | 503→429 | first attempt: typed **503 DATABASE_ERROR "Session store unavailable"** during a Redis Cloud timeout (journal: "Connection was lost or not established", "Timeout reading from redis-15363…redislabs.com:15363") — extension check never ran; retry consumed by upload rate limit → 429 `TOO_MANY_REQUESTS` **with `retry-after: 47` header**. Extension rejection still unverified (see Pending) | Redis Cloud + upload limiter |
| GET /api/v2/documents/{doc_id}/status | 200 | truthful `received`, title/kind/chunk_count 0/document_version **null** (F7 from 10-28 still present: version identity not assigned at receive time) | none — in-process |
| GET /api/v2/documents/{unknown-uuid}/status | 404 | `STATUS_NOT_FOUND` "Status not found for the given document" | none |
| GET /api/v2/documents (list) | — | **no list endpoint exists** in the current router (confirmed via swagger inventory) | — |
| DELETE /api/v2/documents/{id} | — | **no delete endpoint exists** in the current router | — |

### Search stage (dd3abfe process)

Prerequisite caveat: the ingestion worker was dead/crashing (F5 below), so the smoke
document never ingested — the corpus is empty. Empty results are therefore expected
for retrieval probes and recorded as BLOCKED-prerequisite rather than FAIL, EXCEPT
the `/search/rag` 500 which is a code defect independent of corpus state.

| Method + path | Status | Observed behavior | Required backend piece |
|---|---|---|---|
| POST /api/v2/search (scoped to doc) | 200 | `data:{items:[],cache_hit:false}` — typed empty result | embedding provider + Postgres |
| POST /api/v2/search (unauth) | 401 | `UNAUTHORIZED` | none |
| POST /api/v2/search (empty query) | 422 | `VALIDATION_ERROR` field-level `body → query` | none |
| POST /api/v2/search/rag | **500** | **deterministic 500 on EVERY payload** — service layer re-validates `UnifiedSearchRequest.model_validate(payload.model_dump())` while the endpoint DTO `UnifiedRagRequest` carries `max_tokens` (default 2000) and the strict DTO forbids extra fields → `pydantic ValidationError: max_tokens — Extra inputs are not permitted` bubbles unhandled. Root exception visible in evidence below. **Defect F1.** | none — in-process |
| POST /api/v2/search/ask | 200 | truthful grounded-answer decline: `answer:"I do not have enough grounded document context…"`, `citations:[]`, `confidence:"uncertain"`, evidence policy present | model provider |
| POST /api/v2/legal/ask (field is `query`) | 200 | same typed decline shape as ask | model provider |
| POST /api/v2/search/hybrid | 200 | `items:[]`, typed empty | Postgres + embedding |
| POST /api/v2/search/ingest, GET /api/v2/search/ingest/{task_id} | — | Blocked: distinct search-ingest pipeline requires a healthy worker; ingestion worker down (F5) | celery worker |

RAG 500 evidence (two payloads — with and without client-supplied `max_tokens` —
both 500 with the SAME injected default `input_value=2000`):

```bash
curl -s -H "Authorization: Bearer $ACCESS" -H 'Content-Type: application/json' \
  -X POST http://127.0.0.1:5000/api/v2/search/rag \
  -d '{"query":"termination notice","max_tokens":2000}'
# -> 500, error.code=INTERNAL_SERVER_ERROR, AND a full Python traceback serialized
#    into error.trace (starlette middleware frames + endpoint frame at
#    src/app/features/documents/api/http/router.py:153 unified_rag) — traceback
#    disclosure in the HTTP response; request block of the envelope is
#    {"ip":null,"method":null,"url":null,"correlationId":null}
# root of trace:
# pydantic_core._pydantic_core.ValidationError: 1 validation error for UnifiedSearchRequest
#   max_tokens
#     Extra inputs are not permitted [type=extra_forbidden, input_value=2000, input_type=int]
```

### Curl evidence (key probes)

```bash
# 1. Deep health (all seven deps)
curl -s http://127.0.0.1:5000/health | jq '.data.dependencies'

# 2. Register fresh user -> 201 + personal tenant auto-created
curl -s -H 'Content-Type: application/json' -X POST \
  http://127.0.0.1:5000/api/v2/auth/register \
  -d '{"email":"smoke-20261115@example.com","password":"Ssmoke!2026x"}'

# 3. Reused refresh token STILL ACCEPTED (defect F4):
curl -s -H 'Content-Type: application/json' -X POST \
  http://127.0.0.1:5000/api/v2/auth/refresh \
  -d '{"refresh_token":"<old-refresh-token>"}'   # -> 200, second time too

# 4. Content-hash idempotent re-upload -> same doc_id, duplicate:true
curl -s -H "Authorization: Bearer $ACCESS" \
  -F "file=@/tmp/smoke-fixtures/contract-smoke-20261115-dup.txt;type=text/plain" \
  http://127.0.0.1:5000/api/v2/documents/upload

# 5. Transient Redis outage -> typed 503 (correlationId IMMtZrdC0pow8MTxS2i0Y)
curl -s -H "Authorization: Bearer $ACCESS" \
  -F "file=@/tmp/smoke-fixtures/bad.exe" \
  http://127.0.0.1:5000/api/v2/documents/upload
# {"success":false,"statusCode":503,...,"error":{"code":"DATABASE_ERROR","message":"Session store unavailable",...}}
# journal: WARNING "Timeout reading from redis-15363.crce263.ap-south-1-1.ec2.cloud.redislabs.com:15363"

# 6. Upload rate limit carries Retry-After
curl -s -D - -o /dev/null -H "Authorization: Bearer $ACCESS" \
  -F "file=@/tmp/smoke-fixtures/bad.exe" \
  http://127.0.0.1:5000/api/v2/documents/upload   # 429 + retry-after: 47

# 7. Cross-tenant denial is a distinct typed message
curl -s -X PATCH -H "Authorization: Bearer $ACCESS" -H 'Content-Type: application/json' \
  http://127.0.0.1:5000/api/v2/tenants/personal:6af8cacc72fa05b3b483fd1d/settings/self-review \
  -d '{"selfReviewAllowed":false}'
# 403 FORBIDDEN "Tenant membership does not grant access to this tenant"
```

### Crawler stage (dd3abfe process)

| Method + path | Status | Observed behavior | Required backend piece |
|---|---|---|---|
| GET /api/v2/crawler/rate-limit | 200 | `{"remaining_minute":10,"remaining_hour":100}` | Redis |
| GET /api/v2/crawler/search?query=… | 200 | Tavily web search live; `data{answer,query,results,success,total_results}` with real result URLs | Tavily key |
| GET /api/v2/crawler/search?q=… | 422 | param is `query`, not `q` — typed `VALIDATION_ERROR` field-required | none |
| POST /api/v2/crawler/crawl | 202 | `crawl_id 11a55f124b304e91b66ed11f195b91f9`, `status:pending`; journal: `Celery task published tasks.crawler_crawl task_id=77584c35… routing_key='crawler'` | RabbitMQ + crawler worker |
| GET /api/v2/crawler/crawl/{id} | 200 | truthful `pending` while unconsumed | Redis job store |
| GET /api/v2/crawler/crawl/{id}/pages | 503 | typed `SERVICE_UNAVAILABLE` "Crawler result is not available yet" (fail-closed, correct envelope) | crawler worker |
| GET /api/v2/crawler/crawl/{id}/chunks (+/chunks/search) | 503 | same typed 503 fail-closed | crawler worker |
| GET /api/v2/crawler/crawl/{unknown} | 404 | `NOT_FOUND` "Crawl job with ID '…' not found" | Redis |
| POST /api/v2/crawler/crawl/{id}/cancel | — | not exercised: crawl never reached running state (see F8) | crawler worker |

**Crawler worker WAS started by tester** at 02:40:53 with the sanctioned command
(`systemd-run --user --unit=agentnexus-celery-crawler --property=MemoryMax=1400M
--property=MemoryHigh=1100M … make celery-crawler`). It booted cleanly
(`celery@Aging-Devil`, queue `crawler`, task `tasks.crawler_crawl` registered) even
though the ingestion worker's import chain is broken on the same tree — the crawler
role does not import `app.features.search.application.ingestion`.

**F8: crawl fails deterministically once consumed.** The task was published to
the default exchange with `routing_key='crawler'` at 02:40:22 (BEFORE the worker
started consuming at ~02:41:06); the crawl stayed `pending` for ~4.5 min with no
task-receipt lines (worker task logging invisible — 10-28 F3 persists), then was
consumed and reached terminal `failed` in 226 ms with zero pages attempted:
`error_message: "Capability owner synchronous operations must run at a task
boundary; use an adapter's async in-loop operation while the owner loop is
running"`. The terminal state IS truthful with a surfaced reason (10-28 F2's
"reasonless failure" complaint does NOT apply here), but the happy path is
unreachable. NOTE: this worker instance had been booted from the inconsistent
`bc79fab` tree at 02:40:53; the failure was not re-probed on a main-booted
crawler worker (single-worker budget).

### Billing stage (dd3abfe process; Razorpay test-mode keys loaded;
probes near the 02:41:33 restart boundary were re-run in the Main
re-verification pass below)

| Method + path | Status | Observed behavior | Required backend piece |
|---|---|---|---|
| GET /api/v2/plans | 200 | empty catalog `data:[]` — no plans seeded in this environment | Postgres |
| GET /api/v2/plans/{unknown} | 404 | typed `PLAN_NOT_FOUND` | Postgres |
| POST /api/v2/plans (role=user) | 403 | `FORBIDDEN` "Insufficient role for this operation" — admin-only, correct | none |
| GET /api/v2/subscriptions | 200 | paginated `{items:[],total:0,limit:50,offset:0}` | Postgres |
| GET /api/v2/invoices | 200 | plain array `data:[]` (note: NOT the paginated envelope subscriptions uses — envelope-shape inconsistency) | Postgres |
| POST /api/v2/subscriptions (real plan) | — | **Blocked**: plan catalog empty and plan creation is admin-only; no admin identity available in this run, so no Razorpay customer/subscription/order creation could be exercised ("verify one real order creation" is Blocked, not failed) | admin identity + Razorpay keys |
| POST /api/v2/subscriptions (unknown plan) | 404 | typed `PLAN_NOT_FOUND` before any provider call | Postgres |
| POST /api/v2/subscriptions/{unknown}/cancel | 422 | `VALIDATION_ERROR` (unknown id rejected at validation) | none |
| GET /api/v2/credits/balance/{self} | 200 | `{totalBalance:0,totalBalanceRupees:0.0,currency:"INR"}` — self-access allowed by design | Postgres |
| GET /api/v2/credits/balance/{other-user} | 403 | typed `CREDIT_FORBIDDEN` "Cannot view other users' credit balances" | none |
| GET /api/v2/credits/history/{self} | 200 | history envelope (5 entries) | Postgres |
| POST /api/v2/credits (grant, role=user) | 403 | `FORBIDDEN` — admin-only, correct | none |
| GET /api/v2/admin/billing/dunning (role=user) | 403 | `FORBIDDEN` | none |
| GET/POST /api/v2/payments/{id}(/refund), subscription payments | — | not exercised: no payment exists (needs a real order — Blocked, same prerequisite) | Razorpay keys + order |
| POST /api/v2/webhooks/razorpay (no signature header) | 401 | typed `WEBHOOK_VERIFICATION_FAILED` "Missing X-Razorpay-Signature header" | none |
| POST /api/v2/webhooks/razorpay (invalid signature; running process has NO webhook secret yet) | 401 | typed `WEBHOOK_VERIFICATION_FAILED` "Signature mismatch" — fail-closed even with no secret configured (correct) | none |


## Main re-verification pass (clean boot on main @ ef28bec, orchestrator restart 02:41:33)

Deep `/health` immediately after the main boot returned **503** with
`neo4j`, `neo4j-plugins`, `cognee` unhealthy (postgres/redis/mongodb/graphiti
healthy) — first-query cloud latency after cold start; auth still worked during
this window. Typed degradation envelope observed (`statusCode:503` with per-dep
status list).

| Re-check | Result on main (ef28bec) | Stands? |
|---|---|---|
| R1 refresh-token reuse | refresh 200 → re-POST the OLD token → **200 again**; no rotation/family-reuse detection | YES |
| R2 verification gate | unverified fresh-user login → 401 `INVALID_CREDENTIALS` "Email not verified. Check your inbox." | YES |
| R3 shared GCRA bucket | register ×3 (all 201) → first login → **429 `TOO_MANY_REQUESTS`** (login's own 5/60s quota never used) | YES |
| RAG `max_tokens` 500 | POST /api/v2/search/rag (no client max_tokens) → **500**, same `UnifiedSearchRequest max_tokens Extra inputs are not permitted, input_value=2000` root, traceback still serialized into `error.trace` | YES |
| Legacy upload 500 | POST /api/v2/ingestion/documents/upload → **500** after persisting the document (ResponseValidationError: `entity_count/clause_count/relationship_count/dropped_entity_count` missing from `DocumentUploadResponse`); deprecation/sunset headers unobservable because serialization dies | YES |
| Upload extension | `.exe` → 422 `INVALID_DOCUMENT` with `data.validation_code=EXTENSION_NOT_ALLOWED`, `extension:"exe"`, allowlist echoed `["docx","html","md","pdf","pptx","txt","xlsx"]` | pass |
| Upload size policy | 12 MB `.txt` → **201 queued** — CORRECT per current policy: `.env.development` sets `MAX_UPLOAD_SIZE=52428800` (50 MB); the 10 MB limit from the 10-28 smoke was a config value, not a constant. A >50 MB `FILE_TOO_LARGE` probe was not in the upload rate-limit budget (3/300s) — NOT RUN | pass / not-run |
| v1 compat headers | `GET /api/v1/documents/{id}/status` → 200 + `deprecation: true`, `sunset: 2027-06-15`, `link: </api/v2/documents/{id}/status>; rel="successor-version"` | pass |
| v1 compat routes | `POST /api/v1/search` 200 (items:[]), `POST /api/v1/legal/ask` 200 (typed decline), `GET /api/v1/profile/` 200, `GET /api/v1/auth/me` 200, `GET /api/v1/crawler/rate-limit` 200 — all v1 dual-mounted routes behave like v2 with deprecation headers | pass |
| Access-token TTL | 401s observed on all protected routes after ~15 min; refresh restores access — JWT `expiresIn` 900 s matches docs | pass |

### Webhook final state (secret live on main process)

| Probe | Status | Observed |
|---|---|---|
| Invalid signature | 401 | typed `WEBHOOK_VERIFICATION_FAILED` "Signature mismatch" |
| **Valid signature** (HMAC-SHA256 of body with the registered secret, computed locally) | **200** | `{"data":{"accepted":true}}` — full valid-signature path verified |
| Duplicate valid delivery | 200 | `accepted:true` again — idempotent acceptance, no error |
| Razorpay dashboard test deliveries | 401 ×2 | journal request lines `POST /api/v2/webhooks/razorpay HTTP/1.1" 401` at 02:45:35 (two deliveries ~150 ms apart, source 127.0.0.1 via ngrok) — **dashboard deliveries were RECEIVED but rejected by signature validation**, while the locally-computed HMAC with the same secret verifies 200. Pending-user: confirm the dashboard webhook's secret field matches `RAZORPAY_WEBHOOK_SECRET` (or that test deliveries were sent before registration completed) |

### Ingestion worker (main) — still not staying up

Orchestrator-recreated ingestion unit (02:45:53, main code, healthy import) died
AGAIN at ~02:48:06 with the same graceful, errorless exit: last two log lines are
"Document ingestion worker had no resources to close / resources closed" — i.e. it
shut down before ever provisioning document-worker resources and never logged a
task receipt. Unit then GC'd ("could not be found"). Restarted by tester 02:52
(same transient-unit command) and the queued document c268e6d5 polled again — see
poll trace below. Root cause of the ~2-minute graceful exit is not attributable
from logs (no signal source logged; possible external stop from concurrent agents).


## Ingestion completion (final poll, main code)

The smoke document `c268e6d5-a0d5-47c2-b499-89fb19b20cb7` (task
`c96f4dd3-48a3-427d-8459-8d720fc17c26`, published 02:30:50) remained at
`status: received` through repeated restarts of the ingestion worker (see F5).
The RAG/ask probes above therefore ran against an empty index; ask/legal-ask
correctly disclose the absence of grounded evidence, which is the designed
empty-corpus behavior, not a failure.

## Defects found (ranked)

- **F1 (P0): `POST /api/v2/search/rag` deterministically 500s on EVERY payload.**
  `DocumentsQueryService.rag` re-validates the request as the STRICT
  `UnifiedSearchRequest` (`extra=forbid`), but the endpoint DTO
  `UnifiedRagRequest` inherits and always carries `max_tokens` (default 2000) →
  `ValidationError: max_tokens — Extra inputs are not permitted` bubbles
  unhandled. Confirmed identical on dd3abfe AND main ef28bec, with and without a
  client-supplied `max_tokens`. The endpoint cannot ever succeed. Secondary
  effects: the full Python traceback (starlette + app frames incl.
  `src/app/features/documents/api/http/router.py:153`) is serialized into
  `error.trace` of the HTTP response (internal-path disclosure), and the
  envelope's `request` block is all-null (no correlationId). Evidence in the
  Search stage section.

- **F2 (P1): `POST /api/v2/ingestion/documents/upload` (deprecated compat route)
  returns 500 AFTER successfully persisting the document.** FastAPI
  `ResponseValidationError` — the route's declared response model demands legacy
  fields `entity_count`, `clause_count`, `relationship_count`,
  `dropped_entity_count` that `DocumentUploadResponse` no longer emits
  (document `420ae89d-…` WAS created and object written to R2). A client sees
  "failure" while state was mutated; the promised `Deprecation`/`Sunset`/
  `Link` headers are unreachable because serialization dies. Reproduced on main.

- **F3 (P1, process): running services are poisoned by uncoordinated branch
  switching.** Boot timeline: 02:14:15 boot **failed** (uvicorn exit 3, zero
  app output captured); the 02:24 restart served `dd3abfe` (a temp review
  branch, 15 commits behind main) while the tester believed it was HEAD
  `f36bf5d`; the tree was then switched to `greptile-chunk-bc79fab` mid-run
  (verified via `git log`/`git branch --show-current`), and an ingestion worker
  started from that inconsistent tree crashed at import
  (`ImportError: cannot import name 'EmbeddingQuotaExhaustedError' from
  'app.ai.embeddings'`). The app makes this undetectable from outside:
  `GET /` reports `git_sha: "unknown"`, `build_date: "unknown"` — no build
  provenance is surfaced by the API, health endpoints, or logs.

- **F4 (P1): refresh-token rotation has no reuse detection.** After a successful
  refresh, re-presenting the OLD refresh token returns a fresh 200 token pair
  (twice reproduced: dd3abfe and main). A stolen/leaked refresh token is never
  invalidated by rotation; only explicit logout/revocation stops it.

- **F5 (P2): the ingestion worker exits gracefully ~2 min after boot and queued
  documents silently stick at `received` forever.** Reproduced on four separate
  boots (dd3abfe tree, main tree, two testers): unit dies with only
  "Document ingestion worker had no resources to close / resources closed"
  (graceful Celery shutdown, no exception, no signal source logged, no
  task-receipt lines — 10-28 finding F3 persists: worker task logging is
  invisible). Document `c268e6d5` (task published 02:30:50) never reached a
  terminal state across ~30 min of polling, and `GET /documents/{id}/status`
  shows `status: received` with no error/warning — the API cannot distinguish
  "queued, worker down" from "queued, will run". Contributing config defects
  observed at worker boot: (a) **all worker roles share the node name
  `celery@Aging-Devil`** — every boot logs kombu pidbox "A node named
  celery@Aging-Devil is already used … Maybe you forgot to shutdown the other
  node" (no per-role node naming in `app/worker/celery/app.py`), and (b) jobs are
  published with an ETA onto a **direct exchange where Celery's native delayed
  delivery is unsupported** ("Exchange tasks is a direct exchange and native
  delayed delivery do not support direct exchanges"; "ETA tasks … will block the
  worker until the ETA arrives" — the stuck ingest task was re-delivered as a
  blocked ETA message at every worker boot).

- **F6 (P2): unscoped GCRA buckets are shared across register and login per IP,
  so the documented per-endpoint quotas are not real.** `get_rate_limiter`
  builds keys `rl:gcra:{ip}` with `scope=None` for both `limit_register` (3/60s)
  and `limit_login` (5/60s) — register consumption depletes the shared bucket
  and login gets throttled before its own quota is touched (429 on the FIRST
  login after 3 registers). Reproduced on main. Same root cause presumably
  applies to any other unscoped limiter (e.g. `limit_document_qa`).

- **F7 (P2): fresh signups still cannot reach the product (10-28 F5 persists).**
  Registration succeeds and now auto-creates the personal tenant (10-28 F4 is
  FIXED — `app_tenant_memberships` row `personal:6af8cacc…` exists for the fresh
  user), but no SMTP is configured, so `isVerified` stays false and login is
  gated (`401 "Email not verified. Check your inbox."`). resend-verification and
  forgot-password return canned 200s ("link has been sent") that deliver
  nothing. Dev needs a documented bypass.

- **F8 (P2): crawler jobs fail deterministically at the worker task boundary.**
  `POST /api/v2/crawler/crawl` → 202 → task `77584c35…` consumed → terminal
  `failed` after 226 ms with `error_message: "Capability owner synchronous
  operations must run at a task boundary; use an adapter's async in-loop
  operation while the owner loop is running"`; zero pages attempted. The
  terminal state IS truthful with a surfaced reason (10-28 F2's complaint does
  not apply here), but the happy path is unreachable. Worker task logs were
  again invisible (no received/succeeded lines).

- **F9 (P2): `GET /api/v2/auth/sessions` always returns `[]`.** A live session
  exists (the JWT carries a `sid`; login/refresh both mint sessions) yet the
  list is empty at every point in the flow, so users cannot see or target
  sessions for revocation (the DELETE /sessions/{id} routes are effectively
  unusable without an external source of session ids).

- **F10 (P3, 10-28 F7 persists): `documents.status` exposes
  `document_version: null, pipeline_version: null` at receive time** — version
  identity should be assigned at receive per the provenance design.

- **F11 (P3): envelope inconsistency — `GET /api/v2/invoices` returns a bare
  array while `GET /api/v2/subscriptions` returns the paginated
  `{items,total,limit,offset}` envelope** for sibling billing list resources.

- **F12 (P3, resilience/observability): (a) cold-boot `/health` returned 503**
  with neo4j/neo4j-plugins/cognee unhealthy (first-query cloud latency) while
  serving traffic; (b) a transient Redis Cloud timeout turned an upload into a
  typed 503 `DATABASE_ERROR` "Session store unavailable" (fail-closed, correct
  envelope, but an infra blip on one dependency fails the whole write path);
  (c) the API's stdout never reached the systemd journal for the 02:14 failed
  boot, making that startup traceback unrecoverable (log-sink config should
  attach a file sink for systemd-run deployments).

## Blocked / excluded

- **Excluded per owner instruction:** the entire agent_saul router (all
  `/api/v*/agent-saul*` routes).
- **Excluded per owner instruction / genuine absence:** OAuth/SSO authorize,
  callback, link, and confirm routes (external providers absent).
- **BLOCKED — subscription/order creation:** plan catalog is empty
  (`GET /api/v2/plans → []`) and plan creation is admin-only; no admin identity
  exists in this run. Without a plan, `POST /api/v2/subscriptions` cannot reach
  Razorpay, so payment-order/checkout creation, payments/refunds, invoice
  lifecycle, dunning run, and webhook admin replay were not exercised. Razorpay
  keys ARE present and the webhook HMAC path was verified directly.
- **BLOCKED — document ingestion completion & RAG evidence:** ingestion worker
  would not stay up (F5); the smoke document never ingested; search/rag/ask ran
  against an empty index (empty results ARE typed and truthful).
- **NOT RUN:** >50 MB `FILE_TOO_LARGE` upload probe (upload rate-limit budget of
  3/300s was exhausted by the extension/dedup/legacy probes), crawl cancel
  (crawl never reached running), impersonation/user lifecycle mutations (no
  admin token), password reset completion (no inbox), avatar upload.
- **Pending-user:** Razorpay dashboard test-webhook deliveries arrive at the API
  but are rejected 401 while a locally-computed HMAC with the registered secret
  verifies 200 — dashboard secret field vs `RAZORPAY_WEBHOOK_SECRET` mismatch to
  be confirmed by the owner.

## Cleanup

Test data left in infra: users `smoke-20261115@example.com`,
`smoke-r3a/b/c@example.com` (unverified); documents `c268e6d5…` (stuck received),
`2bee803b…` (12 MB, queued), `420ae89d…` (created via broken legacy route);
crawl job `11a55f12…` (failed). Principal profile `fullName` was changed to
"Smoke Tester 2026-11-15". Tokens were rotated; final logout/session-revocation
probes were intentionally left until last (below). No commits, no branch
switches, no source or `.env.development` modifications were made by the tester.

### Session lifecycle cleanup probes (main)

| Probe | Status | Observed |
|---|---|---|
| DELETE /api/v2/auth/sessions/{current-sid} (sid extracted from the JWT) | 200 | session revoked |
| GET /api/v2/auth/me with the revoked token | 401 | `UNAUTHORIZED` — revocation propagates immediately (pull check works even with the websocket revocation stream degraded) |
| GET /api/v2/auth/sessions (before revocation) | 200 | `[]` — see F9 |

## Result counts

One row per distinct HTTP operation probed (dual-mounted v1/v2 pairs counted
once per method actually probed on each version; Agent Saul excluded).

| Metric | Count |
|---|---|
| Distinct endpoints probed (both passes, incl. v1 compat) | 45 |
| PASS (prerequisites available, status/envelope/state correct) | 34 |
| FAIL (prerequisites available, behavior wrong) | 5 (F1 rag 500, F2 legacy-upload 500, F4 refresh reuse, F6 shared GCRA quota, F9 sessions list) |
| BLOCKED (prerequisite absent — not counted as fail) | 4 (subscription/order creation w/o admin+plan; ingestion completion w/o healthy worker; >50 MB probe w/o rate budget; impersonation/admin lifecycle w/o admin token) |
| EXCLUDED (owner instruction) | agent_saul router (all `/api/v*/agent-saul*`), OAuth/SSO/callback routes |
| NOT RUN | crawl cancel, avatar upload, password-reset completion, dunning run, webhook admin replay |
| Unexpected 5xx | 3 distinct deterministic 500/503-class defects (F1, F2, F12b transient) |
| Cross-tenant data exposure | 0 (all ownership probes correctly denied: users:read, corpus-admin, credit-balance, crawl-id, doc-status) |

`pass_rate = PASS / (PASS + FAIL) = 34/39 ≈ 87%` (BLOCKED and NOT RUN out of the
denominator per the plan). `coverage = attempted / enabled planned scenarios`;
Agent Saul + OAuth + admin-only billing lifecycle were not enabled.

### Defects summary

F1 (P0) rag route 500s always · F2 (P1) legacy upload 500-after-persist ·
F3 (P1) branch-switch poisoning + no build provenance (`git_sha: unknown`) ·
F4 (P1) refresh-token reuse undetected · F5 (P2) ingestion worker silent exit +
stuck `received` + node-name collision + direct-exchange ETA blocking ·
F6 (P2) shared unscoped GCRA bucket breaks per-endpoint quotas · F7 (P2) no SMTP
→ fresh users cannot verify (10-28 F5 persists; 10-28 F4 tenant provisioning is
FIXED) · F8 (P2) crawler task fails at capability task boundary · F9 (P2)
/auth/sessions always empty · F10 (P3) version identity null at receive ·
F11 (P3) invoices vs subscriptions envelope inconsistency · F12 (P3) cold-boot
health 503, transient Redis 503 on upload, failed boot left no logs.

## Post-fix verification round (2026-11-15)

Second full sweep AFTER the fix batch landed on main as six commits
(`cf5e0f7` retrieval, `cbb3e85` auth, `831a783` http, `c923804`
worker/security/health, `e9eae0f` rate-limit coverage, `ddeece7` smoke docs)
and the stack was restarted by the orchestrator. Gate: `/tmp/fix_crews_complete`.
Unit suite reported 2570 passed before the round. Evidence: curl status + body
(jq) + unit journals. No commits, no branch switches, no source/`.env` edits,
no unit restarts by the tester.

**Provenance:** all four systemd units active at pre-flight;
`GET /health` → 200 with `data.git_sha = "ddeece7"` (deep health now carries
build provenance). Note: the API unit was stopped/started twice at 04:19:08 and
04:19:21 IST during the sweep (not by the tester; journal shows only systemd
lines) — all probes after ~04:19 hit the restarted process; no probe was
corrupted by the boundary. `GET /` root envelope STILL reports
`git_sha: "unknown"` (residual — see F3 verdict).

**Environment observations during the round:** postgres unhealthy for the
ENTIRE round (latency 3243 ms at pre-flight → 3570 ms at 04:56; `/health`
degraded to overall 503 mid-round, `/health/ready` 503, typed per-dep status
correct). This is not cold-start — it persisted 40+ minutes. One login attempt
returned a typed 503 `DATABASE_ERROR` "Failed to create refresh-token family"
during the latency spike (G4). The uvicorn journal contains **zero application
output lines** in 90 minutes (F12c persists/worsened: not even access lines, so
Razorpay dashboard deliveries and worker task receipts are unobservable from
journals).

### Defect verdicts (evidence-based)

| Defect | Verdict | Evidence |
|---|---|---|
| F1 rag deterministic 500 + traceback leak | **FIXED** | POST /api/v2/search/rag with `{"query":"termination notice","max_tokens":2000}` → **200** `{items:[],context:[],cache_hit:false}`; without max_tokens → 200 (cache_hit:true). Every 5xx body observed this round had `error.trace: null` (no "Traceback", no filesystem paths) |
| F2 legacy upload 500-after-persist | **FIXED** | POST /api/v2/ingestion/documents/upload → **201**, headers `deprecation: true`, `sunset: 2027-06-15`, `link: </documents/upload>; rel="successor-version"`; body carries doc_id `f0102088…` AND the full legacy field set (`entity_count`, `clause_count`, `relationship_count`, `dropped_entity_count`) |
| F3 branch poisoning / provenance | **PARTIAL** | `/health` data.git_sha = `ddeece7` (requirement met); root route `GET /` still `git_sha:"unknown", build_date:"unknown"` |
| F4 refresh-reuse detection | **PARTIAL** (mechanism FIXED, rendering broken — G1) | refresh → 200 rotation; re-POST OLD token → **500 INTERNAL_SERVER_ERROR** (deterministic ×2, correlationId `1grNr4PKC4L2h5r71hvT1`); family + session ARE revoked: the successor child token then → 401 `INVALID_CREDENTIALS` "Session expired or revoked"; garbage token → 401 `UNAUTHORIZED`. Typed code `REFRESH_TOKEN_REUSE` exists in `src/app/features/auth/domain/errors.py` but never reaches the wire — hypothesis: an exception after the Lua-script revocation (e.g. the security-event emit in `service.refresh`) escapes the Failure path and hits the generic handler |
| F5 ingestion worker silent exit | **STILL BROKEN** | Unit (recreated 04:21:02 by orchestrator, clean import) died silently at **04:23:06** (2 m 4 s, 57.3 M memory peak, zero app output, no task receipt). Document `2552abb2…` stayed `received` through ~20 min of 5 s polling (POLL DONE final=received); second doc `f0102088…` also stuck `received`. The role-node-name half of the fix IS in code (`src/app/worker/celery/cli.py` forces `-n {role}@%h` and rejects caller `-n`), but runtime node names are unobservable (no journal output) and the silent exit itself reproduces |
| F6 unscoped GCRA buckets | **FIXED** | Scoped buckets verified: `scope="register"` (3/60) and `scope="login"` (5/60) (`auth/api/http/router.py:49-50`). 3 registers → first login got typed 401 (NOT 429). Register class: 4th rapid register → 429 `TOO_MANY_REQUESTS` `retry-after: 13`. Login class: rapid burst → 200 ×5-ish then 429 ×4 with `retry-after: 3`/`2`. (One earlier 6-burst all-200 anomaly recorded honestly — likely Redis fail-open transient; immediately after, the same burst tripped at #4) |
| F7 verification gate / SMTP | **STILL BROKEN** (unchanged) | fresh-user login → 401 `INVALID_CREDENTIALS` "Email not verified. Check your inbox."; resend-verification/forgot-password → canned 200s. Resend key+templates now configured but delivery to a non-owner inbox is unconfirmable (test mode) — BLOCKED-pending-user |
| F8 crawler task-boundary failure | **STILL BROKEN** (new root cause — G2) | POST /crawl → 202 `c21821a9…`; consumed and terminal `failed` ~45 s with `error_message: "Unexpected error in _crawl_web at line 1448 in _make_config_signature (crawl4ai/browser_manager.py): Error: 'dict' object has no attribute 'latitude'"` + code context. The old "Capability owner synchronous operations" error is GONE (boundary fix landed — the task now executes), but geolocation is passed as a dict where crawl4ai expects an object; happy path still unreachable |
| F9 sessions list empty | **FIXED** | GET /auth/sessions → 200 with `n=3`: `{sessionId, deviceId, deviceName, ipAddress:"127.0.0.1", userAgent:"curl/8.18.0", createdAt, expiresAt, isCurrent}` |
| F10 version null at receive | **STILL BROKEN** | GET /documents/{id}/status at receive: `document_version: null, pipeline_version: null` (title/kind/chunk_count 0 present) |
| F11 invoices vs subscriptions envelope | **STILL BROKEN** | GET /invoices → bare array `[]`; GET /subscriptions → `{items,limit,offset,total}` |
| F12 resilience/observability | **PARTIAL** | (a) health degradation is typed (`/health` 503 with per-dep list; `/health/ready` 503) but postgres stayed unhealthy the whole round (G4); (b) the one transient infra blip on the auth write path produced a typed 503 (G4); (c) journal log sink STILL absent — 0 app lines in 90 min (worse than round 1, where access lines were visible; Razorpay deliveries and worker receipts are now unobservable) |
| git_sha (gate requirement) | **PASS** | `/health` → `data.git_sha = "ddeece7"`, never "unknown" |
| Webhooks | **PASS** | no signature → 401 `WEBHOOK_VERIFICATION_FAILED` "Missing X-Razorpay-Signature header"; invalid sig → 401 "Signature mismatch"; locally-computed valid HMAC (secret from `.env.development`, body hashed byte-exact) → **200 `{"accepted":true}`**; immediate duplicate → **200 `{"accepted":true}`** (idempotent). NOTE: a first "valid" probe failed 401 due to a tester-side newline bug (`echo` added `\n` to the body file while the HMAC was computed without it) — re-run with byte-exact body verified 200. Real Razorpay dashboard deliveries could NOT be observed: the journal has no app lines (F12c) |
| Worker node names | **PASS (code-level)** | `cli.py` derives `-n {role}@%h` and forbids caller `-n`/`--hostname`; pidbox collision impossible by construction. Runtime confirmation unavailable (no journal output) |

### Full endpoint results

| Method + path | Status | Observed behavior | Required backend piece |
|---|---|---|---|
| GET / | 200 | envelope; `git_sha:"unknown"` residual | none |
| GET /health | 200 → 503 | git_sha `ddeece7`; postgres unhealthy 3243→3570 ms all round; others healthy | Timescale |
| GET /api/v2/health/live | 200 | server info envelope | none |
| GET /api/v2/health/ready | 503 | typed unhealthy (postgres) with system block | postgres |
| GET /api/v2/health/self | 200 | identity block | none |
| GET /metrics | 200 | Prometheus text | none |
| GET /swagger.json | 200 | 138 operations | none |
| GET /openapi.json | 404 | docs disabled | none |
| POST /api/v2/auth/register (fresh) | 201 | "Check your email"; Resend configured, delivery unconfirmable | Resend |
| POST /api/v2/auth/register (duplicate) | 409 | `USER_CONFLICT` | none |
| POST /api/v2/auth/register (class exhausted) | 429 | `TOO_MANY_REQUESTS`, `retry-after: 13` | Redis |
| POST /api/v2/auth/login (principal) | 200 (+1×503 transient) | camelCase `accessToken/refreshToken`; the 503 = `DATABASE_ERROR` "Failed to create refresh-token family" during postgres spike (G4) | postgres |
| POST /api/v2/auth/login (fresh unverified) | 401 | `INVALID_CREDENTIALS` email-not-verified (F7 gate) | Resend/verification |
| POST /api/v2/auth/login (class exhausted) | 429 | `TOO_MANY_REQUESTS`, `retry-after: 3`/`2` | Redis |
| GET /api/v2/auth/me | 200 / 401 no-token | same user id; typed `UNAUTHORIZED` | none |
| GET /api/v2/auth/sessions | 200 | 3 sessions, sessionId/ip/ua/isCurrent (F9 FIXED) | Mongo/Redis |
| POST /api/v2/auth/refresh | 200 | rotation OK | none |
| POST /api/v2/auth/refresh (reused old) | **500** | `INTERNAL_SERVER_ERROR` ×2; family+session revoked anyway (G1) | none |
| POST /api/v2/auth/refresh (successor after reuse) | 401 | `INVALID_CREDENTIALS` "Session expired or revoked" — family revocation propagates | none |
| POST /api/v2/auth/refresh (garbage) | 401 | `UNAUTHORIZED` | none |
| POST /api/v2/auth/resend-verification | 200 | canned "If that email is registered…" | Resend |
| POST /api/v2/auth/forgot-password | 200 | canned | Resend |
| GET/PATCH /api/v2/profile/ | 200/200 | PATCH persists (re-read confirms `fullName`); camelCase DTO | none |
| PATCH /api/v2/profile/ (unknown field) | 422 | `VALIDATION_ERROR` | none |
| POST /api/v2/profile/change-password (wrong current) | 401 | `CURRENT_CREDENTIAL_INVALID` | none |
| POST /api/v2/profile/change-password (missing field) | 422 | field-level | none |
| GET /api/v2/users/ + /{id} (role=user) | 403 | `FORBIDDEN` users:read | none |
| PATCH /api/v2/tenants/{own}/members/{self}/legal-roles | 403 | `FORBIDDEN` "Corpus-admin authority is required" | none |
| PATCH /api/v2/tenants/{own}/settings/self-review | 403 | same corpus-admin denial | none |
| PATCH /api/v2/tenants/{foreign}/settings/self-review | 403 | distinct "Tenant membership does not grant access to this tenant" | none |
| POST /api/v2/documents/upload (.txt) | 201 | `queued`, doc `2552abb2…`, task_id null (outbox path) | celery ingestion |
| POST /api/v2/documents/upload (identical bytes) | 201 | `duplicate:true`, SAME doc_id | none |
| POST /api/v2/documents/upload (.exe) | 422 | `INVALID_DOCUMENT` `EXTENSION_NOT_ALLOWED`, allowlist echoed | none |
| POST /api/v2/documents/upload (55 MB) | 422 | `INVALID_DOCUMENT` `FILE_TOO_LARGE`, `max_size: 52428800` (round-1 NOT RUN → now pass) | none |
| GET /api/v2/documents/{id}/status | 200 | truthful `received`; `document_version/pipeline_version` null (F10); one call took 20.1 s (postgres latency) | none |
| GET /api/v2/documents/{unknown}/status | 404 | `STATUS_NOT_FOUND` | none |
| GET/DELETE /api/v2/documents (list/delete) | — | still absent from router | — |
| POST /api/v2/ingestion/documents/upload (legacy) | **201** | deprecation headers + legacy fields (F2 FIXED) | none |
| POST /api/v2/search (scoped / unauth / empty) | 200 / 401 / 422 | typed empty result; typed errors | embedding+postgres |
| POST /api/v2/search/rag (with and without max_tokens) | **200 / 200** | `{items,context,cache_hit}` — F1 FIXED, no trace leak | none |
| POST /api/v2/search/ask | 200 | grounded decline, citations [], confidence "uncertain" | model provider |
| POST /api/v2/legal/ask (field=query) | 200 | same decline shape | model provider |
| POST /api/v2/search/hybrid | 200 | typed empty | postgres |
| POST /api/v2/search/ingest (+poll) | — | BLOCKED: ingestion worker dead (F5) | celery ingestion |
| GET /api/v2/crawler/rate-limit | 200 | `{remaining_minute:10, remaining_hour:99}` | Redis |
| GET /api/v2/crawler/search?query= | 200 | Tavily live, real results | Tavily |
| GET /api/v2/crawler/search?q= | 422 | param is `query` | none |
| POST /api/v2/crawler/crawl | 202 | pending → consumed → terminal `failed` (G2, crawl4ai geolocation) | crawler worker |
| GET /api/v2/crawler/crawl/{id} | 200 | truthful status + surfaced error_message | Redis |
| GET /api/v2/crawler/crawl/{id}/pages, /chunks | 503 | typed fail-closed "not available yet" | crawler worker |
| GET /api/v2/crawler/crawl/{id}/chunks/search | 422 | requires query param | none |
| GET /api/v2/crawler/crawl/{unknown} | 404 | `NOT_FOUND` | Redis |
| POST /api/v2/crawler/crawl/{id}/cancel | — | NOT RUN (crawl never reached running) | — |
| GET /api/v2/plans | 200 | empty catalog `[]` | postgres |
| GET /api/v2/plans/{non-uuid} | **500** | `PLAN_DATABASE_ERROR` with raw SQL + asyncpg error in `error.data` (G3; was 404 `PLAN_NOT_FOUND` in round 1) | none |
| POST /api/v2/plans (role=user) | 403 | admin-only | none |
| GET /api/v2/subscriptions | 200 | paginated envelope | postgres |
| GET /api/v2/invoices | 200 | bare array (F11 persists) | postgres |
| POST /api/v2/subscriptions (unknown plan) | 422 | `VALIDATION_ERROR` — plan_id now UUID-validated at the DTO (round-1 saw 404 `PLAN_NOT_FOUND`; benign earlier-rejection change) | none |
| POST /api/v2/subscriptions/{unknown}/cancel | 422 | validation rejects unknown id | none |
| GET /api/v2/credits/balance/{self} | 200 | `{totalBalance:0,…,currency:"INR"}` | postgres |
| GET /api/v2/credits/balance/{other} | 403 | `CREDIT_FORBIDDEN` | none |
| GET /api/v2/credits/history/{self} | 200 | history envelope | postgres |
| POST /api/v2/credits (grant, role=user) | 403 | admin-only | none |
| GET /api/v2/admin/billing/dunning (role=user) | 403 | `FORBIDDEN` | none |
| POST /api/v2/webhooks/razorpay (no/invalid sig) | 401 | typed `WEBHOOK_VERIFICATION_FAILED` | none |
| POST /api/v2/webhooks/razorpay (valid HMAC + duplicate) | 200 / 200 | `{"accepted":true}`, idempotent | none |
| GET /api/v1/documents/{id}/status, POST /api/v1/search, GET /api/v1/auth/me | 200 | v1 dual-mount behavior + `deprecation: true`, `sunset: 2027-06-15`, successor `Link` headers all present | none |

### New defects

- **G1 (P1): refresh-token reuse renders 500 `INTERNAL_SERVER_ERROR` instead of
  the typed 401 `REFRESH_TOKEN_REUSE`.** Deterministic ×2 (correlationId
  `1grNr4PKC4L2h5r71hvT1` and one earlier). The rotation protocol itself is
  correct — reuse is detected, family + session revoked, successor token
  rejected with 401 — but the wire contract is wrong (F4's typed error exists
  in `src/app/features/auth/domain/errors.py` yet never renders). Suspect an
  exception after revocation (security-event emit path) escaping the Failure
  render. No traceback/paths in the body (`error.trace: null`).
- **G2 (P2): crawler happy path still unreachable with a new root cause.**
  `error_message: "Unexpected error in _crawl_web at line 1448 in
  _make_config_signature (…/crawl4ai/browser_manager.py): Error: 'dict' object
  has no attribute 'latitude'"` — the app passes `geolocation` as a dict where
  crawl4ai expects a `Geolocation` object. Zero pages attempted; terminal state
  truthful. (Round-1 F8's capability-owner error is gone — that fix landed.)
- **G3 (P1): `GET /api/v2/plans/{non-uuid}` → 500 `PLAN_DATABASE_ERROR`** with
  the raw SQLAlchemy statement (full `plans` column list) and the asyncpg
  DataError text serialized into `error.data`. Round-1 behavior was typed 404
  `PLAN_NOT_FOUND` — a regression AND an internal SQL/schema disclosure (no
  filesystem paths or "Traceback", but internal query text leaks).
- **G4 (P2): postgres unhealthy for the entire round** (3.2–3.6 s latency,
  `/health` and `/health/ready` degraded to typed 503 for 40+ minutes — not
  cold-start), and one login during the spike returned typed 503
  `DATABASE_ERROR` "Failed to create refresh-token family" — an infra blip on
  one dependency fails the auth write path. (Observability caveat: the app's
  stdout never reaches journald — 0 app lines in 90 min — so neither this nor
  Razorpay deliveries could be correlated from logs; F12c.)

### Counts

| Metric | Count |
|---|---|
| Distinct HTTP operations probed (v2 + v1 spot checks) | 61 |
| PASS | 53 |
| FAIL (prerequisites available, behavior wrong) | 6 (G1 reuse 500, G2 crawl failed, G3 plans/unknown 500, F5 ingestion round-trip, F10 version null, F11 invoices envelope) |
| BLOCKED | 4 (search/ingest pipeline — worker dead; subscription/order/payments lifecycle — no plan/admin; email delivery confirmation — Resend test mode; Razorpay dashboard-delivery observation — journal sink absent) |
| NOT RUN | crawl cancel, impersonation/admin lifecycle, avatar upload, dunning run, webhook admin replay |
| Unexpected 5xx | 3 distinct (G1, G3, G4-transient) — none leaked tracebacks/filesystem paths |
| Cross-tenant data exposure | 0 |

`pass_rate = 53/59 ≈ 90%` (BLOCKED/NOT RUN out of denominator). Prior-defect
scoreboard: **F1, F2, F6, F9 FIXED · F4 PARTIAL (G1) · F3, F12 PARTIAL ·
F5, F7, F8, F10, F11 STILL BROKEN · git_sha PASS · webhook PASS**.

### Cleanup

Test data left in infra: users `postfix-20261115-a/b/c/d@example.com`
(unverified); documents `2552abb2…` (stuck received), `f0102088…` (legacy route,
stuck received); crawl `c21821a9…` (failed); principal `fullName` set to
"Postfix Sweep Tester". Token families deliberately poisoned during the F4
reuse tests (principal re-logged-in afterward; final tokens left UNREVOKED so
the orchestrator can keep using the principal — no destructive session cleanup
was run). One webhook test event `payment.captured` id `pay_postfix_g1`
accepted (idempotency verified on duplicate). No commits, no branch switches,
no source or `.env.development` modifications were made by the tester.

## Orchestrator observation (post-round-2, 2026-11-15 ~23:35 IST)

- **G5 (P1, billing)**: `POST /api/v2/plans` with the correct `PlanCreateDTO` → plan row PERSISTS
  locally (`GET /plans` shows `smoke-pro-monthly`, id `6eb09826-…`) while the client receives 500
  `PLAN_DATABASE_ERROR` "The billing provider credentials were rejected". Direct probe with the
  same key pair: Razorpay `/v1/webhooks` accepts (PUT 200), `/v1/plans` → **401 Unauthorized** —
  the credentials are valid but the account/key pair lacks Plans capability. Owner action:
  enable/verify Plans on the Razorpay dashboard or issue a key pair with plan rights.
  Two defects in one: (a) local-persist-then-500 state divergence (F2 class), (b) the error code
  `PLAN_DATABASE_ERROR` mislabels a provider-auth failure. Reconciliation of the local row is owed.
- **Admin principal created** (owner-approved): `.testing-credentials.json` → `"admin"` key
  (register → Mongo-seed verify → production role bump; JWT carries `role: admin` + full
  permission set; `GET /api/v2/plans` 200 authenticated). Billing lifecycle unblocked for the
  expanded sweep except provider-side plan mirroring (G5 above).

## Expanded sweep (2026-11-15, ~07:35–08:15 IST)

Third sweep on main @ `e03b3fe` after the poison-message/DLQ crew landed. Stack freshly
restarted by the orchestrator at 07:34:58 (uvicorn) / 07:35:07 (celery-ingestion); all four
units active at pre-flight. Identity: principal (`testing@live-measurement.dev`) + admin
(`testing-admin@live-measurement.dev`, JWT `role: admin` verified). Evidence: curl status +
jq'd body + unit journals + direct broker inspection. No commits, no branch switches, no
source/`.env` edits, no unit restarts by the tester.

### Headline: F5 verdict — worker SURVIVES, round trip WORKS, DLQ routing is a no-op (G6)

- **Survival: PASS.** `agentnexus-celery-ingestion` up continuously since 07:35:07 (verified
  active at sweep start, mid, and end — >40 min without the historical ~2-min silent exit).
- **Runtime node names: PASS (verified live).** `celery inspect` reports three distinct
  nodes `default@Aging-Devil`, `ingestion@Aging-Devil`, `crawler@Aging-Devil` — the F5
  node-collision fix is confirmed at runtime, not just in code.
- **Document round trip: PASS.** Upload `f5-roundtrip-doc.txt` (07:46) → doc
  `6960a106-28cd-4812-b79e-518f1f991ef6` `queued` → terminal
  **`completed_with_warnings`** (`document_version:1`, `pipeline_version:"documents-v1"`,
  `chunk_count:1`, `DEGENERATE_PARSE` warnings for a single-paragraph txt — truthful).
  Search probe against it: `POST /search` → 4 items; `POST /search/rag` → 200 (4 items);
  `POST /search/ask` → `confidence:"high"` with **2 citations**. Ingestion latency from
  upload to searchable: well under 6 minutes (v1 upload later hit terminal in <1 min).
- **Backlog drained, nothing wedged.** At boot the worker consumed the historical
  ETA-blocked messages (4 delayed-delivery warnings 07:36:57–58, incl. one referencing
  `tasks.dlx`); `celery inspect` at 07:57 shows `active/reserved/scheduled = 0` for all
  three roles, and RabbitMQ queue depths `ingestion=0, default=0, crawler=0`.
- **BUT the poison/DLQ acceptance criterion is NOT met — two new defects:**
  - **G6 (P1): the dead-letter infrastructure does not exist on the broker.** `tasks.dlx`
    exchange: NOT_FOUND; `default.dlq` queue (settings.CELERY_DEAD_LETTER_QUEUE): NOT_FOUND;
    no queue is bound to any DLX. A `Reject(requeue=False)` poison message (the fix in
    `ResilientTask.before_start`, `src/app/worker/celery/task.py:76-77`) is therefore
    **silently discarded by RabbitMQ**, not dead-lettered — no retention, no forensics.
  - **G7 (P2): task-time worker logging never reaches the journal.** Boot-time output DOES
    reach journald now (Makefile fd fix confirmed: celery banner, cognee init, ETA warnings),
    yet the entire document-ingestion window (07:44–07:52) produced **zero** journal lines —
    no `Received task`, no `succeeded`, and no `task_payload_rejected` line either. The
    typed DLQ rejection line the fix crews promised is unobservable, and 10-28 F3 (invisible
    task receipts) still holds at task runtime.

### Per-endpoint table (129 swagger ops − 4 agent-saul − 12 OAuth = 113 probed)

| Method + path | Status | Observed behavior | Required backend piece |
|---|---|---|---|
| GET / | 200 | `git_sha:"e03b3fed"` — **root provenance now real (F3 re-probe PASS)** | none |
| GET /health | 200→degraded | all 7 deps healthy at pre-flight; **redis unhealthy late in the sweep (see G8)** | all cloud deps |
| GET /api/v2/health/live, /ready, /self, / | 200 | same as round 2; /ready 200 while healthy | postgres etc. |
| GET /api/v1/health/ + live/ready/self | 200 | enveloped, deprecation headers | none |
| GET /metrics | 200 | Prometheus text | none |
| GET /swagger.json | 200 | 129 operations (recounted) | none |
| GET /openapi.json | 404 | docs disabled | none |
| POST /api/v2/auth/register (fresh) | 201 | canned "Check your email"; unverified (F7 unchanged) | Resend |
| POST /api/v2/auth/verify-email (bogus token) | 404 | typed `USER_NOT_FOUND` (both v1+v2) | none |
| POST /api/v2/auth/reset-password (bogus) | 404 | typed `USER_NOT_FOUND` | none |
| POST /api/v2/auth/forgot-password | 200/429 | canned; class exhaustible (below) | Resend |
| POST /api/v2/auth/resend-verification | 200/429 | canned; class exhaustible (below) | Resend |
| POST /api/v2/auth/login | 200 | both identities; 900 s access TTL observed | postgres+redis |
| GET /api/v2/auth/me | 200/401/503 | typed; **503 `DATABASE_ERROR` after redis went unhealthy (G8)** | redis session store |
| GET /api/v2/auth/sessions (+v1) | 200 | 19 sessions listed (F9 fix holds) | Mongo |
| POST /api/v2/auth/refresh | 200 | rotation OK | none |
| POST /api/v2/auth/refresh (REUSED old) | **401 `REFRESH_TOKEN_REUSE`** | **G1 FIXED** — typed 401 "Refresh token reuse detected; session revoked" (was 500 in round 2); successor child token → 401 `INVALID_CREDENTIALS` "Session expired or revoked" | none |
| POST /api/v2/auth/logout | 200 | requires `refresh_token` body; revoked token → 401 `UNAUTHORIZED`; missing body → 422 | none |
| DELETE /api/v2/auth/sessions/{bogus} | 404 | typed **`USER_NOT_FOUND`** (mislabel candidate: a session-scoped 404 says USER_NOT_FOUND) | none |
| GET/PATCH /api/v2/profile/ (+v1) | 200 | PATCH persists (`full_name`), 422 on unknown field; v1 deprecation headers | none |
| POST /api/v2/profile/change-password (wrong current) | 401/429 | `CURRENT_CREDENTIAL_INVALID`; class exhaustible | none |
| POST /api/v2/profile/avatar (+v1) | 200 | avatar uploaded, URL returned (round-2 NOT RUN → pass) | R2 |
| GET /api/v2/users/ (+v1, role=user) | 403 | `FORBIDDEN` users:read | none |
| GET /api/v2/users/{non-uuid} | 422 | typed `INVALID_USER_ID` (contrast with old G3 behavior) | none |
| GET /api/v2/users/{id} (admin) | 200 | works for admin (v1 too) | postgres |
| PATCH /api/v2/users/{id}/role | 200/422/429 | admin-only; `role` enum admin/moderator/user | postgres |
| PATCH /api/v2/users/{id}/deactivate, /activate | 200/200 | admin lifecycle works | postgres |
| POST /api/v2/users/{id}/impersonate | 200 | token issued (`accessToken`,`expiresIn`,`impersonatingUserId`) — 403 for role=user | postgres |
| DELETE /api/v1/users/{id} | 200 | "User permanently deleted" (throwaway) | postgres |
| PATCH /api/v2/tenants/{tid}/settings/self-review (+v1) | 403 | own tenant: "Corpus-admin authority is required"; foreign: "Tenant membership does not grant access…" | none |
| PATCH /api/v2/tenants/{tid}/members/{uid}/legal-roles (+v1) | 403 | same corpus-admin denial | none |
| POST /api/v2/documents/upload (.txt) | 201 | queued → `completed_with_warnings`; duplicate bytes → 201 `duplicate:true` (round-1/2 idempotency holds) | celery ingestion |
| GET /api/v2/documents/{id}/status (+v1) | 200 | **`document_version:1, pipeline_version:"documents-v1" AT RECEIVE TIME** (F10 re-probe PASS — was null) | none |
| GET /api/v2/documents/{unknown}/status | 404 | `STATUS_NOT_FOUND` | none |
| POST /api/v2/ingestion/documents/upload (legacy, +v1) | 201 | deprecation + sunset + Link headers; legacy field set present (F2 fix holds) | none |
| POST /api/v1/documents/upload | 201 | deprecation header; doc ingested to terminal <1 min | celery ingestion |
| POST /api/v2/search | 200/401/422 | scoped search returns 4 items from the F5 doc | embedding+postgres |
| POST /api/v2/search/rag (+v1) | 200 | F1 fix holds; items returned; no trace leak | none |
| POST /api/v2/search/ask (+v1) | 200 | `confidence:"high"`, 2 citations from the ingested doc | model provider |
| POST /api/v2/legal/ask (+v1) | 200 | grounded decline on out-of-corpus question | model provider |
| POST /api/v2/search/hybrid | 200 | typed empty (hybrid path still returns [] with items present in vector path — see G9 note) | postgres |
| POST /api/v2/search/ingest | **503** | typed `SEARCH_INFRASTRUCTURE_ERROR` "Could not stage search ingest job" with `data.document_id` present — stage-failure AFTER creating the doc row (F2-class divergence, see G9) | redis/jobs |
| GET /api/v2/search/ingest/{task_id} | 404 | typed for unknown task | redis |
| GET /api/v2/crawler/rate-limit (+v1) | 200 | `{remaining_minute,remaining_hour}` | redis |
| GET /api/v2/crawler/search (+v1) | 200/422 | Tavily live (10 results); `q` → 422 field-required | Tavily |
| POST /api/v2/crawler/crawl | 202 | consumed → terminal `failed` — **G2 UNCHANGED**: crawl4ai `_make_config_signature` "Error: 'dict' object has no attribute 'latitude'" | crawler worker |
| GET /api/v2/crawler/crawl/{id} (+v1) | 200 | truthful failed + surfaced error | redis |
| GET /api/v2/crawler/crawl/{id}/pages, /chunks (+v1) | 503 | typed fail-closed "not available yet" | crawler worker |
| GET /api/v2/crawler/crawl/{id}/chunks/search | 503 | same (no results to search) | crawler worker |
| POST /api/v2/crawler/crawl/{id}/cancel (+v1) | 200 | "Crawler cancellation requested" — succeeds on an already-FAILED crawl (idempotent-accept; questionable state check) | redis |
| GET /api/v2/plans (admin/user) | 200 | catalog readable by role=user now (round-1 403 — deliberate opening, users browse plans) | postgres |
| GET /api/v2/plans/{id} | 200 | full plan incl. `razorpayPlanId`, idempotency metadata | postgres+razorpay |
| GET /api/v2/plans/{non-uuid} | **404 `PLAN_NOT_FOUND`** | **G3 FIXED** — typed 404, `data.plan_id`, no SQL/asyncpg leakage (was 500 PLAN_DATABASE_ERROR in round 2) | none |
| POST /api/v2/plans (admin) | **201** | **G5 FIXED** — plan mirrored: `razorpayPlanId:"plan_U0HeO8T1ATOO7d"`, metadata `billing_provider_sync:"synced"`; local+provider row consistent | razorpay plans |
| PATCH /api/v2/plans/{id} | 200→bricked | **G6b (new): description-only PATCH flips `isActive:true→false`**; further PATCH → 422 `INVALID_PLAN_UPDATE` "Cannot update inactive plan"; subscription create against it → 422 "Plan is not active". A plan edit bricks the plan. | postgres |
| POST /api/v2/plans/{id}/archive | 200 | archives (isActive→false), typed | postgres |
| POST /api/v2/subscriptions (real active plan) | **500** | `DATABASE_ERROR` "The billing provider rejected the request" (`provider_code:"invalid_request"`) — Razorpay Subscriptions product not enabled on the key pair (owner action, BLOCKED per brief). **AND the local subscription row persists anyway** (`GET /subscriptions` shows it, `status:"created"`, `razorpaySubscriptionId:null`, `remoteSync:"permanent_failure"`) — persist-then-500 divergence (F2 class, see G9) | razorpay subscriptions |
| GET /api/v2/subscriptions (+{id}, change-preview, payments) | 200/422 | paginated envelope; change-preview 422 "Proration is only valid for active subscriptions" | postgres |
| POST /api/v2/subscriptions/{id}/pause, /resume, /cancel, /change-plan, /trial-extension | 422 | typed state-machine rejections (`INVALID_STATE_TRANSITION` created→paused/active/cancelled; trial-extension "Subscription is not in trial") — correct DTOs verified via swagger | postgres |
| GET /api/v2/invoices | 200 | **F11 FIXED — paginated `{items,total,limit,offset}` envelope now** (was bare array) | postgres |
| GET /api/v2/invoices/{unknown} | 404 | `INVOICE_NOT_FOUND` | postgres |
| POST /api/v2/invoices/{unknown}/void | 422 | typed validation (limiter class blocked by G8 outage — see 429 matrix) | postgres |
| GET /api/v2/payments/{unknown} | 404 | `PAYMENT_NOT_FOUND` | postgres |
| POST /api/v2/payments/{unknown}/refund | 422 | typed validation | postgres |
| POST /api/v2/credits (admin grant) | 201 | needs `metadata_.target_user_id` + `admin_user_id` (typed `CREDIT_METADATA_MISSING` otherwise); grant of ₹50 to principal succeeded (balance 5000 paisa) | postgres |
| GET /api/v2/credits/balance/{self} (+admin viewing other) | 200/403 | self 200; cross-user `CREDIT_FORBIDDEN` (even for admin — admin cannot view user balances; noted) | postgres |
| GET /api/v2/credits/history/{self} | 200 | envelope with entries | postgres |
| POST /api/v2/credits/apply-to-invoice | 422 | typed (no invoice exists) | postgres |
| GET /api/v2/admin/billing/dunning (admin) | 200 | `{retryDelayDays:[1,3,7,14], maxRetries:4}` (role=user → 403 per round 1) | postgres |
| POST /api/v2/admin/billing/dunning/run (admin) | 200 | `{checked:0, retried:0, items:[]}` — typed empty run | postgres |
| POST /api/v2/admin/billing/webhooks/{id}/replay (admin) | 404 | typed `WEBHOOK_EVENT_NOT_FOUND` | postgres |
| POST /api/v2/webhooks/razorpay (no/bad sig) | 401 | `WEBHOOK_VERIFICATION_FAILED` | none |
| POST /api/v2/webhooks/razorpay (valid HMAC) | 200/200 | `{"accepted":true}` + idempotent duplicate | none |

### 429 matrix (limiter classes, exhausted sequentially; Retry-After asserted)

| Class | Route | 429 at attempt | Retry-After | Verdict |
|---|---|---|---|---|
| profile-mutation | PATCH /api/v2/profile/ | 11 | 3 s | PASS (scoped) |
| change-password | POST /api/v2/profile/change-password | 6 | 58 s | PASS |
| avatar-upload | POST /api/v2/profile/avatar | 6 | 56 s | PASS |
| forgot-password | POST /api/v2/auth/forgot-password | 4 | 93 s | PASS |
| resend-verification | POST /api/v2/auth/resend-verification | 3 | 150 s | PASS |
| tenant-mutation | PATCH /api/v2/tenants/{tid}/settings/self-review | 24 | 2 s | PASS |
| user-mutation | PATCH /api/v2/users/{id}/role | 34 (40-parallel burst) | present | PASS (quota 30/60; sequential pacing never trips it — burst-only) |
| billing-mutation | POST /api/v2/credits | — | — | BLOCKED by G8 redis outage (503 DATABASE_ERROR; limiter fail-open) |
| (final-state note) | — | — | — | G8 persisted to sweep end (~22 min): `max number of clients reached` on direct ping; **admin re-login during the outage → 503 `DATABASE_ERROR` "Failed to store session" — full login lockout**; DELETE /api/v2/auth/sessions could not be exercised (admin token expired, re-login blocked by the outage) |
| billing-admin | POST /api/v2/admin/billing/dunning/run | — | — | BLOCKED by G8 |
| payment-refund | POST /api/v2/payments/{id}/refund | — | — | BLOCKED by G8 |
| invoice-void | POST /api/v2/invoices/{id}/void | — | — | BLOCKED by G8 |

All observed 429 bodies were typed `TOO_MANY_REQUESTS`; limiter identity is per-user for
authenticated calls (`user:<sub>` keying) as designed. **The 40-parallel user-mutation
burst triggered G8 below.**

### G8 (P1, NEW — incident during sweep): redis client exhaustion → persistent unhealthy state, no self-recovery

A 40-request parallel burst on `PATCH /users/{id}/role` (user-mutation class) was followed by
Redis Cloud refusing all connections with **`ConnectionError: max number of clients reached`**
(direct ping from the project env, still failing after 6+ minutes of idle). Consequences
observed live: `/health` → 503-degraded with `redis:unhealthy (0 ms)`; `GET /auth/me` → typed
503 `DATABASE_ERROR` (session store unavailable); the remaining four limiter classes fail OPEN
(no 429s possible — quota enforcement silently disabled, by design "fail open"). The app's
pool does not recover on its own (idle connections are held, Redis Cloud plan client cap
exceeded); a unit restart is needed. Root suspects: no pool `max_connections` cap vs the
Redis Cloud plan limit, no idle reaping, and limiter+session traffic over one pool. The state persisted to sweep end (~22 min, never self-recovered) and
**escalated to a full login lockout**: `POST /auth/login` → 503 `DATABASE_ERROR` "Failed to
store session" for every identity (refresh-token family + session writes need Redis).
Every unit journal check during the incident showed the units still active — the outage is
invisible in logs (F12c) and only observable via `/health` and typed 503s.

### Regression re-probe scoreboard

| Probe | Verdict | Evidence |
|---|---|---|
| G1 refresh-reuse | **FIXED** | 401 `REFRESH_TOKEN_REUSE` typed; successor revoked; family propagates |
| G3 plans/{non-uuid} | **FIXED** | 404 `PLAN_NOT_FOUND`, no SQL in body |
| F3 root git_sha | **FIXED** | `GET /` → `git_sha:"e03b3fed"` (and `/health` carries it) |
| F10 versions at receive | **FIXED** | fresh doc status at `received`: `document_version:1, pipeline_version:"documents-v1"` |
| F11 invoices envelope | **FIXED** | paginated `{items,total,limit,offset}` |
| F12c journal sink | **PARTIAL (workers fixed, API still silent)** | ingestion journal carries celery/cognee lines at boot; uvicorn journal has ZERO lines since 07:34:58 boot (unit runs `uv run uvicorn` directly, bypassing the Makefile fd fix); task-time worker lines still invisible (G7) |
| G5 plan create | **FIXED** | POST /plans → 201, Razorpay plan mirrored (`plan_U0HeO8T1ATOO7d`, sync "synced") |
| G2 crawler | **STILL BROKEN** | same crawl4ai geolocation dict error, zero pages |
| F7 SMTP gate | **STILL BROKEN** | fresh signup unverified, canned emails |

### New defects

- **G6 (P1): dead-letter infrastructure absent at the broker.** No `tasks.dlx` exchange, no
  `default.dlq` queue, no binding. Poison messages are dropped on reject, and the promised
  typed rejection line is additionally invisible because task-time logging never reaches the
  journal (G7). Worker stability itself is real — F5's wedge is gone.
- **G6b (P1): plan PATCH bricks the plan.** A description-only `PATCH /api/v2/plans/{id}`
  (200, "Plan updated", description persisted) silently set `isActive:false`; the plan then
  rejects further PATCHes (`INVALID_PLAN_UPDATE`) and subscription creation
  ("Plan is not active"). Reproduced once + archived a second plan safely. State divergence:
  client sees success, plan is dead.
- **G7 (P2): task-time worker logging invisible** (details under F5 verdict).
- **G8 (P1): redis client exhaustion incident** (details above; blocks the remaining four
  limiter classes and redis-backed reads until restart).
- **G9 (P2): persist-then-fail divergences in two new places.** (a) `POST /search/ingest`
  → 503 `SEARCH_INFRASTRUCTURE_ERROR` with `data.document_id` present — the document row was
  created before staging failed; (b) `POST /subscriptions` → 500 `DATABASE_ERROR`
  "provider rejected" while the local subscription row persists (`status:"created"`,
  `remoteSync:"permanent_failure"`, no provider id). Same F2-class shape: the client sees
  failure, the store holds state. Also: the subscription provider-rejection renders as 500
  `DATABASE_ERROR` (should be 4xx, provider_code surfaced).

### Counts

| Metric | Count |
|---|---|
| Operations probed this sweep (113 reachable + sub-probes) | 88 distinct rows above |
| PASS | 70 |
| FAIL | 6 (G2 crawl failed, G6b plan brick, G9a ingest 503-after-persist, G9b sub 500-after-persist, G8 redis exhaustion + limiter fail-open, DELETE bogus session mislabeled USER_NOT_FOUND) |
| BLOCKED | deep money-path (Razorpay Subscriptions product off — provider 500), 4 limiter classes (G8 outage), email delivery confirmation (Resend test mode) |
| EXCLUDED | agent_saul router, OAuth/SSO routes (owner instruction) |
| Unexpected 5xx | 3 distinct sources (G9a 503, G9b 500, G8 503-cascade); no tracebacks/paths in any body |
| Cross-tenant data exposure | 0 |

### Cleanup

Test data left in infra: users `sweep3-20261115@example.com` (unverified, role→moderator→deleted
via DELETE probe), plans `sweep3-pro-monthly` (bricked by G6b probe, id `7bea49e7…`) and
`sweep3-sub-plan` (archived after probe, id `dab4e654…`), plan `smoke-pro-monthly` (pre-existing
G5 orphan, still listed), subscription `d15593de…` (orphan `created`, provider-rejected),
documents `6960a106…`/`ae22c2ac…`/`ded66afe…`/`42adfaf3…`/`40a5d2cf…` (ingested or queued),
crawl `a0c34dba…` (failed), credit `7525df22…` (₹50 admin_grant to principal), webhook test
event `payment.captured` id `pay_sweep3_g1` (accepted ×2, idempotent). Principal profile set
to "Expanded Sweep Tester"; principal refresh family deliberately poisoned by the G1 probe
then re-login (current principal tokens left UNREVOKED for the orchestrator). Admin sessions
left active. Redis left in the G8 saturated state — **needs a unit restart by the
orchestrator**. No commits, no branch switches, no source/`.env` modifications.

## Remaining-endpoints sweep (2026-11-19)

Fourth sweep on main @ `2a7c9f66` (one commit past `32a38be0`: the billing
dead-create→cancelled fix, G10). All four units active at pre-flight
(uvicorn + default + ingestion + crawler); `GET /health` 200 with all 7 deps
healthy; the uvicorn journal NOW carries structured app output (request lines,
SQL traces — F12c observability for the API unit materially improved since
round 3). Identities: principal (`testing@live-measurement.dev`) + admin
(`testing-admin@live-measurement.dev`). Evidence: curl status + jq'd bodies +
unit journals + source reads (no edits). No commits, no branch switches, no
source/`.env` edits, no unit restarts.

**Scope note: v1 routes skipped per owner instruction (deprecated; v2 is the
supported surface).** Mid-sweep the owner amended the scope to /api/v2 only;
probes already run against /api/v1 before the amendment are kept below but
marked **[v1-deprecated]**.

### Untested-set enumeration (deduplication)

Method: live inventory from `GET /swagger.json` (138 operations, recounted)
minus the union of every method+path row and named sub-probe in the three
sweep tables above, minus the excluded sets (agent_saul: 4 ops; OAuth/SSO
authorize/callback/link/confirm/token routes: 12 ops) → **122 in-scope ops,
105 already probed across rounds 1–3, 17 untested after deduplication**.

The 17 decompose as:

- **2 untested /api/v2 operations** (the real v2-coverage gap):
  `DELETE /api/v2/auth/sessions` (revoke-all) and
  `DELETE /api/v2/users/{user_id}`. A third candidate,
  `POST /api/v2/crawler/crawl/{id}/cancel`, turned out to be a dedup
  false-positive (probed in rounds 2+3).
- **15 v1 duals** of already-probed v2 operations. 13 of them were probed
  pre-amendment and are kept as [v1-deprecated] rows; the remaining 2
  (`GET /api/v1/crawler/crawl/{id}/chunks/search`,
  `POST /api/v1/profile/change-password`) were **skipped per owner
  instruction** (their v2 equivalents were probed in rounds 1–3).

Everything the brief listed as "expected untested candidates" (avatar,
dunning config GET, credits admin surfaces, health sub-endpoints, search
hybrid/ingest/task-status, invoices detail, plan archive, webhook GETs,
impersonate/activate/deactivate, session revoke variants on v2) was verified
against the live spec and found already covered by round 3 — the residue is
exactly the set above.

### Per-endpoint results

| Method + path | Status | Observed behavior | Required backend piece |
|---|---|---|---|
| DELETE /api/v2/auth/sessions?keep_current=false | 200 | "All sessions revoked"; the calling admin token → 401 `UNAUTHORIZED` "Session expired or revoked" on the next request (revocation propagates immediately); admin re-login → 200 restores full access | Mongo session store + Redis revocation stream |
| DELETE /api/v2/users/{user_id} (admin) | 200 | "User permanently deleted"; follow-up `GET /api/v2/users/{id}` → 404 `USER_NOT_FOUND` (hard delete, no remnant) | Postgres |
| DELETE /api/v2/auth/sessions/{revoked-sid} (sub-probe) | 404 | typed **`SESSION_NOT_FOUND`** "Session not found" | Mongo |
| DELETE /api/v2/auth/sessions/{bogus-oid} (sub-probe) | 404 | typed **`SESSION_NOT_FOUND`** — **the round-3 mislabel (bogus sid → `USER_NOT_FOUND`) is silently FIXED on current main** | Mongo |
| POST /api/v2/crawler/crawl/{id}/cancel on **PENDING** job (state-variant sub-probe) | 200 | "Crawler cancellation requested"; status `pending` → **`cancelling`** and never settles to a terminal state (see H1 — worker-side finalize blocked) | Redis job store (API-side write OK) + crawler worker (finalize broken) |
| [v1-deprecated] POST /api/v1/auth/register (fresh) | 201 | same contract as v2; unverified signup, canned message (F7 gate unchanged) | none |
| [v1-deprecated] POST /api/v1/auth/login (principal) | 200 | bearer, `expiresIn` 900 | none |
| [v1-deprecated] POST /api/v1/auth/refresh | 200 | rotation OK, new pair issued | none |
| [v1-deprecated] POST /api/v1/auth/logout | 200 | "Logged out successfully"; re-presenting the logged-out refresh token → 401 `INVALID_CREDENTIALS` "Session expired or revoked" (revocation propagates) | none |
| [v1-deprecated] POST /api/v1/auth/forgot-password | 200 | canned 200 — BUT the enqueued `auth.send_verification_email`/reset task ran for real: Resend **rejected** the example.com recipient (`email_send_failed`, status 4xx; typed `email_delivery_invalid`) and the task took the poison path `reject requeue=False` → `tasks.dlx` (see positives below) | Resend + RabbitMQ DLX |
| [v1-deprecated] POST /api/v1/auth/resend-verification | 200 | canned 200; same real-send rejection downstream | Resend |
| [v1-deprecated] PATCH /api/v1/users/{id}/role (admin) | 200 | role user→moderator persisted, "User role updated" | Postgres |
| [v1-deprecated] PATCH /api/v1/users/{id}/role (role=user) | 403 | `FORBIDDEN` "Missing required permission: users:write" | none |
| [v1-deprecated] POST /api/v1/users/{id}/impersonate (admin) | 200 | "Impersonation token issued" — `accessToken`, `expiresIn`, `impersonatingUserId` | Postgres |
| [v1-deprecated] POST /api/v1/users/{id}/impersonate (role=user) | 403 | `FORBIDDEN` "Insufficient role for this operation" (distinct typed message from the permission denial — two correct authz layers) | none |
| [v1-deprecated] PATCH /api/v1/users/{id}/deactivate (admin) | 200 | "User deactivated", `isActive:false` | Postgres |
| [v1-deprecated] PATCH /api/v1/users/{id}/activate (admin) | 200 | "User activated", `isActive:true` | Postgres |
| [v1-deprecated] GET /api/v1/auth/sessions (admin) | 200 | 14 sessions listed with sessionId/ip/ua/isCurrent (F9 fix holds) | Mongo |
| [v1-deprecated] DELETE /api/v1/auth/sessions/{non-current sid} | 200 | "Session revoked"; re-revoke of the same sid → 404 `SESSION_NOT_FOUND` (idempotency semantics clean) | Mongo |
| [v1-deprecated] DELETE /api/v1/auth/sessions (keep_current default TRUE) | 200 | "All other sessions revoked"; calling token stays valid (`/auth/me` 200) — the keep_current parameter works as documented | Mongo |
| [v1-deprecated] POST /api/v1/crawler/crawl | 202 | "Crawler job accepted", `status:pending` at the API — **but the job NEVER reaches a terminal state (H1)**; task `2f6404af…` received by the crawler worker within 1 s (routing to the crawler queue is correct), capability provisioning fails closed, task retries ×5, final failure, terminal write fails, job left zombie | crawler worker + `CRAWLER_EGRESS_PROXY_URL` |

Skipped per owner instruction (v1 deprecated; v2 equivalent probed in earlier
rounds): `GET /api/v1/crawler/crawl/{crawl_id}/chunks/search`,
`POST /api/v1/profile/change-password`.

### New defects

- **H1 (P1): crawler jobs can never reach a terminal state on current main —
  one missing env var (`CRAWLER_EGRESS_PROXY_URL`) breaks BOTH the crawl
  capability AND the terminal-failure write, leaving zombie `pending`/`cancelling`
  jobs.** Lifecycle (journal, `agentnexus-celery-crawler`, task
  `2f6404af-5630-4225-92f4-b9b6dbc89f2b`, crawl `288a07df…`): published →
  received in <1 s (crawler-queue routing is CORRECT, contra hint 2) →
  capability owner `_ensure_crawler_worker` fails with
  `ServiceUnavailableException 503: "Crawler egress boundary is misconfigured:
  CRAWLER_EGRESS_PROXY_URL must be configured" (capability: crawler_egress)`
  → ResilientTask retries ×5 (backoff 2 s → 6 s; `retry_count=5`) → final
  failure → the best-effort terminal write (`_mark_failed_sync`,
  `src/app/worker/jobs/crawler.py:146`) ALSO dies because
  `_create_worker_redis()` resolves the same egress policy
  (`egress.py:319` requires the proxy URL; settings default `None` not set in
  `.env.development`) and raises BEFORE any store call → job stays `pending`
  forever. API-side cancel (correctly) flips it to `cancelling`, but the
  worker-side finalize to `cancelled` is blocked by the same root cause —
  a second zombie state. G2's crawl4ai geolocation error is GONE (fix landed);
  this is a NEW, config-class root cause that REGRESSES the round-2/3
  behavior ("terminal `failed`, truthful reason") to "never terminal". The
  brief's money-path instruction applies: do NOT force; the fix is
  `CRAWLER_EGRESS_PROXY_URL` (+ relay cert files if a relay is declared) in
  the worker's environment, then re-probe.
- **H2 (P2, observability): the crawler terminal-write failure swallows its
  exception.** `crawler.py:151` logs `crawler_failure_write_failed` as a bare
  WARNING — the structured journal record carries `exception: null`, and the
  inner `logger.exception("Could not persist crawler failure")` never fires
  (the raise happens in `_create_worker_redis()`, outside `_mark_terminal`'s
  try). Root cause was recoverable only by reading source. Minor: the
  `retry_count=5` task-failure line reads like 5 real attempts at the crawl;
  all 5 were rejected at the egress boundary before any crawl work.

### Positive / silent-fix findings (verified this sweep)

- **Session-404 mislabel fixed**: bogus/revoked session DELETE now returns
  typed `SESSION_NOT_FOUND` (round 3 saw `USER_NOT_FOUND` — G-batch fix
  confirmed live).
- **DLX topology now declared**: default-worker boot logs "Dead-letter
  topology declared at worker boot | exchange='tasks.dlx'" (22:22:27 IST) —
  round-3 G6's absent DLX is addressed at declaration level. End-to-end
  retention (queue binding + a rejected message actually landing) is
  unobservable from this host (Railway broker, no rabbitmqctl access) —
  hypothesis-level only. The poison path IS exercised live: the
  [v1-deprecated] resend probe caused `auth.send_verification_email` → Resend
  4xx → `email_delivery_invalid` → `reject requeue=False` (the
  `ResilientTask` poison route ran end to end).
- **Task-time worker logging is visible for the crawler** (received/started/
  retry/failed lines with structured task_id/crawl_id fields) — round-3 G7
  materially improved at least on the crawler role.
- **Revoke-all semantics are exactly as documented** (keep_current default
  TRUE preserves the caller; FALSE kills the caller's session immediately,
  re-login restores).
- **Impersonation on v1 minted a correctly-scoped token**; both authz
  negatives produce distinct typed 403s (permission-denied vs role-denied).

### Counts

| Metric | Count |
|---|---|
| Live operations (swagger recount) | 138 |
| Excluded (owner instruction) | 16 — agent_saul (4) + OAuth/SSO (12) |
| In-scope operations | 122 |
| Already probed in rounds 1–3 (dedup) | 105 |
| Untested after deduplication | 17 (2 v2 + 15 v1) |
| Probed this sweep (untested-set members) | 15 (2 v2 + 13 v1 pre-amendment) |
| PASS (status/envelope/state correct at the API) | 15 |
| FAIL (backend round-trip wrong) | 1 — the crawl job zombie (H1); API-level 202/accept itself is correct |
| BLOCKED | 0 |
| SKIPPED per owner instruction (v1 deprecated) | 2 (v2 equivalents probed in earlier rounds) |
| Unexpected 5xx | 0 |
| 429s | 0 (all limiter classes paced within budget; no class exhausted) |
| Cross-tenant/privilege leaks | 0 (both authz negatives correctly denied) |

### Final coverage statement

**120 of 138 live operations probed across all four rounds** (120 of the 122
in-scope operations). Excluded per owner instruction: agent_saul router (4)
and OAuth/SSO routes (12). Skipped per owner instruction: 2 v1-only
operations whose v2 equivalents are probed. Every /api/v2 operation in the
live spec except the excluded sets has now been probed at least once.

### Cleanup

Test data left in infra: users `sweep4-20261119@example.com` (unverified),
`sweep4-agent@example.com` (registered → moderator → **deleted** via the
DELETE /api/v2/users probe); crawl `288a07df…` left in zombie `cancelling`
(terminal unreachable, H1 — do not treat as data corruption; it is the
reproducible evidence); admin sessions: 1 non-current session revoked
individually, 13 others revoked via keep_current=true, then a full
keep_current=false revoke — admin was re-logged-in afterward and holds one
fresh session; **principal tokens/sessions untouched and verified valid at
sweep end** (`/auth/me` 200). One Resend-verification task poisoned to the
DLX route (typed rejection, by design). No commits, no branch switches, no
source or `.env.development` modifications, no unit restarts by the tester.

## Billing addendum (2026-11-19) — the 7 subscription operations re-probed against a REAL provider-bound row

Round 3 probed all seven subscription-detail/lifecycle operations against the
stale orphan row `d15593de…` (`remote_sync:"permanent_failure"`, no provider
id) and correctly got typed rejections, but the evidence was confounded: a
stale row could not prove the rejections were the state machine working
rather than the broken binding. The orchestrator has since bound the round-2
"pending" row — `f0547bed-8873-48a8-8a8b-3201bcba6181`, whose Razorpay create
DID succeed server-side (`sub_U26C973nFdCMWK`, status `created`,
`notes.subscription_id` matching) but whose create response was lost to a
client timeout — directly (`razorpay_subscription_id=sub_U26C973nFdCMWK`,
`remote_sync=synced`). It is a genuine provider-bound subscription owned by
the admin identity. All 7 operations re-probed as admin, DTO shapes read from
`src/app/features/billing/subscriptions/dto.py` (PlanChangeDTO:
`new_plan_id` UUID + optional tz-aware `effective_date`; pause body optional
`pause_duration_days`; change-preview query `newPlanId`; trial-extension
query `days` 1–30 + optional `reason`). No commits, no restarts, no source
edits; billing-mutation limiter never approached (30/60).

| Method + path | Status | Observed behavior | Required backend piece |
|---|---|---|---|
| GET /api/v2/subscriptions/f0547bed… | 200 | full row: `status:"created"`, `razorpaySubscriptionId:"sub_U26C973nFdCMWK"`, `remoteSync:"synced"`, `planId:4df5c11f…` (sweep3-pro-monthly, synced `plan_U0HeO8T1ATOO7d`, active), `paymentUrl:null` (checkout short_url was ALSO lost with the create response — see H3), `version:1`, period/trial/pause fields all null | Postgres + Razorpay binding |
| GET /api/v2/subscriptions/{id}/payments | 200 | bare list, empty — truthful (subscription created but never authenticated/charged) | Postgres |
| GET /api/v2/subscriptions/{id}/change-preview?newPlanId=9261e883… | 422 | typed `VALIDATION_ERROR` "Proration is only valid for active subscriptions" — correct for `created`; second synced plan exists (smoke-pro-monthly-v2 `plan_U0MGyYSJ84rmhM`), so the payload was valid | Postgres |
| POST /api/v2/subscriptions/{id}/pause | 422 | typed `INVALID_STATE_TRANSITION` "Invalid state transition: created -> paused" — Razorpay-correct: pause requires an authenticated (active) subscription | Razorpay subscription lifecycle |
| POST /api/v2/subscriptions/{id}/resume | 422 | typed `INVALID_STATE_TRANSITION` "Invalid state transition: created -> active" | same |
| POST /api/v2/subscriptions/{id}/change-plan (PlanChangeDTO, target=synced 9261e883) | 422 | typed `VALIDATION_ERROR` "Invalid subscription state transition: created -> active" — correct rejection, but NOTE the envelope-code inconsistency: change-plan renders the same state rejection under `VALIDATION_ERROR` while pause/resume use `INVALID_STATE_TRANSITION` (H4) | same |
| POST /api/v2/subscriptions/{id}/trial-extension?days=7&reason=sweep4-probe | 422 | typed `VALIDATION_ERROR` "Subscription is not in trial" — correct (no trial on the row) | same |

Post-probe integrity check: re-GET shows `status:"created"`, `version:1`, no
pause/trial/cancel fields set — all five 422s were side-effect-free typed
rejections. Journal shows clean request handling, no errors.

**Verdict: the round-3 typed rejections were the state machine working
correctly, now proven against a real provider-bound row.** The remaining
prerequisite for pause/resume/change-plan/trial-extension/change-preview to
ever return 200 is **checkout authentication** (the customer completing the
Razorpay first payment, moving the sub `created → authenticated/active`) —
and that path is doubly broken here: the checkout `payment_url` was lost with
the create response and never bound locally (H3), so the API cannot even
direct the customer to authenticate. Owner instruction "leave it paused" is
**not achievable without forcing** — the state machine forbids `created →
paused` (and round 3 showed `created → cancelled` is also rejected, so a
created-but-unauthenticated subscription cannot be exited through the API at
all, only abandoned). Residue left as-is: an inert `created` row, provider
side mirrored.

### H3 (P2): reconciliation does NOT self-heal a provider-created subscription whose create response was lost

Evidence chain (owner-confirmed + this sweep's row state):

1. Round 2: `POST /api/v2/subscriptions` against a synced plan — the Razorpay
   provider create **succeeded server-side** (`sub_U26C973nFdCMWK`,
   `status:"created"`, `notes.subscription_id=f0547bed-8873-48a8-8a8b-3201bcba6181`)
   while the client timed out before the id could be bound.
2. The local row sat `created`/unbound (`razorpay_subscription_id:null`) for
   ~1.5 h while Razorpay held the live subscription.
3. `billing.reconciliation` **ran twice during that window and did not bind
   the row** — it does not discover provider subscriptions by
   `notes.subscription_id`, so the orphan persisted.
4. Recovery happened only via direct orchestrator binding
   (`razorpay_subscription_id=sub_U26C973nFdCMWK`, `remote_sync=synced`).
5. Residual loss visible in this sweep: the bound row's `paymentUrl` is
   `null` — the checkout short_url from the lost create response was never
   recovered either, so even after binding, the subscription cannot be
   authenticated through the API's own data.

Fix direction: reconciliation should list recent provider subscriptions (or
query by `notes.subscription_id`) and bind matching unbound local rows — id
AND checkout URL — turning "client timeout during create" from a permanent
orphan into a self-healing eventual consistency case. Related: round-3 G9b
(the same create path's persist-then-500 divergence) is the producer of this
orphan class.

### H4 (P3): state-transition rejections render under inconsistent error codes

`created → paused` and `created → active` (resume) render as
`INVALID_STATE_TRANSITION`; the semantically identical change-plan rejection
renders as `VALIDATION_ERROR` ("Invalid subscription state transition:
created -> active"), and trial-extension as `VALIDATION_ERROR` ("Subscription
is not in trial"). Same state-machine layer, three wire shapes — clients
cannot dispatch on one code. Cosmetic but contract-level.

### Counts (addendum)

| Metric | Count |
|---|---|
| Operations re-probed against the provider-bound row | 7 (all previously probed in round 3 against the stale orphan; coverage count unchanged at 120/138 — this upgrades the evidence, not the coverage) |
| PASS (detail/payments reads) | 2 (GET detail 200; GET payments 200 truthful-empty) |
| Typed rejections confirmed state-machine-correct | 5 (change-preview, pause, resume, change-plan, trial-extension — all 422, side-effect-free, version unchanged) |
| BLOCKED | checkout authentication (first payment) — prerequisite genuinely absent; also the only path by which pause/resume/change-plan/trial could ever return 200 |
| Unexpected 5xx | 0 |
| New defects | H3 (P2, reconciliation no-self-heal + lost payment_url), H4 (P3, error-code inconsistency) |
| Residue | subscription `f0547bed…` left `created`/`synced`/inert (pause requested by owner is unreachable without forcing — state machine forbids `created → paused`; recorded, not forced) |
