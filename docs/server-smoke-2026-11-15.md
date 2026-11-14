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
