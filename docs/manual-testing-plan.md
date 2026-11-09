# Manual API Testing Plan

Use this plan to run a repeatable local or dedicated-development test of the
AgentNexus API and Agent Saul workflow. The companion
[Postman collection](../postman/agentnexus-local.postman_collection.json) is the
request inventory; this document is the execution order and the pass/fail
standard.

The collection was reconciled against the current app's generated OpenAPI
schema during this update: **138 unique HTTP operations** across v1 and v2.
WebSocket frames, workers, and scheduled jobs are separate checks. The checked-in
[`docs-site/openapi.json`](../docs-site/openapi.json) can lag the running app;
when the inventories disagree, use the app's `/swagger.json` when enabled and
confirm behavior against the router before treating a route as current.

## 1. Prepare the environment

Use local or dedicated test infrastructure only. Never run destructive account,
tenant, billing, webhook, impersonation, or rate-limit tests against production.

1. Follow the [README quickstart](../README.md#quickstart). Use Python 3.12 and
   `uv`; bring up the local database and RabbitMQ, apply the expected provisioned
   schema, and configure `.env.development`. Do not assume a brand-new database
   can traverse the documented Alembic lineage.
2. Start the API:

   ```bash
   uv run uvicorn app.main:app --host 0.0.0.0 --port 5000
   ```

   Startup took about 60–80 seconds in the last recorded server smoke. Wait for
   the startup result and readiness checks; an HTTP process listening on the port
   is not enough.
3. Start only the workers needed for the scenarios in this run:

   ```bash
   make celery-ingestion
   make celery-crawler
   ```

   Start `make celery` for default-queue work and `make celery-beat` only when
   testing scheduled jobs. Keep each worker in its own terminal and record which
   roles are running.
4. Configure the providers and services required by the chosen scenarios:
   PostgreSQL is required for core data flows; Redis, MongoDB, object storage,
   email, model access, Neo4j/Graphiti, and RabbitMQ are required only for
   scenarios that exercise them. Do not mark an endpoint failed solely because
   its declared external prerequisite is absent.
5. Prepare:
   - A disposable, verified user with an active tenant, plus an administrator
     identity for admin-only scenarios.
   - A small non-sensitive `.txt` or PDF contract fixture accepted by the current
     upload allowlist. Keep a second identical copy for deduplication and one
     unsupported file for validation. Do not use real customer or legal data.
   - A private Postman environment with the variables in §2. Keep tokens and
     provider secrets out of the collection, screenshots, and reports.

### Known setup blockers to re-check

The [2026-10-28 server smoke](./server-smoke-2026-10-28.md) recorded that fresh
registrations might not receive a personal tenant (F4), SMTP might not be
configured for email verification (F5), an ingestion failure could lack a
surfaced reason (F2), and Agent Saul initialization could fail after a transient
startup error (F6). Check whether these remain true on the branch under test.
If registration, verification, tenant setup, or a provider is unavailable,
record that scenario as **Blocked** and use an already-provisioned disposable
test account; do not silently count it as a pass or run an unreviewed data
migration to force it through.

## 2. Configure Postman

1. Import `postman/agentnexus-local.postman_collection.json`.
2. Create/select a private environment and set:
   - `baseUrl`: `http://localhost:5000` (origin only, no trailing slash).
   - `email`: a unique disposable address accepted by the app's email validator.
     The last smoke found reserved test-TLD addresses rejected.
   - `password`: a disposable value meeting the configured password policy.
   - `access_token`, `refresh_token`, `admin_access_token`: leave unset until
     login; never use a real account credential.
   - `allowClientError=false` and `allowDestructive=false` for the normal pass.
3. Successful login/refresh requests populate token variables. The collection
   also captures common IDs when the response shape exposes them. For the other
   placeholders, set the value from a successful response before sending the
   dependent request: `doc_id`, `thread_id`, Agent Saul `session_id`,
   `auth_session_id`, `ws_url`, `crawl_id`, `tenant_id`, `action_token`, and
   billing IDs. Treat OAuth `action_token` as a secret.
4. In multipart upload requests, change/select the `file` form field as a **File**
   and choose the fixture. Do not manually set `Content-Type`; Postman must add
   the multipart boundary.
5. Run the stages below in order. **Do not use “Run collection” as a single
   end-to-end pass.** The inventory includes both API versions, state-changing
   operations, provider-only flows, and requests that require IDs created by
   earlier requests. Run v1 compatibility separately after the v2 workflow.
   The Postman folders follow this stage order: Root → Monitoring → health →
   auth → users → tenants → profile → ingestion → documents → search →
   agent-saul → Crawler → billing. Mixed-version folders list v2 requests before
   v1 compatibility requests.
6. To test a negative case, set `allowClientError=true` only for that request,
   send it, inspect the exact status and error code, and immediately restore it
   to `false`. The Postman script treats any 4xx as an intentional client error;
   that alone does not prove the *correct* failure occurred.
7. Leave `allowDestructive=false` until deliberately testing a disposable
   mutation. Turn it on only for the specific isolated operation, then turn it
   off again. The collection skips guarded operations by default.

## 3. Run the core workflow in this order

Use v2 for the main pass. Record each step before moving on; do not skip a
prerequisite and then interpret a dependent failure as an API defect.

| Order | Request(s) | Prepared request / action | Pass evidence |
|---|---|---|---|
| 0. Service gate | `GET /`, `GET /health`, `GET /api/v2/health/live`, `GET /api/v2/health/ready`, `GET /api/v2/health/self`, `GET /metrics`, then `GET /swagger.json` if enabled | No body. Start with root and health before any write. | HTTP 200 where documented; PostgreSQL healthy; optional dependency degradation is explicit. Metrics is Prometheus text. OpenAPI may be disabled in a given environment. |
| 1. Account | `POST /api/v2/auth/register` → `POST /api/v2/auth/verify-email` → `POST /api/v2/auth/login` → `GET /api/v2/auth/me` | Use the bodies in §4. If email delivery is unavailable, stop and use a pre-verified test account. | Registration returns 201; verification succeeds; login returns tokens; `/me` identifies the same user. Account and tenant prerequisites are real, not inferred from a 2xx. |
| 2. Authorization/profile | `GET/PATCH /api/v2/profile/`; `POST /api/v2/profile/change-password`; admin-only `GET /api/v2/users/`, `GET /api/v2/users/{user_id}`, tenant role/settings routes | Use user token for self-service and admin token for admin/tenant operations. Use a disposable user for mutations. | Caller can read its own profile; unauthorized role/tenant changes are denied; authorized changes persist on a fresh read. |
| 3. Upload | `POST /api/v2/documents/upload` | Multipart `file` fixture. Save `doc_id` and `task_id` from the response. | 201 with a document ID and queued/initial status. A byte-identical repeat returns duplicate semantics and the original identity. |
| 4. Ingestion completion | Poll `GET /api/v2/documents/{doc_id}/status` every 5 seconds, for up to 2 minutes for the small smoke fixture. | Use the ID from step 3. Do not upload repeatedly to “see if it works”; uploads are rate limited. | Terminal status is `completed` or `completed_with_warnings`; inspect warning details and verify title, document kind, chunk count, verified chunk count, and persisted state. If dependencies are healthy but no terminal state appears by the cap, mark failed and capture worker evidence. |
| 5. Retrieval and answer | `POST /api/v2/search` → `POST /api/v2/search/rag` → `POST /api/v2/search/ask` → `POST /api/v2/legal/ask` | Use the fixture-specific query bodies in §4 and restrict to `doc_id`. | Search results are scoped to the fixture. Answers include evidence/citations and disclose empty or unverified evidence; do not accept unsupported claims as a pass. Empty strict graph-verified results are valid when no verified evidence exists. |
| 6. Agent Saul | `POST /api/v2/agent-saul/sessions` → connect to returned `ws_url` → send the `start` frame → handle each `hitl_interrupt` with a `resume` frame → `GET /api/v2/agent-saul/sessions/{thread_id}/export` | Use §4. The transcript export path parameter is the Agent Saul `thread_id` (the session identifier for this workflow). | WebSocket produces typed progress frames and reaches a terminal `done` or a typed, actionable `error`; approval/rejection/modify decisions resume the same thread; export is ordered and ownership-checked. |
| 7. Crawler | `GET /api/v2/crawler/rate-limit` → `GET /api/v2/crawler/search` → `POST /api/v2/crawler/crawl` → status/pages/chunks/chunk-search routes | Check remaining quota first. Use a stable, permitted public URL and the body in §4. | Crawl reaches a terminal state; pages/chunks match the returned `crawl_id`; cancellation is tested only while a disposable crawl is active. Honor `Retry-After`. |
| 8. Billing (optional) | Plans → subscriptions → payments/invoices → credits/dunning → webhooks | Sandbox credentials and disposable customer/data only. Use the existing collection examples; no live payment provider. | State transitions are valid; ledger balances reconcile; webhook signature/idempotency behavior is correct. If sandbox setup is absent, mark the whole scenario blocked. |
| 9. v1 compatibility | Run paired v1 requests needed by existing clients after the v2 pass; do not repeat registration with the same email. | Use existing test identities and a fresh fixture if needed. | Expected response behavior plus `Deprecation: true`, `Sunset`, and successor `Link` headers. The current documented sunset is 2027-06-15. |

### Endpoint order within the dependent workflows

- **Auth:** register → verify-email → login → me → list sessions → refresh.
  Keep logout and session revocation until cleanup; they invalidate credentials.
- **Profile and access control:** read profile → update profile → upload avatar
  (if object storage is configured) → change password last. With an admin token,
  list users → read the disposable target → change role or deactivate/activate
  using separate disposable users → impersonate only when explicitly permitted
  → delete a disposable user last. Test tenant legal-role and self-review
  settings with an administrator, then verify the returned state and restore the
  original setting. Never use your only test account for these mutations.
- **Optional auth recovery/OAuth:** test resend-verification and forgot/reset
  password only with a controlled inbox and a disposable account. For OAuth
  linking, authorize → complete the provider redirect → preview the fresh
  `action_token` → confirm → verify `/me` → unlink only if another login method
  remains. Do not copy callback URLs or tokens into reports.
- **Documents:** upload → poll status → duplicate upload → search/answer. For the
  deprecated compatibility route, use
  `POST /api/v2/ingestion/documents/upload` only as a migration check; compare
  deprecation headers and successor link. The canonical route is
  `POST /api/v2/documents/upload`.
- **Search service (optional):** `POST /api/v2/search/ingest` → poll
  `GET /api/v2/search/ingest/{task_id}` → `POST /api/v2/search/hybrid`.
  This is distinct from document upload and requires its own valid request data.
- **Crawler:** rate-limit → web search → create crawl → status → pages → chunks
  → chunk search; cancel only an in-progress disposable crawl.
- **Billing:** list/create/get/update plan → create/list/get subscription →
  change-preview/change-plan → pause/resume/trial-extension as valid transitions
  → inspect subscription payments and invoices → apply/grant/inspect credits →
  test sandbox webhook signature/idempotency and admin replay → refund, void,
  dunning, cancel, and archive last. Do not run every transition on one
  subscription; use a fresh disposable record for each terminal path.

## 4. Copy/paste request examples

Postman requests are pre-populated; these are minimal known-good starting
payloads. Replace variables with values from the private environment and tailor
the query to the fixture.

### Auth

```json
{
  "email": "{{email}}",
  "password": "{{password}}"
}
```

Use that body for registration and login. Verification:

```json
{
  "token": "<token-from-the-development-verification-link>"
}
```

Refresh:

```json
{
  "refresh_token": "{{refresh_token}}"
}
```

### Search and grounded answers

`POST /api/v2/search`:

```json
{
  "query": "What does this agreement say about termination notice?",
  "limit": 5,
  "candidate_limit": 25,
  "metadata_filter": {
    "document_ids": ["{{doc_id}}"]
  },
  "bypass_cache": true
}
```

`POST /api/v2/search/rag` accepts the same fields plus `max_tokens` (for example
`2000`). For `/api/v2/search/ask` and `/api/v2/legal/ask`, use
`{"query":"...","doc_ids_filter":["{{doc_id}}"],"bypass_cache":true}`. To test
empty results, use a real but unrelated document ID or a query absent from the
fixture; do not remove authorization and mistake an error for an empty result.

### Agent Saul HTTP preflight

```json
{
  "doc_id": "{{doc_id}}",
  "allowed_doc_ids": ["{{doc_id}}"],
  "primary_doc_id": "{{doc_id}}",
  "user_query": "Review the agreement's termination and notice provisions. Cite the exact supporting text."
}
```

Connect to the returned `ws_url` and send:

```json
{
  "type": "start",
  "thread_id": "{{thread_id}}",
  "allowed_doc_ids": ["{{doc_id}}"],
  "primary_doc_id": "{{doc_id}}",
  "user_query": "Review the agreement's termination and notice provisions. Cite the exact supporting text."
}
```

At a human-review interrupt, use the actual thread ID from the frame:

```json
{
  "type": "resume",
  "thread_id": "{{thread_id}}",
  "action": "approve",
  "feedback": "Approved for this disposable test run."
}
```

Use `reject` or `modify` only in separate runs with the matching interrupt
payload. The client-provided `reviewer_role` is not authorization.

### Crawler

```json
{
  "url": "https://example.com/",
  "mode": "markdown",
  "max_depth": 1,
  "max_pages": 1,
  "bypass_cache": true,
  "timeout": 30
}
```

Replace the URL with a page you are permitted to crawl. Structured extraction
requires `extract_structured=true` plus a valid `schema_type` or
`custom_schema`; do not turn it on for the basic smoke request.

### Upload

Upload is multipart, not JSON: choose the local fixture in the form field named
`file`. Leave `Content-Type` unset so Postman generates the multipart boundary.
The current upload size/extension policy is configured by the app; do not rely
on a stale hard-coded limit.

## 5. Negative, authorization, and lifecycle checks

Run these after the corresponding happy path, one at a time:

| Probe | Expected result | Evidence to record |
|---|---|---|
| Protected request without token / invalid token | 401 | Stable error envelope and code; no data returned. |
| Normal user tries an admin-only operation | 403 (or the documented concealment response) | No state change; verify using an authorized fresh read. |
| Cross-user/tenant resource ID | Denied (commonly 403 or deliberate 404) | No foreign data in body, logs, export, or later reads. |
| Duplicate email / conflicting state transition | Conflict response documented by the route | Error code is stable; original state remains intact. |
| Missing field, wrong type, invalid enum/range, empty query | 422 | Field-level validation is useful; no background job or mutation was created. |
| Unknown resource ID | 404 or documented authorization-safe response | No cross-tenant existence leak. |
| Invalid/duplicate webhook signature/event | 4xx for bad signature; duplicate event is idempotent | No payment/ledger mutation; event trace is auditable. |
| Rate limit | 429 with `Retry-After` | Wait the advertised duration; do not probe repeatedly or from a shared IP. |

Assert the route's actual documented status and `error.code`; the Postman
negative-test switch only proves “some 4xx” and is not sufficient by itself.
Never use a live user, real billing key, or broad request loop for negative
testing.

## 6. Pass/fail rules and metrics

For each request, a pass requires all applicable checks below—not just a 2xx:

1. The actual HTTP status matches the route's expected status.
2. For JSON API envelopes, `statusCode` equals the HTTP status and `success`
   agrees with whether the status is 2xx. Error responses include a stable
   `error.code`. Raw metrics, NDJSON transcript exports, redirects, and file
   responses are checked by their own content type/headers.
3. The returned state is correct after a fresh read, and any background task
   reaches its expected terminal state.
4. Authorization boundaries, tenant/document scope, and idempotency behave as
   expected.
5. Warnings are inspected and classified; `completed_with_warnings` is not
   automatically a clean pass.

Use these result labels:

- **PASS:** prerequisites were available and all expected response, state, and
  security checks passed.
- **FAIL:** prerequisites were available but status, payload, state, timing
  budget, or isolation was wrong.
- **BLOCKED:** a required service/account/provider/fixture was unavailable. State
  the missing prerequisite; do not count as a pass.
- **NOT RUN:** intentionally out of scope for this run.

Report:

- `pass_rate = PASS / (PASS + FAIL)`; keep BLOCKED and NOT RUN visible and out of
  the denominator.
- `coverage = attempted planned scenarios / enabled planned scenarios`.
- Count unexpected 5xx, incorrect 4xx/error codes, state mismatches, and
  ownership/RBAC failures separately. Any cross-tenant data exposure or
  unintended real-world side effect is an immediate stop.
- Record response time for each request and the total duration for ingestion,
  search, and Agent Saul. No latency SLA is defined by this manual plan; compare
  with a previous run rather than inventing a threshold.

| Run date / commit | Stage / request | Role | Expected status | Actual status | Duration | Error code / correlation ID | Persisted or async result | PASS / FAIL / BLOCKED / NOT RUN |
|---|---|---|---:|---:|---:|---|---|---|
| | | | | | | | | |

Capture sanitized response excerpts and relevant API/worker logs. Never include
tokens, passwords, OAuth action tokens, webhook signatures, contract text, or
customer data.

## 7. Cleanup and tester shortcuts

- Finish token/session revocation and destructive checks only after all other
  requests that need those credentials.
- Use a new, unique fixture per fresh ingestion run. Reusing the exact bytes
  intentionally tests deduplication and should return the original `doc_id`.
- Keep one Postman environment per run and clear stale IDs before a new run.
  Collection variables can outlive a request; a stale `doc_id` can make a later
  test look successful against the wrong document.
- In Postman Console, check why a guarded request was skipped before marking it
  not run. Placeholder values such as `replace-with-*` are not real test data.
- If upload reaches `failed`, save the document ID, task ID, status payload, and
  worker logs. The last smoke found that the API could omit the underlying
  failure reason; absence of an error detail is not evidence that the worker
  succeeded.
- If Agent Saul is absent after startup, check its startup logs and readiness
  before retrying. Do not reuse a session after a typed terminal error unless the
  workflow explicitly supports resumption.
- Record the app commit, environment type, worker roles, provider readiness, and
  fixture hash in each run. Keep manual results reproducible without storing
  sensitive fixture contents.
