# Manual Testing Plan

This is a hands-on test plan for the Agent Saul / AgentNexus API and its documented workflows. The companion [Postman collection](../postman/agentnexus-local.postman_collection.json) is generated from the development server's `/swagger.json` and contains all 123 HTTP operations exposed at generation time, grouped into 11 API tag folders. WebSocket, Celery, scheduled-task, and infrastructure behavior require the separate checks below.

## 1. Prepare a safe test environment

Use a local or dedicated development environment only. Do not run destructive, billing, webhook, impersonation, or rate-limit tests against production.

Before testing:

- Follow the repository README prerequisites and configure `.env.development`. PostgreSQL is required; the schema requires `vector`, `vectorscale`, `pg_trgm`, and `pg_textsearch`. Bring the database to the expected provisioned revision before starting the app.
- Start the API with the README command: `uv run uvicorn app.main:app --reload --reload-dir src --host 0.0.0.0 --port 5000 --no-access-log`. Startup may take about a minute while connections and application graphs initialize.
- Start RabbitMQ and the Celery workers for ingestion, crawler, email, billing, and outbox workflows. Use the README's separate `default`, `ingestion`, and `crawler` queues; start Celery Beat for scheduled jobs.
- Configure Gemini and the integrations needed for the feature being tested. Redis, MongoDB, and Neo4j may degrade at boot, but tests that exercise their features require them. Verify object storage and email configuration before upload/avatar/verification flows.
- Have a small, non-sensitive contract fixture in a supported format (PDF, TXT, DOCX, XLSX, PPTX, Markdown, or HTML), under the documented 10 MB upload limit. Keep a second byte-identical copy for duplicate-upload checks and one unsupported/oversized file for validation checks.
- Use disposable users and development payment-provider credentials. Keep a verified user token, an admin token (provisioned through the project's approved local setup), and sample resource IDs available.

Basic readiness checks:

1. `GET {{baseUrl}}/` returns HTTP 200 and a healthy root payload.
2. `GET {{baseUrl}}/health` and `GET {{baseUrl}}/api/v1/health/ready` report the expected dependency state. A degraded optional dependency should be visible; PostgreSQL must be healthy.
3. `GET {{baseUrl}}/api/v1/health/self` and the equivalent v2 endpoint respond. `GET {{baseUrl}}/metrics` returns Prometheus text.
4. `GET {{baseUrl}}/swagger.json` returns the OpenAPI document. The app configures docs at `/api-docs`, ReDoc at `/api-redoc`, and OpenAPI at `/swagger.json` (not `/openapi.json`).

## 2. Import and configure Postman

1. In Postman, import `postman/agentnexus-local.postman_collection.json`.
2. Open the collection variables and set `baseUrl` to `http://localhost:5000` (no trailing slash), plus the test `email` and `password`.
3. Run the auth registration/login flow first. Successful login and refresh responses automatically save `access_token` and `refresh_token` to collection variables. Protected requests inherit Bearer auth; public requests are explicitly unauthenticated.
4. Replace each `replace-with-*` path variable with a real ID from a successful response. The collection saves common `doc_id`, `crawl_id`, `thread_id`, and `ws_url` values when present.
5. Edit sample request bodies to use valid domain values. They are schema-derived scaffolds, not guaranteed business-ready payloads. For multipart requests, select the local fixture in the `file` form field.
6. Run the collection in the order below, or execute an individual folder after its prerequisites exist. Collection tests fail on HTTP 5xx and verify that an API envelope's `statusCode` matches the actual HTTP status.

The collection includes both v1 and v2 operations where exposed. v1 is deprecated and returns deprecation headers; exercise v2 for new integrations and retain v1 checks for compatibility until its documented sunset.

## 3. Core workflow tests

### A. Authentication and account lifecycle

1. Register a unique disposable account. Verify the success envelope and creation status. Try an already-used email and malformed email/password to check conflict and validation errors.
2. Complete email verification using the development email/link or the project's configured verification path. Check resend-verification and forgot-password behavior with a controlled test inbox; do not expect the API to reveal whether an arbitrary email exists.
3. Log in with valid credentials. Confirm the access and refresh tokens are returned, the access cookie is set, and `GET /api/v2/auth/me` returns the correct principal. Repeat with an incorrect password and with no token.
4. Refresh once and confirm the newly issued tokens work. Try an expired/revoked refresh token and confirm a stable 401 envelope. Logout and confirm the refresh token can no longer be used.
5. List sessions; revoke one session; revoke all remaining sessions. Confirm refresh tokens from revoked sessions fail. Check that a stale access token follows the documented expiry/revocation behavior.
6. Check OAuth authorize for Google and GitHub in development. Verify state cookie creation and invalid provider/state handling. Complete callback only with provider credentials and redirect settings configured; avoid sharing callback URLs containing tokens.

### B. Authorization, users, profile, and audit-sensitive actions

1. With a normal user token, read and update the caller's profile, change the password, and upload an avatar. Verify persistence after a fresh read; test invalid current password and invalid avatar type/size.
2. With a normal user token, try user listing, role changes, activation/deactivation, deletion, and impersonation. These should be denied unless that role is explicitly permitted.
3. With an admin token, list users and exercise `GET/DELETE /users/{user_id}`, role update, activate/deactivate, and impersonation against disposable accounts. Verify state changes and that tokens/sessions behave as documented after role or status changes.
4. Exercise session export with an authorized owner and an unrelated user. Confirm the latter cannot read another user's transcript.

### C. Contract upload and ingestion

1. Upload a small valid contract via `POST /api/v2/documents/upload`. Check the 201 response, document ID, task ID/status, and warnings. Also exercise the v2 ingestion upload route; the collection contains both document and ingestion upload operations.
2. Poll `GET /documents/{doc_id}/status` until `completed` or `completed_with_warnings`. Confirm the worker consumes the ingestion queue and final status includes title, document kind, chunk count, verified chunk count, and typed warnings.
3. Upload byte-identical content again. Verify it reports a duplicate and preserves the original document identity rather than creating duplicate chunks.
4. Try empty content, unsupported extension, a file over 10 MB, and a missing file field. Expect a validation envelope and no queued work.
5. Try another user's document ID with a valid token, then try the status endpoint without a token. Verify ownership and authentication boundaries.
6. If document parsing or persistence fails, inspect the Celery task result/logs and document status; confirm retryable failures do not silently appear as successful ingestion.

### D. Search and grounded legal answers

First upload and finish ingesting a known contract so the test corpus is predictable.

1. `POST /api/v2/search`: search for a unique clause term. Verify ranked items, scores, document IDs, cache behavior, and metadata filters. Change each of jurisdiction, contract type, clause type, parties, document IDs, and graph-verification filters. Confirm unknown fields and invalid limits return 422.
2. `POST /api/v2/search/rag`: ask a question tied to the fixture. Verify answer, citations/context, confidence, warnings, and cache flag. Try `bypass_cache` and a bounded larger token budget.
3. `POST /api/v2/search/ask`: ask a corpus-only question. Compare citations and behavior to `/search/rag`.
4. `POST /api/v2/legal/ask`: repeat with graph verification enabled. Verify only graph-verified evidence qualifies; empty results under strict filtering are acceptable when there is no verified evidence.
5. Repeat each call without auth, with an invalid token, with an empty query, and with filters that match no documents. Verify standard error envelopes and no uncited answer claims.

### E. Agent Saul, resumability, and human review

The HTTP collection covers session creation and transcript export. Use Postman's WebSocket request UI (or `websocat`) for the actual stream; WebSocket messages are not HTTP operations in `/swagger.json`.

1. Create a session with an ingested `doc_id` and a focused legal `user_query`. Save the returned `thread_id` and `ws_url`.
2. Connect to `ws_url` and send a `start` frame with the same document and query. Confirm `node_start`/`node_end`, state updates, and token frames arrive and progress through the documented pipeline.
3. At `hitl_interrupt`, read the complete payload. Test each applicable gate: `approve` with reviewer role/feedback, `reject`, and `modify` with a corrected plan or clause override. Confirm analysis does not persist before approval and that the audit state records feedback and reviewer role.
4. Disconnect mid-run, reconnect with the same thread ID, and confirm checkpoint-based resumption. Test keepalive `ping`/`pong`.
5. Send malformed first frame, mismatched thread ID, invalid resume action, and an invalid document ID. Confirm close/error codes match the workflow guide (4000, 4001, 4002, or 1011 for unexpected server faults) and retryability is meaningful.
6. Export the completed transcript. Verify access control and that the completed run contains the decisions and final summary. Do not treat model correctness as a binary API assertion: manually compare findings and citations against the fixture.

### F. Crawler and web research

1. Check `GET /api/v2/crawler/rate-limit` before testing. Search a stable public query with `GET /crawler/search`; verify result URLs, scores, and configured answer behavior.
2. Submit a crawl of a stable, permitted public page with shallow depth. Inspect pages, chunks, chunk search, cache behavior, and crawl status using the returned crawl ID. Try cancellation while a crawl is still active.
3. Exercise markdown, text, HTML, summary, structured extraction, depth/page limits, and cache bypass one at a time. Structured extraction requires the configured model. Avoid sites that prohibit automated access.
4. Test malformed URL, unsupported schema, excessive depth/page count, and timeout bounds. Verify the error envelope and `Retry-After` behavior for rate limiting; do not exhaust limits casually.

### G. Billing, credits, and webhooks (sandbox only)

Use disposable users and provider sandbox/test keys. Billing mutations can trigger real side effects if configured against live services.

1. Create/list/get/update/archive a test plan. Verify archived plans cannot be purchased or mutated where policy forbids it.
2. Create a subscription for a test user; list and fetch it. Exercise change preview, plan change, pause, resume, cancel, and trial extension. Verify legal state transitions, idempotency, and conflict responses.
3. List subscription payments; fetch a payment; refund a sandbox payment; inspect invoices and void only a disposable invoice.
4. Grant credits, read balance and history, and apply credits to a test invoice. Check that balances and ledger history reconcile after each mutation and that cross-user IDs are denied.
5. Submit a valid signed Razorpay sandbox webhook, an invalid signature, a duplicate event, and an out-of-order/retry case. Confirm idempotent event handling; replay a failed event only through the admin operation.
6. Read dunning configuration and run dunning manually only with test records. Check worker logs/outbox events and prevent duplicate reminders or charges.

## 4. Exhaustive endpoint sweep

The Postman collection is the route-by-route checklist: it contains every GET, POST, PUT, PATCH, and DELETE operation found in the running OpenAPI schema (123 operations at generation time) under these folders:

- `Root`, `Monitoring`, and `health`
- `auth`, `users`, and `profile`
- `documents` and `ingestion`
- `Crawler`
- `agent-saul`
- `billing` (plans, subscriptions, payments, invoices, webhooks, admin, and credits)

For each request:

1. Supply a valid owner/admin token and real IDs for the success path; use disposable data for mutations.
2. Record the HTTP status, `success`, `statusCode`, `error.code`, relevant headers, and resulting state. For v1, also verify `Deprecation`, `Sunset`, and successor `Link` headers.
3. Run a negative variant for protected operations (no token, invalid token, wrong role, wrong owner, nonexistent ID) and a validation variant for JSON requests (missing required field, wrong type, invalid enum/range). Do not send malformed traffic in a tight loop.
4. Mark endpoints requiring provider sandbox, email access, object storage, worker availability, or pre-existing resources as blocked when that prerequisite is unavailable; do not interpret infrastructure failure as an endpoint pass.

## 5. Cross-cutting behavior and non-HTTP features

- **Response contract:** every API JSON success/error uses the documented envelope; verify the HTTP status mirrors `statusCode`, error codes are stable, and correlation IDs appear in logs. Raw metrics and binary/file responses are exceptions.
- **Versioning:** compare v1 and v2 behavior for paired routes; verify v1 deprecation headers and v2-only billing paths.
- **Rate limiting:** test a single controlled limit in a disposable environment. Expect 429 and `Retry-After`; honor it and do not trigger IP auto-ban on a shared network.
- **Cache/idempotency:** repeat identical search, crawl, upload, webhook, and billing requests; verify documented cache/duplicate behavior and use bypass options to distinguish a cache hit from stale data.
- **Celery/outbox:** stop and restart a worker in development, observe queued/retried/failed work, then verify delivery and dead-letter behavior. Check that queue consumers use the intended `default`, `ingestion`, and `crawler` queues and that the outbox relay does not publish duplicates.
- **Scheduled work:** run Celery Beat in development and verify configured periodic tasks with test data only (dunning, credit expiry, cleanup/sweeps). Do not wait for production schedules or use live customer data.
- **Security:** test ownership isolation, RBAC, refresh-token revocation, webhook signatures, upload constraints, request validation, and rate limits. Never paste real tokens into screenshots or bug reports.

## 6. Record results

For each run, record date/commit, environment, user role, request name, expected result, actual status/result, correlation ID, worker/provider prerequisites, and pass/fail/blocked. Attach sanitized request/response excerpts and relevant worker logs. A manual pass means both the response contract and the resulting persisted/background state are correct; a 2xx response alone is not sufficient.
