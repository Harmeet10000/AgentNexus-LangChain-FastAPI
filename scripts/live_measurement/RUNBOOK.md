# LIVE Measurement Run — Runbook

Everything needed to resume the live measurement run without re-deriving anything.
Last updated: 2026-11-09.

## Stack start (canonical commands)

| Process | Command | Notes |
|---|---|---|
| API | `cd /home/harmeet/Desktop/Projects/langchain-fastapi-production && nohup uv run uvicorn app.main:app --host 127.0.0.1 --port 5000 > /tmp/uvicorn.log 2>&1 &` | **No make target** (API ships in docker; local dev is manual uvicorn). Boot takes **6–7 min** (heavy import graph + cloud connects). Log: `/tmp/uvicorn.log`. |
| Worker | `cd /home/harmeet/Desktop/Projects/langchain-fastapi-production && nohup make celery > /tmp/celery_worker.log 2>&1 &` | `make celery` = `uv run worker --role default --loglevel=info`. Do NOT pass `--concurrency` (the worker entrypoint owns it — duplicate flag errors). Log: `/tmp/celery_worker.log`. |
| Beat (optional) | `make celery-beat` | Not needed for ingestion. |

Verify readiness: `curl http://127.0.0.1:5000/health` → **200** (it is `/health`, NOT
`/api/v1/health` — that 404s). The app serves requests before the log shows readiness.

## Credentials

- File: `.testing-credentials.json` (**gitignored**) — email, password, user_id,
  cookies for the testing principal `testing@live-measurement.dev`
  (user_id `6af208ec501517bb55b81c8b`, tenant `personal:<user_id>`).
- Regenerate: boot the API, then `uv run python scripts/live_measurement/testing_user.py`
  (idempotent; register tolerates 409/429; verification is seeded via Mongo
  `verification_token_hash` then flipped through the real `/auth/verify-email`).

## Rate limits (live API)

- register 3/60s · login 5/60s · **document upload 3/300s** — the ingest driver
  (`scripts/live_measurement/ingest_corpus.py`) bypasses the HTTP limiter by calling
  the production `DocumentCommandService.upload_document` in-process (still enqueues
  the real Celery task on CloudAMQP).

## Configuration changes made for the run (data-informed)

- `.env.development` `MAX_UPLOAD_SIZE=52428800` (50 MiB) — SRC-04 is 37 MB; the
  harden-2.4 10 MiB limit predated corpus measurement. This is the 2.4 feedback loop.

## Known gotchas

- **Transient Timescale LISTEN timeout kills the app**: the outbox/revocation
  LISTEN keepalive (`SELECT 1`, 1.0s budget) timed out once → graceful full
  shutdown at ~12 min uptime. Restart and proceed promptly; do not leave the API
  idle. (Resilience finding — record against harden.)
- `ruff format` sweeps reformat concurrent sessions' in-flight files — avoid
  tree-wide format; format touched files only.
- Golden chunk ids are a **corpus convention** (`SRC-04:p3:c01` — page-text blocks
  per `evals/golden/legal_retrieval_v2.CHUNKING.md`), NOT production chunk ids.
  Live scoring maps retrieved chunks → convention ids by text containment.
- Ingestion skip list: SRC-05/SRC-06 (originals intentionally not in-repo; zero
  golden rows reference them).
- Mapping file: `.live-ingestion-mapping.json` (source_id → doc_id/status) —
  written by the ingest driver, read by the scoring runner.

## Run artifacts

- Ingest driver: `scripts/live_measurement/ingest_corpus.py`
- Testing user: `scripts/live_measurement/testing_user.py`
- Reports land in `evals/reports/live_*.json`

## Flags (all default-off in src/app/config/settings.py — restore after runs)

- `DOCUMENTS_CONTEXTUAL_PREAMBLE_ENABLED` (G2)
- `DOCUMENTS_SECTION_HYDRATION_ENABLED` (G5)
- `DOCUMENTS_GRAPHITI_FUSION_ENABLED` (G4)

## Ingestion facts (learned 2026-11-10)

- **TWO workers are required**: `make celery` (default queue) AND `make celery-ingestion`
  (the ingestion dispatches to `CELERY_INGESTION_QUEUE` — the default worker never sees it).
- `DocumentUploadResponse.doc_id` is the JOB's id, **not** the document row id. The real
  document id: `documents.object_uri` contains the content sha256 — map via sha256 of the
  artifact file (done by `.live-ingestion-mapping.json` remap; hashes in /tmp/artifact_hashes.json).
- The object key is `<sha256>.<ext>` — friendly filenames are not preserved.
- Statuses: received → processing → parsed → stored_postgres → completed(_with_warnings)/failed.

## Ingestion failure finding (2026-11-11)

12/15 corpus documents terminally FAILED at extraction (`extraction_incomplete: true`),
sources: SRC-03/04/07/13/14/15/16 + SYN-01..05 (mixed formats AND sizes — the tiny .txt
SYN fixtures failed too, so this is systematic, not size-related). Succeeded: SRC-01,
SRC-02, SRC-12. Diagnosability gap: NO tracebacks in either worker's log — the
extraction failures left no worker-log trace, only the document status flag. A harden
task should make extraction failures persist a typed reason on the document record.
Retry hypothesis (unconfirmed): Gemini rate limits during the 15-doc burst; Celery
autoretry exhausted. Rerun the driver when resuming (content-hash duplicates skip).

## ROOT CAUSE of the 12/15 ingestion failures (2026-11-10)

**Gemini free-tier daily embedding quota exhausted** — 1000 embed-content
requests/day on `gemini-embedding-2`; the 15-document ingestion burst + Celery
autoretries burned it. Quota resets ~12h from first 429 (2026-11-10 ~09:00 UTC).

**Resume procedure (zero code changes needed):**
1. Ensure API + `make celery` + `make celery-ingestion` are running (RUNBOOK top).
2. `uv run python scripts/live_measurement/ingest_corpus.py` — idempotent; the
   12 failed documents re-ingest (content-hash duplicates skip the 3 successes).
3. `uv run python scripts/live_measurement/live_score.py` — the live baseline +
   per-class report (baseline pass already validated through chunk loading).
4. G2/G5/G4 comparisons per the task plan; tick roadmap 2.2/4.3 with numbers.

Provider note: switching embeddings to OpenAI is NOT viable without a schema
change — `chunks.embedding` is vector(768) and no OpenAI model emits 768 dims;
mixed-provider vectors are meaningless. The quota reset is the path.
