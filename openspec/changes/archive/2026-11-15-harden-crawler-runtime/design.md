# Design: harden-crawler-runtime

## Context

The 2026-11-15 live worker smoke observed Crawl4AI rejecting a plain geolocation mapping before attempting navigation. Existing unit tests mocked above the external configuration constructor and therefore did not exercise the failing contract. This change repairs only worker execution compatibility; `harden-crawler-egress` separately owns connection-time SSRF enforcement and deployment isolation.

## Goals / Non-Goals

**Goals:**
- Normalize optional geolocation at the Crawl4AI adapter boundary.
- Pin the installed library contract by constructing the real `CrawlerRunConfig`.
- Exercise the worker coroutine through real service processing and Redis job persistence with deterministic I/O edges.
- Preserve every public and asynchronous crawler contract.

**Non-Goals:**
- Provide or claim a browser network-security boundary; that is `harden-crawler-egress`.
- Depend on an uncontrolled public website in the default test suite.
- Rework chunking, token accounting, cache policy, authentication, circuit breaking, metrics, or the agent-facing API.

## Decisions

### D1: Normalize geolocation at the external adapter boundary

**Choose:** Build Crawl4AI's real `GeolocationConfig` only when both coordinates exist, then pass it through the real `CrawlerRunConfig` constructor.
**Why:** The defect is a third-party shape mismatch. Correcting and testing the narrow adapter keeps library knowledge out of the worker and fails fast when the dependency contract changes.
**Alternative rejected:** Keep a plain mapping or catch the constructor failure. Both preserve the production outage while hiding its cause.

### D2: Test the real orchestration path with controlled I/O edges

**Choose:** Run `_run_crawl_job` through the real `WebCrawler`, `CrawlerService`, result conversion, and `CrawlJobStore`; inject a deterministic browser adapter that captures the real `CrawlerRunConfig`, and use an isolated Redis-compatible test store.
**Why:** This crosses the original failure boundary and persistence seam without making the default suite depend on internet reachability, DNS, Chromium installation state, or a third-party page.
**Alternative rejected:** Mock `CrawlerService.crawl`. That proves only worker persistence and bypasses the configuration constructor that failed in production.

### D3: Preserve the existing job contract

**Choose:** Retain task name, routing key, payload, Redis job model, endpoint shapes, ownership, limits, retry policy, and TTL.
**Why:** This is a worker adapter repair, not a client or data migration.

## Risks / Trade-offs

- [The controlled browser adapter drifts from Crawl4AI] → Require the test to receive the real installed `CrawlerRunConfig` and keep a deployment smoke as an operational check after dependency upgrades.
- [A deterministic integration test is mistaken for network-security proof] → Name the seam explicitly and keep all egress requirements in `harden-crawler-egress`.
- [The repair re-enables unsafe arbitrary-URL crawling before egress hardening] → Deploy only in trusted/restricted environments or keep public crawler acceptance disabled until the separate egress change is complete.

## Migration Plan

1. Land the adapter normalization and deterministic worker-path regression.
2. Run focused compatibility tests and the existing worker smoke in a trusted environment.
3. Preserve existing queued payloads; no data or task migration is required.
4. Roll back the adapter change if configuration construction regresses; do not broaden public crawler exposure while `harden-crawler-egress` remains incomplete.

## Blast Radius

| Symbol/module | Callers | Dependents |
|---|---|---|
| `CrawlerConfig.to_crawler_run_config` | `WebCrawler.crawl`, `WebCrawler.crawl_recursive` | Crawl4AI run-configuration construction |
| `WebCrawler.crawl` | `CrawlerService.crawl` | Single-page worker outcomes |
| `_run_crawl_job` | Celery `crawl_job` | Worker lifecycle and durable result writes |
| `CrawlJobStore` | Crawler routes and worker | Existing owner-scoped status and result reads |

## Async Semantics

| Question | Decision |
|---|---|
| Delivery semantics | Preserve existing Celery acknowledgement and bounded retry behavior; do not claim exactly-once execution. |
| Ordering guarantees | No ordering guarantee across jobs; stable `crawl_id` identifies retries of one job. |
| Expiry / retry policy | Preserve current retry/backoff and Redis TTL. |
| Payload schema versioning | No task payload or API schema change. |
| Read models while stale | Existing pending/running/terminal visibility remains unchanged. |
