# Change: Restore crawler worker execution

**Class:** M

## Why

The 2026-11-15 live smoke showed that a queued crawl is consumed but fails before navigation because the installed Crawl4AI version rejects the configured geolocation shape. The worker needs a narrowly scoped compatibility repair and a deterministic regression test that crosses configuration, service orchestration, and durable result persistence without depending on the public internet.

## What Changes

- Normalize optional geolocation into the value type accepted by the installed Crawl4AI version and omit it when incomplete.
- Exercise the worker coroutine through the real `CrawlerRunConfig`, `WebCrawler`, `CrawlerService`, and `CrawlJobStore` seams with a controlled injected browser adapter.
- Preserve the existing crawl API, task name, queue routing, payload, ownership, result bounds, retries, and paginated retrieval contract.
- Keep connection-time SSRF enforcement out of this availability repair; `harden-crawler-egress` owns that independent deployment/security boundary.

## Capabilities

### New Capabilities

- `crawler-worker-execution` — durable crawler jobs construct supported Crawl4AI configuration and persist truthful terminal outcomes without changing their public or task contracts.

### Modified Capabilities

## Impact

- **Code**: crawler configuration adapter, worker execution path, and focused crawler tests.
- **APIs**: no endpoint, request, response, or ownership changes.
- **Dependencies**: no new runtime dependency.
- **Async flows**: the existing Celery crawler command remains wire-compatible; no event, task, or payload is added.
