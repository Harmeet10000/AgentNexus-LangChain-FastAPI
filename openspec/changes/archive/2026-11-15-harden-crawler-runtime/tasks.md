# Tasks: harden-crawler-runtime

## 1. Restore worker crawl execution

- [x] 1.1 Normalize configured geolocation to the value type accepted by the installed Crawl4AI version and omit it when unset.
      Implements: Crawler jobs use supported browser configuration / Configured geolocation is applied; Geolocation is absent or incomplete; Configuration contract changes upstream
      Channel: `crawlerTask` / `consumeCrawlerJob`
      verify: `uv run pytest -q tests/unit/test_crawler_hardening.py` — the real installed `CrawlerRunConfig` accepts configured geolocation and omits incomplete coordinates
- [x] 1.2 Add a deterministic worker-path regression that executes `_run_crawl_job` through real `WebCrawler`, `CrawlerService`, and `CrawlJobStore` seams, injects a controlled browser adapter that receives the real run configuration, and verifies a successful page is persisted.
      Implements: Worker execution persists truthful terminal outcomes / Controlled browser adapter returns a successful page
      Channel: `crawlerTask` / `consumeCrawlerJob`
      verify: `uv run pytest -q tests/integration/test_crawler_worker_execution.py::test_worker_persists_page_after_real_run_config_construction`
- [x] 1.3 Add the genuine-target-failure case after valid configuration construction and assert the stored failure describes target execution rather than configuration shape.
      Implements: Worker execution persists truthful terminal outcomes / Target execution fails after valid configuration
      Channel: `crawlerTask` / `consumeCrawlerJob`
      verify: `uv run pytest -q tests/integration/test_crawler_worker_execution.py::test_worker_records_target_failure_after_valid_configuration`

## 2. Preserve compatibility

- [x] 2.1 Verify task dispatch, idempotency, cancellation, ownership, status, retries, and paginated result retrieval remain compatible.
      Implements: Existing crawl job contract remains compatible / Existing crawler job is dispatched; Existing status and results are retrieved
      Channel: `crawlerTask` / `consumeCrawlerJob`
      verify: `uv run pytest -q tests/unit/test_crawler_job_bus_route.py tests/unit/test_crawler_jobs.py tests/unit/test_crawler_durable_additions.py tests/unit/test_crawler_worker_boundary.py`

## 3. Acceptance gate

- [x] 3.1 Run focused crawler tests and repository quality gates.
      Implements: Crawler jobs use supported browser configuration / all scenarios; Worker execution persists truthful terminal outcomes / all scenarios; Existing crawl job contract remains compatible / all scenarios
      verify: `uv run ruff format src/ tests/ && uv run ruff check src/ tests/ && uv run ty check src/ && uv run pytest && ast-grep scan src/` — the crawler feature contributes zero failures (full suite: 2802 passed, 1 skipped). The repository-wide red is pre-existing and outside this change: `ty` reports 15 diagnostics in `src/app/worker/celery/app.py` and `src/app/features/documents/**`, and `ast-grep` reports 42 findings across `documents`, `auth`, `agent_saul`, and `search`.
- [x] 3.2 Validate the OpenSpec change strictly.
      Implements: Crawler jobs use supported browser configuration / all scenarios; Worker execution persists truthful terminal outcomes / all scenarios; Existing crawl job contract remains compatible / all scenarios
      verify: `openspec validate harden-crawler-runtime --type change --strict`

## Task Dependency Graph

- **Wave 1:** 1.1
- **Wave 2:** 1.2, 1.3
- **Wave 3:** 2.1
- **Wave 4:** 3.1, 3.2
