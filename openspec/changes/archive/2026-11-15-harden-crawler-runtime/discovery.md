# Discovery: Event Storm harden-crawler-runtime

## Actors

| Actor | System role |
|---|---|
| API client / agent | Requests a crawl and polls or retrieves its result |
| Crawler API | Validates and records the request, then dispatches the existing task |
| Crawler worker | Builds Crawl4AI configuration, runs the injected browser capability, and persists terminal state |
| Redis job store | Serves job state and paginated result reads |

## Commands (blue commands)

| Command | Actor | Payload essentials | Produces events |
|---|---|---|---|
| Create crawl job | API client / agent | URL, crawl options, owner; stable `crawl_id` assigned by API | No published event; enqueues the existing crawler task |
| Execute crawl | Crawler worker | Existing `crawl_id` and validated request payload | No published event; updates the Redis job read model |
| Read crawl status/results | API client / agent | `crawl_id`, owner, optional page/chunk cursor | None |

## Domain Events (orange stickies — past tense)

No domain events are published. Job lifecycle transitions remain Redis read-model updates, and this change adds no channel or payload.

## Policies (lilac — WHEN/THEN)

| WHEN | THEN | Kind (reactive / command / timer+expiry) |
|---|---|---|
| A crawl task is consumed with valid job state | The worker constructs supported Crawl4AI configuration and persists either bounded results or a truthful terminal failure | Command |
| An already-terminal or cancelled job is encountered | Existing idempotency and cancellation semantics win; the compatibility repair does not create a second lifecycle | Command |

## Read Models

| Read model | Serves | Derives from | Staleness tolerance |
|---|---|---|---|
| Redis crawl job metadata and bounded result | Status, ownership, page/chunk retrieval | Existing API creation and worker writes | Eventual until the worker writes a terminal state; no new consistency guarantee |

## Hotspots

| # | Question | Who decides | Resolved in |
|---|---|---|---|
| 1 | How can the regression test exercise the external configuration contract without uncontrolled internet access? | Design owner | `design.md`: construct the real run config and inject a deterministic browser adapter at the existing capability seam |
