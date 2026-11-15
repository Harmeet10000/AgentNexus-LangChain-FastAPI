# crawler-worker-execution Specification

## Purpose
This capability ensures durable crawler jobs construct configuration accepted by the deployed Crawl4AI version and persist truthful outcomes without changing the existing job contract.
## Requirements
### Requirement: Crawler jobs use supported browser configuration
The crawler worker MUST construct browser and crawl configuration accepted by the deployed Crawl4AI version. An optional configuration value MUST NOT make every otherwise valid crawl fail before navigation.

#### Scenario: Configured geolocation is applied
- **WHEN** both configured geolocation coordinates are present
- **THEN** the worker passes a `GeolocationConfig` accepted by the installed Crawl4AI version into the real crawler run configuration

#### Scenario: Geolocation is absent or incomplete
- **WHEN** either configured coordinate is absent
- **THEN** the worker omits geolocation from the crawler run configuration

#### Scenario: Configuration contract changes upstream
- **WHEN** the installed Crawl4AI version no longer accepts the constructed run configuration
- **THEN** the focused contract test fails before deployment

### Requirement: Worker execution persists truthful terminal outcomes
The worker path MUST cross request validation, real crawler run-configuration construction, service result processing, and durable job persistence while preserving its existing capability ownership boundary.

#### Scenario: Controlled browser adapter returns a successful page
- **WHEN** a valid pending job is executed with the worker-owned browser capability replaced by a deterministic controlled adapter
- **THEN** the real crawler and service path constructs `CrawlerRunConfig`, persists a completed bounded result, and exposes the page through the existing owner-scoped store

#### Scenario: Target execution fails after valid configuration
- **WHEN** configuration construction succeeds but the controlled browser adapter reports a target failure
- **THEN** the job records a truthful terminal failure rather than a configuration-construction error

### Requirement: Existing crawl job contract remains compatible
The repair MUST preserve the existing public crawl API, task name, queue routing, request payload, job ownership, lifecycle, retries, and result retrieval contract.

#### Scenario: Existing crawler job is dispatched
- **WHEN** a valid crawl request is accepted by the API
- **THEN** the worker consumes the existing task payload without a schema migration

#### Scenario: Existing status and results are retrieved
- **WHEN** the worker completes or fails a crawl
- **THEN** status and bounded results remain available through the existing owner-scoped retrieval operations
