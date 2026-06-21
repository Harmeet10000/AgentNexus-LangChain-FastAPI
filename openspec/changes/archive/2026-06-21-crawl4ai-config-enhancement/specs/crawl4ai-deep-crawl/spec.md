## ADDED Requirements

### Requirement: Native BFSDeepCrawlStrategy

The system SHALL use `crawl4ai.deep_crawling.BFSDeepCrawlStrategy` for recursive crawls instead of manual BFS. The strategy SHALL be configured with `max_depth`, `max_pages`, and `include_external=False`.

#### Scenario: Recursive crawl uses native strategy

- **WHEN** `WebCrawler.crawl_recursive()` is called with `max_depth=2, max_pages=10`
- **THEN** `CrawlerRunConfig` SHALL include `deep_crawl_strategy=BFSDeepCrawlStrategy(max_depth=2, max_pages=10, include_external=False)`

#### Scenario: Single-page crawl does not use deep crawl

- **WHEN** `WebCrawler.crawl()` is called (not `crawl_recursive`)
- **THEN** `CrawlerRunConfig` SHALL NOT include `deep_crawl_strategy`

### Requirement: Rate limiter on dispatcher

The system SHALL apply `RateLimiter` to `MemoryAdaptiveDispatcher` for all deep crawls. The default SHALL be `base_delay=(0.5, 1.0)` and `max_retries=2`.

#### Scenario: Rate limiter applied to recursive crawl

- **WHEN** `WebCrawler.crawl_recursive()` is called
- **THEN** the `MemoryAdaptiveDispatcher` SHALL include `rate_limiter=RateLimiter(base_delay=(0.5, 1.0), max_retries=2)`

#### Scenario: Custom rate limit via env vars

- **WHEN** `CRAWL4AI_RATE_LIMIT_DELAY_MIN=1.0` and `CRAWL4AI_RATE_LIMIT_DELAY_MAX=3.0` are set
- **THEN** `RateLimiter` SHALL use `base_delay=(1.0, 3.0)`

### Requirement: URL-specific config routing

The system SHALL support URL-specific `CrawlerRunConfig` objects via `CrawlerConfig.url_patterns`. When `url_patterns` is non-empty, `to_crawler_run_configs()` SHALL return one `CrawlerRunConfig` per pattern with `url_matcher=[pattern]` and `match_mode=MatchMode.OR`.

#### Scenario: No URL patterns (default)

- **WHEN** `CrawlerConfig.url_patterns` is empty
- **THEN** `to_crawler_run_configs()` SHALL return a single `CrawlerRunConfig` with no `url_matcher`

#### Scenario: URL patterns configured

- **WHEN** `CrawlerConfig.url_patterns=["*api/*", "*docs/*"]`
- **THEN** `to_crawler_run_configs()` SHALL return two `CrawlerRunConfig` objects, each with its own `url_matcher` pattern

### Requirement: Lifespan crawler config parity

The system SHALL use `CrawlerConfig.to_browser_config()` for the lifespan singleton crawler, not just `BrowserConfig(headless=True)`.

#### Scenario: Lifespan crawler uses full config

- **WHEN** `create_crawl4ai_crawler()` is called
- **THEN** `BrowserConfig` SHALL be created from `get_crawler_config().to_browser_config()`, including timeout, user-agent, sandbox flags, viewport, and stealth settings

#### Scenario: Lifespan crawler uses lazy imports

- **WHEN** `connections/crawl4ai.py` is imported
- **THEN** `get_crawler_config` and `WebCrawler` SHALL be imported lazily inside functions, not at module level
