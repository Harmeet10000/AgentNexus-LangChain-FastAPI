## ADDED Requirements

### Requirement: Browser viewport configuration

The system SHALL configure browser viewport dimensions via `CrawlerConfig.viewport_width` (default 1280) and `CrawlerConfig.viewport_height` (default 720). These SHALL be passed to `BrowserConfig` via `to_browser_config()`.

#### Scenario: Default viewport applied

- **WHEN** `CrawlerConfig` is created with default settings
- **THEN** `to_browser_config()` SHALL include `viewport={"width": 1280, "height": 720}`

#### Scenario: Custom viewport via env vars

- **WHEN** `CRAWL4AI_VIEWPORT_WIDTH=1920` and `CRAWL4AI_VIEWPORT_HEIGHT=1080` are set
- **THEN** `to_browser_config()` SHALL include `viewport={"width": 1920, "height": 1080}`

### Requirement: Stealth mode

The system SHALL support stealth mode via `CrawlerConfig.enable_stealth` (default False). When enabled, `BrowserConfig` SHALL be created with `stealth=True`.

#### Scenario: Stealth mode disabled by default

- **WHEN** `CrawlerConfig` is created with default settings
- **THEN** `to_browser_config()` SHALL NOT include `stealth=True`

#### Scenario: Stealth mode enabled via env var

- **WHEN** `CRAWL4AI_STEALTH=true` is set
- **THEN** `to_browser_config()` SHALL include `stealth=True`

### Requirement: User-agent randomization

The system SHALL support user-agent randomization via `CrawlerConfig.user_agent_mode` (default "fixed"). When set to "random", crawl4ai SHALL rotate user-agent per crawl.

#### Scenario: Fixed user-agent by default

- **WHEN** `CrawlerConfig` is created with default settings
- **THEN** `user_agent_mode` SHALL be `"fixed"`

#### Scenario: Random user-agent via env var

- **WHEN** `CRAWL4AI_USER_AGENT_MODE=random` is set
- **THEN** `user_agent_mode` SHALL be `"random"`

### Requirement: Page timeout

The system SHALL configure page load timeout via `CrawlerConfig.page_timeout` (default 30000ms). This SHALL be passed to `CrawlerRunConfig` via `to_crawler_run_config()`.

#### Scenario: Default page timeout applied

- **WHEN** `CrawlerConfig` is created with default settings
- **THEN** `to_crawler_run_config()` SHALL include `page_timeout=30000`

#### Scenario: Custom page timeout via env var

- **WHEN** `CRAWL4AI_PAGE_TIMEOUT=60000` is set
- **THEN** `to_crawler_run_config()` SHALL include `page_timeout=60000`

### Requirement: Wait strategy

The system SHALL support `wait_until` (default None) and `wait_for` (default None) on `CrawlerConfig`. These SHALL be passed to `CrawlerRunConfig` when set.

#### Scenario: No wait strategy by default

- **WHEN** `CrawlerConfig` is created with default settings
- **THEN** `to_crawler_run_config()` SHALL NOT include `wait_until` or `wait_for`

#### Scenario: Networkidle wait via env var

- **WHEN** `CRAWL4AI_WAIT_UNTIL=networkidle` is set
- **THEN** `to_crawler_run_config()` SHALL include `wait_until="networkidle"`

#### Scenario: CSS selector wait via env var

- **WHEN** `CRAWL4AI_WAIT_FOR=css:main` is set
- **THEN** `to_crawler_run_config()` SHALL include `wait_for="css:main"`
