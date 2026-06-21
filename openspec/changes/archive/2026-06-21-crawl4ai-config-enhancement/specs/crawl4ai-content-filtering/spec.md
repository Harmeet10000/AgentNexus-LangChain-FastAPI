## ADDED Requirements

### Requirement: Content noise filtering via excluded_tags

The system SHALL strip specified HTML tags from crawled content before markdown conversion. The default tags SHALL be `["nav", "footer", "header", "aside", "form"]`.

#### Scenario: Default excluded tags applied

- **WHEN** `CrawlerConfig` is created with default settings
- **THEN** `to_crawler_run_config()` SHALL include `excluded_tags=["nav", "footer", "header", "aside", "form"]`

#### Scenario: Custom excluded tags via env var

- **WHEN** `CRAWL4AI_EXCLUDED_TAGS` env var is set to `"nav,footer,aside"`
- **THEN** `CrawlerConfig.excluded_tags` SHALL be `["nav", "footer", "aside"]`

### Requirement: PruningContentFilter for noise reduction

The system SHALL apply `PruningContentFilter` to all crawls by default via `get_markdown_generator()`. The filter SHALL use configurable `pruning_threshold` (default 0.5) and `word_count_threshold` (default 30).

#### Scenario: Default pruning filter applied

- **WHEN** `WebCrawler.crawl()` is called
- **THEN** the `CrawlerRunConfig` SHALL include a `DefaultMarkdownGenerator` with `PruningContentFilter(threshold=0.5, min_word_threshold=30)`

#### Scenario: Custom pruning thresholds via env vars

- **WHEN** `CRAWL4AI_PRUNING_THRESHOLD=0.7` and `CRAWL4AI_WORD_COUNT_THRESHOLD=50` are set
- **THEN** the `PruningContentFilter` SHALL use `threshold=0.7` and `min_word_threshold=50`

### Requirement: MarkdownGenerator options

The system SHALL configure `DefaultMarkdownGenerator` with `ignore_links=True` and `ignore_images=True` by default to reduce markdown noise.

#### Scenario: Links and images excluded by default

- **WHEN** `get_markdown_generator()` is called without arguments
- **THEN** the returned `DefaultMarkdownGenerator` SHALL have `options={"ignore_links": True, "ignore_images": True}`

#### Scenario: Custom markdown options via env vars

- **WHEN** `CRAWL4AI_IGNORE_LINKS=false` is set
- **THEN** `get_markdown_generator()` SHALL use `ignore_links=False`
