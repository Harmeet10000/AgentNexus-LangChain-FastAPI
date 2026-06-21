## 1. CrawlerConfig Fields

- [ ] 1.1 Add `excluded_tags`, `word_count_threshold`, `pruning_threshold`, `pruning_min_words`, `ignore_links`, `ignore_images` fields to `CrawlerConfig`
- [ ] 1.2 Add `viewport_width`, `viewport_height`, `enable_stealth`, `user_agent_mode` fields to `CrawlerConfig`
- [ ] 1.3 Add `page_timeout`, `wait_until`, `wait_for` fields to `CrawlerConfig`
- [ ] 1.4 Add `rate_limit_delay`, `rate_limit_retries` fields to `CrawlerConfig`
- [ ] 1.5 Add `url_patterns` field to `CrawlerConfig`

## 2. Config Methods

- [ ] 2.1 Update `to_browser_config()` to pass viewport, stealth, user_agent_mode
- [ ] 2.2 Update `to_crawler_run_config()` to pass excluded_tags, word_count_threshold, page_timeout, wait_until, wait_for
- [ ] 2.3 Add `get_markdown_generator()` method with PruningContentFilter defaults
- [ ] 2.4 Add `to_crawler_run_configs()` method for URL-specific routing

## 3. Settings

- [ ] 3.1 Add 11 new `CRAWL4AI_*` env vars to `settings.py`

## 4. Crawler Integration

- [ ] 4.1 Update `WebCrawler.crawl()` to use `get_markdown_generator()` with PruningContentFilter
- [ ] 4.2 Replace manual BFS in `crawl_recursive()` with native `BFSDeepCrawlStrategy`
- [ ] 4.3 Add `RateLimiter` to `MemoryAdaptiveDispatcher` in `crawl_recursive()`
- [ ] 4.4 Extract `_to_crawl_result()` static method for result conversion

## 5. Lifespan Parity

- [ ] 5.1 Update `create_crawl4ai_crawler()` to use full `CrawlerConfig.to_browser_config()`
- [ ] 5.2 Move `get_crawler_config` and `WebCrawler` to lazy imports in `connections/crawl4ai.py`

## 6. Verification

- [ ] 6.1 Run `ruff check` and `ruff format` on all changed files
- [ ] 6.2 Run `ty check` on all changed files
- [ ] 6.3 Verify `CrawlerConfig()` builds without errors
- [ ] 6.4 Verify `crawl_recursive()` works with BFSDeepCrawlStrategy
