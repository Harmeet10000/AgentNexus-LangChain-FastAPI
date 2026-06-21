## Why

Production crawler has 3 independent instantiation paths with inconsistent config. `CrawlerConfig` covers ~18% of crawl4ai's options — missing content filtering, viewport/stealth, wait strategies, rate limiting, and recursive crawl optimization. Raw markdown includes nav/footers/ads, JS-heavy pages return incomplete content, and recursive crawls hammer servers with no delay.

## What Changes

- Add content noise filtering (excluded_tags, PruningContentFilter, word_count_threshold) to reduce markdown 30-60%
- Add viewport, stealth mode, and user-agent randomization for anti-bot bypass
- Add page timeout, wait_until, and wait_for for JS-heavy page rendering
- Add MarkdownGenerator helper with PruningContentFilter defaults
- Replace 90-line manual BFS with native `BFSDeepCrawlStrategy`
- Add RateLimiter to MemoryAdaptiveDispatcher for polite crawling
- Add URL-specific config routing via url_matcher patterns
- Fix lifespan crawler to use full CrawlerConfig (not just headless=True)
- Add 11 new env vars for all new config fields

## Capabilities

### New Capabilities

- `crawl4ai-content-filtering`: Content noise filtering via excluded_tags, PruningContentFilter, and MarkdownGenerator options
- `crawl4ai-browser-config`: Viewport, stealth mode, user-agent randomization, page timeout, and wait strategies
- `crawl4ai-deep-crawl`: Native BFSDeepCrawlStrategy replacing manual BFS, RateLimiter on dispatcher, URL-specific config routing

### Modified Capabilities

(none — all new capabilities)

## Impact

- **Code**: `src/app/shared/crawler/config.py`, `src/app/shared/crawler/crawler.py`, `src/app/connections/crawl4ai.py`, `src/app/config/settings.py`
- **Config**: 11 new `CRAWL4AI_*` env vars with sensible defaults
- **Behavior**: All crawls now get content filtering; recursive crawls use native strategy with rate limiting; lifespan crawler gets full browser config
- **Dependencies**: crawl4ai deep_crawling, content_filter_strategy, async_dispatcher modules (already installed)
