## Context

Production crawler (`src/app/shared/crawler/`) has 3 independent instantiation paths:
1. **Lifespan singleton** (`connections/crawl4ai.py`): `BrowserConfig(headless=True)` — missing timeout, user-agent, sandbox flags
2. **Per-call** (`crawler.py` → `WebCrawler.crawl()`): Full `CrawlerConfig.to_browser_config()` but no content filters, no wait strategy
3. **LangGraph tool** (`open_deep_search/utils.py`): Bare `CrawlerRunConfig` — no browser config at all

`CrawlerConfig` has 11 fields covering ~18% of crawl4ai's options. Crawl4AI exposes ~60+ `BrowserConfig` + `CrawlerRunConfig` options. Key gaps: content filtering, viewport/stealth, wait strategies, rate limiting, recursive crawl optimization.

## Goals / Non-Goals

**Goals:**
- Close the gap between crawl4ai best practices (44 example files) and production config
- Make all 3 instantiation paths use the same `CrawlerConfig`
- Add content noise filtering (30-60% markdown reduction)
- Replace 90-line manual BFS with native `BFSDeepCrawlStrategy`
- Add rate limiting for polite crawling
- Add stealth/viewport for anti-bot bypass when needed

**Non-Goals:**
- Cover all 60+ crawl4ai options (cover ~60% — the 80/20)
- Migrate `open_deep_search/utils.py` to use `CrawlerConfig` (separate change)
- Add proxy rotation (use env var `CRAWL4AI_PROXY` instead)
- Add JavaScript code injection (wait_for + wait_until covers most cases)

## Decisions

### D1: Lazy imports for crawl4ai modules in `connections/crawl4ai.py`

**Choice**: Move `get_crawler_config` and `WebCrawler` to local imports inside functions.

**Rationale**: Avoids circular imports between `app.shared.crawler` and `app.connections.crawl4ai`. The `TYPE_CHECKING` block handles type annotations.

**Alternative**: Top-level imports with `# noqa: PLC0415` — rejected because it's cleaner to use `TYPE_CHECKING` for type hints.

### D2: PruningContentFilter as default markdown generator

**Choice**: `get_markdown_generator()` defaults to `PruningContentFilter(threshold=0.5, min_word_threshold=30)`.

**Rationale**: 30-60% markdown reduction with minimal configuration. The threshold is conservative — users can tune via env vars.

**Alternative**: No default filter — rejected because raw markdown is too noisy for LLM consumption.

### D3: Native BFSDeepCrawlStrategy replaces manual BFS

**Choice**: Use crawl4ai's `BFSDeepCrawlStrategy` with `arun_many()` instead of manual BFS loop.

**Rationale**: Eliminates 90 lines of code. Gets streaming, rate limiting, and URL pattern matching for free.

**Alternative**: Keep manual BFS with added rate limiting — rejected because it's redundant when crawl4ai provides this natively.

### D4: RateLimiter with conservative defaults

**Choice**: `RateLimiter(base_delay=(0.5, 1.0), max_retries=2)` on all deep crawls.

**Rationale**: Polite crawling. 0.5-1.0s delay between requests is reasonable for most sites.

**Alternative**: No rate limiter — rejected because recursive crawls without delay hammer servers.

### D5: URL-specific configs via url_matcher patterns

**Choice**: `url_patterns: list[str]` field on `CrawlerConfig` with `MatchMode.OR` routing.

**Rationale**: API docs, blog posts, and landing pages have different content structures. URL patterns let you apply different extraction strategies.

**Alternative**: Single config for all URLs — rejected because it's too coarse for production use.

## Risks / Trade-offs

- **Risk**: `BFSDeepCrawlStrategy` may behave differently than manual BFS for edge cases (redirects, robots.txt). → **Mitigation**: Test with production URLs before deploying.
- **Risk**: `PruningContentFilter` may over-prune on short pages. → **Mitigation**: `min_word_threshold=30` is conservative; configurable via env var.
- **Risk**: `user_agent_mode="random"` may trigger anti-bot detection on some sites. → **Mitigation**: Default `"fixed"` preserves current behavior; opt-in via env var.
- **Trade-off**: Lazy imports add minor runtime overhead (one-time import per function call). → Acceptable because crawl4ai modules are already imported elsewhere in the process.
