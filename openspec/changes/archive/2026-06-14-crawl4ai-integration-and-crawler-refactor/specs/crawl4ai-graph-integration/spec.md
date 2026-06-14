## ADDED Requirements

### Requirement: Lifespan-managed AsyncWebCrawler
The system SHALL initialize a Crawl4AI `AsyncWebCrawler` instance during FastAPI application startup and shut it down during application shutdown. The crawler SHALL be stored in `app.state.crawl4ai_crawler`.

#### Scenario: AsyncWebCrawler starts on application startup
- **WHEN** the FastAPI application starts
- **THEN** an `AsyncWebCrawler` is created with `BrowserConfig(headless=True)` and stored in `app.state.crawl4ai_crawler`

#### Scenario: AsyncWebCrawler shuts down on application shutdown
- **WHEN** the FastAPI application shuts down
- **THEN** `app.state.crawl4ai_crawler.close()` is awaited

#### Scenario: Startup failure does not block application
- **WHEN** Crawl4AI browser fails to start during lifespan
- **THEN** `app.state.crawl4ai_crawler` is set to `None` and application continues without crawl capability

### Requirement: Crawler client injected through RunnableConfig
The `build_open_deep_search_config` function SHALL accept and forward a crawl4ai client reference through `RunnableConfig.configurable`. The graph node SHALL read the client from `RunnableConfig`.

#### Scenario: Crawler client available in researcher node
- **WHEN** a researcher subgraph node runs with `RunnableConfig` containing `crawl4ai_crawler`
- **THEN** the node SHALL read the client from `config["configurable"]["crawl4ai_crawler"]`

#### Scenario: Crawler client unavailable
- **WHEN** the crawl tool is invoked but `config["configurable"]["crawl4ai_crawler"]` is `None`
- **THEN** the tool SHALL return an error message indicating the crawler is unavailable

### Requirement: crawl_webpage tool
The researcher LLM SHALL have a `crawl_webpage` tool that accepts a single URL string and returns the rendered markdown content of the page using Crawl4AI.

#### Scenario: Successful crawl
- **WHEN** the researcher LLM calls `crawl_webpage(url="https://example.com")`
- **THEN** the tool returns the page markdown content with word count, title, and crawl time

#### Scenario: Invalid URL
- **WHEN** the researcher LLM calls `crawl_webpage` with an invalid URL
- **THEN** the tool returns an error message describing the validation failure

#### Scenario: Crawl timeout
- **WHEN** the target page does not load within the configured timeout
- **THEN** the tool returns an error message with "Crawl timeout" and the URL

#### Scenario: JavaScript-heavy page
- **WHEN** the target page requires JavaScript rendering
- **THEN** Crawl4AI SHALL render the page before extracting markdown content

### Requirement: Dedicated crawl_executor graph node
The researcher subgraph SHALL have a dedicated `crawl_executor` node that executes `crawl_webpage` tool calls. This node SHALL be separate from the existing `researcher_tools` node.

#### Scenario: crawl_executor routes crawl tool calls
- **WHEN** the researcher LLM returns tool calls with `name == "crawl_webpage"`
- **THEN** `crawl_executor` SHALL execute the crawl for each call and return `ToolMessage` results

#### Scenario: crawl_executor does not process non-crawl tools
- **WHEN** the researcher LLM returns tool calls that do not include `crawl_webpage`
- **THEN** `crawl_executor` SHALL short-circuit and return no updates

#### Scenario: crawl_executor handles partial failures
- **WHEN** multiple `crawl_webpage` calls are made and some fail
- **THEN** successful results SHALL be returned alongside error messages for failed URLs

### Requirement: Updated tool assembly in get_all_tools
The `get_all_tools` function in `open_deep_search/utils.py` SHALL include `crawl_webpage` alongside `tavily_search`, `think_tool`, and `ResearchComplete`.

#### Scenario: crawl_webpage appears in tool list
- **WHEN** `get_all_tools()` is called
- **THEN** the returned list SHALL include a `crawl_webpage` tool with `metadata.type == "crawl"` and `metadata.name == "crawl_webpage"`

### Requirement: Updated researcher system prompt
The `_RESEARCH_SYSTEM_PROMPT` SHALL be updated to teach the researcher LLM when to use `crawl_webpage` vs `web_search`.

#### Scenario: Prompt includes crawl guidance
- **WHEN** the researcher LLM receives its system prompt
- **THEN** the prompt SHALL instruct the LLM to use `crawl_webpage` when deeper content is needed from a specific URL found in search results or provided by the user
