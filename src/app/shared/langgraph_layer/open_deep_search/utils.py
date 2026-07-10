"""Utility functions and tools for the Tavily-backed deep research graph."""

from __future__ import annotations

import asyncio
import time
from datetime import UTC, datetime
from typing import (  # noqa: TC003 — Annotated, Any, Literal used at runtime by Pydantic/LangChain
    TYPE_CHECKING,
    Annotated,
    Any,
    Literal,
    cast,
)

import httpx
from crawl4ai import CacheMode, CrawlerRunConfig
from crawl4ai.markdown_generation_strategy import DefaultMarkdownGenerator
from langchain_core.messages import AIMessage, HumanMessage, filter_messages
from langchain_core.runnables import (
    RunnableConfig,  # noqa: TC002 — RunnableConfig used at runtime by LangChain tool
)
from langchain_core.tools import InjectedToolArg, tool
from playwright.async_api import Error as PlaywrightError

from app.shared.crawler import sanitize_url, validate_url
from app.shared.langchain_layer import _build_chat_model
from app.shared.services import search
from app.utils import ExternalServiceException, logger

from .config import Configuration
from .prompts import _SUMMARIZE_WEBPAGE_PROMPT
from .state import ResearchComplete, Summary

if TYPE_CHECKING:
    from langchain_core.messages import MessageLikeRepresentation
    from langchain_core.tools import BaseTool

    from app.shared.services.tavily import SearchResponse

TAVILY_SEARCH_DESCRIPTION = (
    "Search the web with Tavily for current, source-backed research. "
    "Use focused queries and prefer multiple narrow searches over one broad query."
)


@tool(description=TAVILY_SEARCH_DESCRIPTION)
async def tavily_search(
    queries: list[str],
    max_results: Annotated[int, InjectedToolArg] = 5,
    topic: Annotated[Literal["general", "news", "finance"], InjectedToolArg] = "general",
    config: RunnableConfig | None = None,
) -> str:
    """Fetch and summarize Tavily search results."""
    search_results = await tavily_search_async(
        search_queries=queries,
        max_results=max_results,
        topic=topic,
        include_raw_content=True,
        config=config,
    )
    unique_results: dict[str, dict[str, str | None]] = {}
    for response in search_results:
        for result in response.results:
            if result.url not in unique_results:
                unique_results[result.url] = {
                    "title": result.title,
                    "content": result.content,
                    "raw_content": result.raw_content,
                    "query": response.query,
                }

    configurable: Configuration = Configuration.from_runnable_config(config)
    summarization_model = (
        _build_chat_model(
            model_name=configurable.summarization_model,
            max_tokens=configurable.summarization_model_max_tokens,
        )
        .with_structured_output(Summary)
        .with_retry(stop_after_attempt=configurable.max_structured_output_retries)
    )

    async def summarize_result(result: dict[str, str | None]) -> str | None:
        raw_content = result.get("raw_content")
        if not raw_content:
            return None
        return await summarize_webpage(
            summarization_model,
            raw_content[: configurable.max_content_length],
        )

    summaries = await asyncio.gather(
        *(summarize_result(result) for result in unique_results.values())
    )
    if not unique_results:
        return "No valid search results found. Try narrower or different search queries."

    lines = ["Search results:"]
    for index, ((url, result), summary) in enumerate(
        zip(unique_results.items(), summaries, strict=True),
        start=1,
    ):
        content = result["content"] if summary is None else summary
        lines.extend(
            [
                "",
                f"--- SOURCE {index}: {result['title']} ---",
                f"URL: {url}",
                "",
                f"SUMMARY:\n{content}",
            ]
        )
    return "\n".join(lines)


async def tavily_search_async(
    search_queries: list[str],
    max_results: int = 5,
    topic: Literal["general", "news", "finance"] = "general",
    include_raw_content: bool = True,
    config: RunnableConfig | None = None,
) -> list[SearchResponse]:
    """Execute bounded Tavily searches through the shared service client."""
    http_client = _get_httpx_client_from_config(config)
    search_log = logger.bind(
        component="open_deep_search",
        search_api="tavily",
        queries=len(search_queries),
        max_results=max_results,
        topic=topic,
    )
    try:
        responses: list[SearchResponse] = await asyncio.gather(
            *(
                search(
                    query=query,
                    max_results=max_results,
                    topic=topic,
                    include_answer=False,
                    include_raw_content=include_raw_content,
                    http_client=http_client,
                )
                for query in search_queries
            )
        )
    except ExternalServiceException:
        search_log.exception("tavily_search_async_failed")
        raise
    search_log.info("tavily_search_async_complete")
    return list(responses)


def _get_httpx_client_from_config(config: RunnableConfig | None) -> httpx.AsyncClient | None:
    """Read the lifespan-owned HTTPX client from RunnableConfig when present."""
    if not config:
        return None
    configurable = config.get("configurable", {})
    http_client = configurable.get("httpx_client") or configurable.get("tavily_http_client")
    if isinstance(http_client, httpx.AsyncClient):
        return http_client
    return None


def _get_crawl4ai_crawler_from_config(config: RunnableConfig | None) -> Any:
    """Read the lifespan-owned Crawl4AI AsyncWebCrawler from RunnableConfig when present."""
    if not config:
        return None
    configurable = config.get("configurable", {})
    return configurable.get("crawl4ai_crawler")


async def summarize_webpage(model: Any, webpage_content: str) -> str:
    """Summarize webpage content with timeout protection."""
    try:
        prompt_content = _SUMMARIZE_WEBPAGE_PROMPT.format(
            webpage_content=webpage_content,
            date=get_today_str(),
        )
        summary = cast(
            "Summary",
            await asyncio.wait_for(
                model.ainvoke([HumanMessage(content=prompt_content)]),
                timeout=60.0,
            ),
        )
        return (  # noqa: TRY300 — return must be inside try for timeout handling
            f"<summary>\n{summary.summary}\n</summary>\n\n"
            f"<key_excerpts>\n{summary.key_excerpts}\n</key_excerpts>"
        )
    except TimeoutError:
        logger.warning("summarization_timeout")
        return webpage_content
    except (RuntimeError, ValueError, AttributeError) as exc:
        logger.bind(error=str(exc)).warning("summarization_failed")
        return webpage_content


CRAWL_WEBPAGE_DESCRIPTION = (
    "Crawl a single URL with a headless browser to extract its full rendered markdown content. "
    "Use this when you need deeper content from a specific URL than Tavily's summary provides, "
    "when the target page requires JavaScript rendering, or when a user explicitly provides a URL."
)


@tool(description=CRAWL_WEBPAGE_DESCRIPTION)
async def crawl_webpage(
    url: str,
    config: RunnableConfig | None = None,
) -> str:
    """Fetch and return rendered markdown content from a URL using Crawl4AI."""
    url = sanitize_url(url)
    is_valid, error_msg = validate_url(url)
    if not is_valid:
        return f"Error: cannot crawl URL — {error_msg}"

    crawler = _get_crawl4ai_crawler_from_config(config)
    if crawler is None:
        return "Error: Crawl4AI browser is not available (not configured or failed to start)"

    start_time = time.time()
    try:
        run_config = CrawlerRunConfig(
            markdown_generator=DefaultMarkdownGenerator(),
            cache_mode=CacheMode.BYPASS,
        )
        result = await crawler.arun(url=url, config=run_config)
    except TimeoutError:
        return f"Error crawling {url}: Crawl timeout"
    except (httpx.HTTPError, PlaywrightError) as exc:
        exc.add_note(f"url={url}")
        return f"Error crawling {url}: {exc!s}"

    elapsed = int((time.time() - start_time) * 1000)
    if result.success:
        markdown = result.markdown.raw_markdown if result.markdown else None
        word_count = len(markdown.split()) if markdown else 0
        title = result.metadata.get("title") if result.metadata else None

        parts = [f"Crawl result for: {result.url}"]
        if title:
            parts.append(f"Title: {title}")
        parts.append(f"Words: {word_count} | Time: {elapsed}ms")
        parts.append("")
        parts.append(markdown or "(no content extracted)")
        return "\n".join(parts)

    return f"Error crawling {url}: {result.error_message or 'Unknown error'} ({elapsed}ms)"


@tool(description="Strategic reflection tool for research planning")
def think_tool(reflection: str) -> str:
    """Record a short reflection before deciding whether to search again."""
    return f"Reflection recorded: {reflection}"


async def get_all_tools(config: RunnableConfig | None = None) -> list[BaseTool]:
    """Assemble research tools (Tavily search + Crawl4AI + reflection)."""
    _ = config
    search_tool = tavily_search
    search_tool.metadata = {
        **(search_tool.metadata or {}),
        "type": "search",
        "name": "web_search",
    }
    crawl_tool = crawl_webpage
    crawl_tool.metadata = {
        **(crawl_tool.metadata or {}),
        "type": "crawl",
        "name": "crawl_webpage",
    }
    return [tool(ResearchComplete), think_tool, search_tool, crawl_tool]


def get_notes_from_tool_calls(messages: list[MessageLikeRepresentation]) -> list[str]:
    """Extract notes from tool call messages."""
    return [str(tool_msg.content) for tool_msg in filter_messages(messages, include_types="tool")]


def is_token_limit_exceeded(exception: Exception, model_name: str | None = None) -> bool:
    """Detect common Gemini context limit errors."""
    _ = model_name
    error_text = str(exception).lower()
    exception_type = str(type(exception)).lower()
    return any(
        marker in error_text or marker in exception_type
        for marker in (
            "context length",
            "context window",
            "maximum context",
            "prompt is too long",
            "resourceexhausted",
            "token limit",
        )
    )


def get_model_token_limit(model_string: str) -> int | None:
    """Look up token limits for configured Gemini models."""
    model_name = model_string.lower()
    if "gemini-1.5-pro" in model_name:
        return 2_097_152
    if "gemini-1.5-flash" in model_name:
        return 1_048_576
    if "gemini" in model_name:
        return 1_000_000
    return None


def remove_up_to_last_ai_message(
    messages: list[MessageLikeRepresentation],
) -> list[MessageLikeRepresentation]:
    """Truncate message history up to the last AI message."""
    for index in range(len(messages) - 1, -1, -1):
        if isinstance(messages[index], AIMessage):
            return messages[:index]
    return messages


def get_today_str() -> str:
    """Get current UTC date formatted for prompts."""
    now = datetime.now(tz=UTC)
    return f"{now:%a} {now:%b} {now.day}, {now:%Y}"
