"""Utility functions and tools for the Tavily-backed deep research graph."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Annotated, Any, Literal, cast  # noqa: TC003

import httpx
from langchain_core.messages import AIMessage, HumanMessage, filter_messages
from langchain_core.runnables import RunnableConfig  # noqa: TC002
from langchain_core.tools import InjectedToolArg, tool

from app.shared.langchain_layer.models import _build_chat_model
from app.shared.services import search
from app.utils import ExternalServiceException, logger

from .configuration import Configuration
from .prompts import summarize_webpage_prompt
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
    search_log  = logger.bind(
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


async def summarize_webpage(model: Any, webpage_content: str) -> str:
    """Summarize webpage content with timeout protection."""
    try:
        prompt_content = summarize_webpage_prompt.format(
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
        return (  # noqa: TRY300
            f"<summary>\n{summary.summary}\n</summary>\n\n"
            f"<key_excerpts>\n{summary.key_excerpts}\n</key_excerpts>"
        )
    except TimeoutError:
        logger.warning("summarization_timeout")
        return webpage_content
    except (RuntimeError, ValueError, AttributeError) as exc:
        logger.bind(error=str(exc)).warning("summarization_failed")
        return webpage_content


@tool(description="Strategic reflection tool for research planning")
def think_tool(reflection: str) -> str:
    """Record a short reflection before deciding whether to search again."""
    return f"Reflection recorded: {reflection}"


async def get_all_tools(config: RunnableConfig | None = None) -> list[BaseTool]:
    """Assemble Tavily-only research tools."""
    _ = config
    search_tool = tavily_search
    search_tool.metadata = {
        **(search_tool.metadata or {}),
        "type": "search",
        "name": "web_search",
    }
    return [tool(ResearchComplete), think_tool, search_tool]


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
