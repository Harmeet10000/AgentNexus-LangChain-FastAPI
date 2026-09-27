"""Utility functions and tools for the Tavily-backed deep research graph."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from typing import (  # noqa: TC003 — Annotated, Any, Literal used at runtime by Pydantic/LangChain
    TYPE_CHECKING,
    Annotated,
    Any,
    Literal,
    cast,
)

import httpx
from langchain_core.exceptions import LangChainException
from langchain_core.messages import AIMessage, HumanMessage, filter_messages
from langchain_core.runnables import (
    RunnableConfig,  # noqa: TC002 — RunnableConfig used at runtime by LangChain tool
)
from langchain_core.tools import InjectedToolArg, tool
from pydantic import BaseModel, ConfigDict
from returns.result import Failure

from app.shared.langchain_layer import build_chat_model
from app.shared.services import search
from app.shared.services.tavily import (
    SearchResponse,  # noqa: TC001 — resolved at runtime by Pydantic
)
from app.utils import ExternalServiceException, logger

from .config import Configuration
from .execution import gather_limited, get_research_execution_gate
from .prompts import _SUMMARIZE_WEBPAGE_PROMPT
from .state import ResearchComplete, Summary

if TYPE_CHECKING:
    from langchain_core.messages import MessageLikeRepresentation
    from langchain_core.tools import BaseTool

TAVILY_SEARCH_DESCRIPTION = (
    "Search the web with Tavily for current, source-backed research. "
    "Use focused queries and prefer multiple narrow searches over one broad query."
)


class TavilySearchBatch(BaseModel):
    """Search responses plus the requests that were not successfully executed."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    responses: list[SearchResponse]
    failed_queries: list[str]
    omitted_queries: list[str]


class _SummaryOutcome(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    content: str | None
    label: str


@tool(description=TAVILY_SEARCH_DESCRIPTION)
async def tavily_search(
    queries: list[str],
    config: RunnableConfig,
    max_results: Annotated[int, InjectedToolArg] = 5,
    topic: Annotated[Literal["general", "news", "finance"], InjectedToolArg] = "general",
) -> str:
    """Fetch and summarize Tavily search results."""
    configurable: Configuration = Configuration.from_runnable_config(config)
    search_batch = await tavily_search_async(
        search_queries=queries,
        max_results=max_results,
        topic=topic,
        include_raw_content=True,
        config=config,
    )
    unique_results: dict[str, dict[str, str | None]] = {}
    for response in search_batch.responses:
        for result in response.results:
            if result.url not in unique_results:
                unique_results[result.url] = {
                    "title": result.title,
                    "content": result.content,
                    "raw_content": result.raw_content,
                    "query": response.query,
                }

    notices: list[str] = []
    if search_batch.omitted_queries:
        notices.append(
            "Warning: query limit reached; these queries were not searched: "
            + ", ".join(search_batch.omitted_queries)
        )
    if search_batch.failed_queries:
        notices.append(
            "Warning: these queries failed and have no results: "
            + ", ".join(search_batch.failed_queries)
        )
    if not unique_results:
        return "\n".join(
            [
                *notices,
                "No valid search results found. Try narrower or different search queries.",
            ]
        )

    summarization_model = (
        build_chat_model(
            model_name=configurable.summarization_model,
            max_tokens=configurable.summarization_model_max_tokens,
        )
        .with_structured_output(Summary)
        .with_retry(stop_after_attempt=configurable.max_structured_output_retries)
    )

    async def summarize_result(result: dict[str, str | None]) -> str | None:
        """Summarize one bounded search result, if raw content is available."""
        raw_content = result.get("raw_content")
        if not raw_content:
            return None
        return await summarize_webpage(
            summarization_model,
            raw_content[: configurable.max_content_length],
        )

    gate = get_research_execution_gate(config)

    async def run_summary(result: dict[str, str | None]) -> _SummaryOutcome:
        """Run one summary and label fallbacks so raw text cannot masquerade as synthesis."""
        try:
            return _SummaryOutcome(
                content=await gate.run_summary(lambda: summarize_result(result)),
                label="SUMMARY",
            )
        except (ExternalServiceException, LangChainException, TimeoutError) as exc:
            exc.add_note("operation=summarize_webpage")
            logger.bind(operation="summarize_webpage", error=str(exc)).warning(
                "summarization_failed"
            )
            raw_content = result.get("raw_content")
            return _SummaryOutcome(
                content=(raw_content[: configurable.max_content_length] if raw_content else None),
                label="RAW CONTENT (summarization failed)",
            )

    summaries = await gather_limited(
        (lambda result=result: run_summary(result) for result in unique_results.values()),
        limit=configurable.max_concurrent_summaries,
    )
    lines = ["Search results:", *notices]
    for index, ((url, result), summary_outcome) in enumerate(
        zip(unique_results.items(), summaries, strict=True),
        start=1,
    ):
        content = summary_outcome.content
        label = summary_outcome.label
        if content is None:
            content = result["content"] or "No summary or content excerpt available."
            label = "CONTENT EXCERPT"
        lines.extend(
            [
                "",
                f"--- SOURCE {index}: {result['title']} ---",
                f"URL: {url}",
                "",
                f"{label}:\n{content}",
            ]
        )
    return "\n".join(lines)


async def tavily_search_async(
    search_queries: list[str],
    max_results: int = 5,
    topic: Literal["general", "news", "finance"] = "general",
    include_raw_content: bool = True,
    config: RunnableConfig | None = None,
) -> TavilySearchBatch:
    """Execute bounded Tavily searches through the shared service client."""
    configurable = Configuration.from_runnable_config(config)
    http_client = _get_httpx_client_from_config(config)
    requested_queries = list(
        dict.fromkeys(query.strip() for query in search_queries if query.strip())
    )
    normalized_queries = requested_queries[: configurable.max_search_queries]
    omitted_queries = requested_queries[configurable.max_search_queries :]
    search_log = logger.bind(
        component="open_deep_search",
        search_api="tavily",
        queries=len(normalized_queries),
        omitted_queries=len(omitted_queries),
        max_results=max_results,
        topic=topic,
    )
    gate = get_research_execution_gate(config)

    async def run_query(query: str) -> Any:
        """Run one Tavily query and convert expected provider failures to skips."""
        try:
            return await gate.run_search(
                lambda: search(
                    query=query,
                    max_results=max_results,
                    topic=topic,
                    include_answer=False,
                    include_raw_content=include_raw_content,
                    http_client=http_client,
                )
            )
        except (ExternalServiceException, TimeoutError) as exc:
            search_log.bind(query_length=len(query), error=str(exc)).warning("tavily_query_failed")
            return None

    results = await gather_limited(
        (lambda query=query: run_query(query) for query in normalized_queries),
        limit=configurable.max_concurrent_search_requests,
    )
    responses: list[SearchResponse] = []
    failed_queries: list[str] = []
    for query, result in zip(normalized_queries, results, strict=True):
        if result is None:
            failed_queries.append(query)
            continue
        if isinstance(result, Failure):
            error = result.failure()
            search_log.bind(query=query, error_code=error.code.value).error(
                "tavily_search_async_failed", error=error.message
            )
            failed_queries.append(query)
            continue
        responses.append(result.unwrap())
    if normalized_queries and not responses:
        raise ExternalServiceException(
            service="Tavily",
            detail="All Tavily queries failed",
        )
    search_log.info("tavily_search_async_complete")
    return TavilySearchBatch(
        responses=responses,
        failed_queries=failed_queries,
        omitted_queries=omitted_queries,
    )


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
        logger.bind(operation="summarize_webpage").warning("summarization_timeout")
        return webpage_content
    except (RuntimeError, ValueError, AttributeError) as exc:
        logger.bind(operation="summarize_webpage", error=str(exc)).warning("summarization_failed")
        return webpage_content


@tool(description="Strategic reflection tool for research planning")
def think_tool(reflection: str) -> str:
    """Record a short reflection before deciding whether to search again."""
    return f"Reflection recorded: {reflection}"


async def get_all_tools(config: RunnableConfig | None = None) -> list[BaseTool]:
    """Assemble immediate research tools; durable Crawl4AI is API-only."""
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
