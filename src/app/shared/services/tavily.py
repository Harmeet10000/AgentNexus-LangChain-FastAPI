"""Tavily search service integration."""

from __future__ import annotations

from collections.abc import Mapping

import httpx
from pydantic import BaseModel, ConfigDict

from app.config import get_settings
from app.connections.tavily import get_shared_tavily_http_client
from app.utils import ExternalServiceException, ValidationException, logger


class SearchResult(BaseModel):
    """Normalized Tavily search result."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    url: str
    title: str
    content: str
    score: float
    published_date: str | None = None
    raw_content: str | None = None


class SearchResponse(BaseModel):
    """Normalized Tavily search response."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    query: str
    results: list[SearchResult]
    answer: str | None = None
    total_results: int


def _validate_search_inputs(query: str, max_results: int, topic: str) -> None:
    """Validate search inputs.

    Args:
        query: The search query string
        max_results: Maximum number of results requested
        topic: Search topic category

    Raises:
        ValidationException: If inputs are invalid
    """
    settings = get_settings()
    if not settings.TAVILY_API_KEY:
        raise ValidationException(detail="Tavily API key not configured")
    if not query.strip():
        raise ValidationException(detail="Search query is required")
    if max_results < 1:
        raise ValidationException(detail="max_results must be greater than 0")
    if topic not in {"general", "news", "finance"}:
        raise ValidationException(detail="topic must be one of: general, news, finance")


def _read_string(data: Mapping[str, object], key: str) -> str:
    """Read a required string from a mapping."""
    value = data.get(key)
    return value if isinstance(value, str) else ""


def _read_optional_string(data: Mapping[str, object], key: str) -> str | None:
    """Read an optional string from a mapping."""
    value = data.get(key)
    return value if isinstance(value, str) else None


def _read_float(data: Mapping[str, object], key: str) -> float:
    """Read a float from a mapping."""
    value = data.get(key)
    if isinstance(value, int | float):
        return float(value)
    return 0.0


def _build_search_result(result: Mapping[str, object]) -> SearchResult:
    """Build a SearchResult from API response data."""
    return SearchResult(
        url=_read_string(result, "url"),
        title=_read_string(result, "title"),
        content=_read_string(result, "content"),
        score=_read_float(result, "score"),
        published_date=_read_optional_string(result, "published_date"),
        raw_content=_read_optional_string(result, "raw_content"),
    )


async def search(
    query: str,
    max_results: int = 10,
    topic: str = "general",
    include_answer: bool = True,
    include_raw_content: bool = False,
    include_images: bool = False,
    http_client: httpx.AsyncClient | None = None,
) -> SearchResponse:
    """Search the web using Tavily.

    Args:
        query: The search query
        max_results: Maximum number of results (capped by TAVILY_MAX_RESULTS_LIMIT)
        topic: Search topic category: "general", "news", or "finance"
        include_answer: Whether to include an answer in the response
        include_raw_content: Whether to include raw page content
        include_images: Whether to include images from results
        http_client: Optional HTTPX client (uses shared client if not provided)

    Returns:
        SearchResponse containing results and metadata

    Raises:
        ValidationException: If inputs are invalid or API key not configured
        ExternalServiceException: If Tavily API request fails
    """
    settings = get_settings()
    _validate_search_inputs(query=query, max_results=max_results, topic=topic)

    payload = {
        "api_key": settings.TAVILY_API_KEY,
        "query": query,
        "max_results": min(max_results, settings.TAVILY_MAX_RESULTS_LIMIT),
        "topic": topic,
        "include_answer": include_answer,
        "include_raw_content": include_raw_content,
        "include_images": include_images,
        "search_depth": "basic",
    }

    log = logger.bind(
        service="tavily",
        query=query,
        max_results=payload["max_results"],
        include_answer=include_answer,
        topic=topic,
    )
    log.info("Executing Tavily search")

    if http_client is None:
        http_client = get_shared_tavily_http_client()

    request_url = "/search"
    try:
        response = await http_client.post(request_url, json=payload)
        response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        log.bind(status_code=exc.response.status_code).warning("Tavily returned an error response")
        raise ExternalServiceException(
            service="Tavily",
            detail=f"HTTP {exc.response.status_code}",
            status_code=exc.response.status_code,
        ) from exc
    except httpx.TimeoutException as exc:
        log.warning("Tavily request timed out")
        raise ExternalServiceException(
            service="Tavily",
            detail="request timed out",
        ) from exc
    except httpx.HTTPError as exc:
        log.bind(error=str(exc)).warning("Tavily request failed")
        raise ExternalServiceException(
            service="Tavily",
            detail="network error",
        ) from exc

    data = response.json()
    if not isinstance(data, Mapping):
        log.warning("Tavily returned an invalid response payload")
        raise ExternalServiceException(
            service="Tavily",
            detail="invalid response payload",
        )

    results = [
        _build_search_result(result)
        for result in data.get("results", [])
        if isinstance(result, Mapping)
    ]

    log.bind(returned_results=len(results)).info("Tavily search completed")
    return SearchResponse(
        query=query,
        results=results,
        answer=_read_optional_string(data, "answer"),
        total_results=len(results),
    )


async def get_context(
    query: str,
    max_results: int = 5,
    http_client: httpx.AsyncClient | None = None,
) -> str:
    """Build a plain-text context block from Tavily results.

    Args:
        query: The search query
        max_results: Maximum number of results to include
        http_client: Optional HTTPX client (uses shared client if not provided)

    Returns:
        Plain-text context combining search answer and top results
    """
    response = await search(
        query=query,
        max_results=max_results,
        include_answer=True,
        http_client=http_client,
    )

    context_parts: list[str] = []
    if response.answer:
        context_parts.append(f"Answer: {response.answer}")

    context_parts.extend(
        (f"Source: {result.title}\nURL: {result.url}\nContent: {result.content}")
        for result in response.results[:max_results]
    )

    return "\n\n".join(context_parts)


async def get_tavily_client() -> None:
    """Deprecated: Use search() or get_context() functions directly with DI.

    This function is kept for backward compatibility but should not be used.
    """
    msg = "get_tavily_client() is deprecated. Use search() or get_context() functions directly."
    logger.warning(msg)
