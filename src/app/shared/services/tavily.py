"""Tavily search service integration."""

from __future__ import annotations

from collections.abc import Mapping

import httpx
from fastapi import Request
from pydantic import BaseModel, ConfigDict

from app.config import get_settings
from app.connections import get_shared_httpx_client
from app.utils import ExternalServiceException, ValidationException, logger

TAVILY_BASE_URL = "https://api.tavily.com"
TAVILY_MAX_RESULTS_LIMIT = 20
TAVILY_TIMEOUT_SECONDS = 30.0


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


class TavilyClient:
    """Async client for the Tavily search API."""

    def __init__(
        self,
        api_key: str | None = None,
        http_client: httpx.AsyncClient | None = None,
        base_url: str = TAVILY_BASE_URL,
    ) -> None:
        self.api_key = api_key or get_settings().TAVILY_API_KEY
        self.base_url = base_url.rstrip("/")
        self._http_client = http_client
        self._owns_http_client = http_client is None

    async def search(
        self,
        query: str,
        max_results: int = 10,
        topic: str = "general",
        include_answer: bool = True,
        include_raw_content: bool = False,
        include_images: bool = False,
    ) -> SearchResponse:
        """Search the web using Tavily."""
        self._validate_search_inputs(query=query, max_results=max_results, topic=topic)

        payload = {
            "api_key": self.api_key,
            "query": query,
            "max_results": min(max_results, TAVILY_MAX_RESULTS_LIMIT),
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

        client = self._get_http_client()
        request_url = self._get_search_url()
        try:
            response = await client.post(request_url, json=payload)
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
            self._build_search_result(result)
            for result in data.get("results", [])
            if isinstance(result, Mapping)
        ]

        log.bind(returned_results=len(results)).info("Tavily search completed")
        return SearchResponse(
            query=query,
            results=results,
            answer=self._read_optional_string(data, "answer"),
            total_results=len(results),
        )

    async def get_context(
        self,
        query: str,
        max_results: int = 5,
    ) -> str:
        """Build a plain-text context block from Tavily results."""
        response = await self.search(
            query=query,
            max_results=max_results,
            include_answer=True,
        )

        context_parts: list[str] = []
        if response.answer:
            context_parts.append(f"Answer: {response.answer}")

        context_parts.extend(
            (
                f"Source: {result.title}\nURL: {result.url}\nContent: {result.content}"
            )
            for result in response.results[:max_results]
        )

        return "\n\n".join(context_parts)

    async def close(self) -> None:
        """Close the owned HTTP client."""
        if self._owns_http_client and self._http_client is not None:
            await self._http_client.aclose()
            self._http_client = None

    def _get_http_client(self) -> httpx.AsyncClient:
        if self._http_client is None:
            self._http_client = httpx.AsyncClient(
                base_url=self.base_url,
                timeout=TAVILY_TIMEOUT_SECONDS,
            )
        return self._http_client

    def _get_search_url(self) -> str:
        if self._owns_http_client:
            return "/search"
        return f"{self.base_url}/search"

    def _validate_search_inputs(self, query: str, max_results: int, topic: str) -> None:
        if not self.api_key:
            raise ValidationException(detail="Tavily API key not configured")
        if not query.strip():
            raise ValidationException(detail="Search query is required")
        if max_results < 1:
            raise ValidationException(detail="max_results must be greater than 0")
        if topic not in {"general", "news", "finance"}:
            raise ValidationException(detail="topic must be one of: general, news, finance")

    @classmethod
    def from_request(cls, request: Request) -> TavilyClient:
        """Build a Tavily client using the lifespan-owned HTTPX client."""
        return cls(http_client=request.app.state.httpx_client)

    @staticmethod
    def _build_search_result(result: Mapping[str, object]) -> SearchResult:
        return SearchResult(
            url=TavilyClient._read_string(result, "url"),
            title=TavilyClient._read_string(result, "title"),
            content=TavilyClient._read_string(result, "content"),
            score=TavilyClient._read_float(result, "score"),
            published_date=TavilyClient._read_optional_string(result, "published_date"),
            raw_content=TavilyClient._read_optional_string(result, "raw_content"),
        )

    @staticmethod
    def _read_string(data: Mapping[str, object], key: str) -> str:
        value = data.get(key)
        return value if isinstance(value, str) else ""

    @staticmethod
    def _read_optional_string(data: Mapping[str, object], key: str) -> str | None:
        value = data.get(key)
        return value if isinstance(value, str) else None

    @staticmethod
    def _read_float(data: Mapping[str, object], key: str) -> float:
        value = data.get(key)
        if isinstance(value, int | float):
            return float(value)
        return 0.0


async def get_tavily_client() -> TavilyClient:
    """Create a Tavily client backed by the shared lifespan HTTPX client."""
    return TavilyClient(http_client=get_shared_httpx_client())
