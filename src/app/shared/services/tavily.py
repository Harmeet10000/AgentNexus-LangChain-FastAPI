"""Tavily search service integration."""

from dataclasses import dataclass
from typing import Any

from app.config.settings import get_settings


@dataclass
class SearchResult:
    """Result from Tavily search."""

    url: str
    title: str
    content: str
    score: float
    published_date: str | None = None


@dataclass
class SearchResponse:
    """Response from search."""

    query: str
    results: list[SearchResult]
    answer: str | None = None
    total_results: int


class TavilyClient:
    """Client for Tavily search API."""

    def __init__(self):
        self.api_key = get_settings().TAVILY_API_KEY
        self.base_url = "https://api.tavily.com"

    async def search(
        self,
        query: str,
        max_results: int = 10,
        include_answer: bool = True,
        include_raw_content: bool = False,
        include_images: bool = False,
    ) -> SearchResponse:
        """
        Search using Tavily API.

        Args:
            query: Search query
            max_results: Maximum number of results (max 20)
            include_answer: Include AI-generated answer
            include_raw_content: Include raw content from top results
            include_images: Include images in results

        Returns:
            SearchResponse with results
        """
        import aiohttp

        if not self.api_key:
            raise ValueError("Tavily API key not configured")

        max_results = min(max_results, 20)

        payload = {
            "api_key": self.api_key,
            "query": query,
            "max_results": max_results,
            "include_answer": include_answer,
            "include_raw_content": include_raw_content,
            "include_images": include_images,
            "search_depth": "basic",
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.base_url}/search",
                json=payload,
                timeout=aiohttp.ClientTimeout(total=30),
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise Exception(
                        f"Tavily API error: {response.status} - {error_text}"
                    )

                data = await response.json()

        results = [
            SearchResult(
                url=result.get("url", ""),
                title=result.get("title", ""),
                content=result.get("content", ""),
                score=result.get("score", 0.0),
                published_date=result.get("published_date"),
            )
            for result in data.get("results", [])
        ]

        return SearchResponse(
            query=query,
            results=results,
            answer=data.get("answer"),
            total_results=len(results),
        )

    async def get_context(
        self,
        query: str,
        max_results: int = 5,
    ) -> str:
        """
        Get context string from Tavily for RAG.

        Args:
            query: Search query
            max_results: Number of results to include

        Returns:
            Formatted context string
        """
        response = await self.search(
            query=query,
            max_results=max_results,
            include_answer=True,
        )

        context_parts = []

        if response.answer:
            context_parts.append(f"Answer: {response.answer}")

        for result in response.results[:max_results]:
            context_parts.append(
                f"Source: {result.title}\nURL: {result.url}\nContent: {result.content}"
            )

        return "\n\n".join(context_parts)


async def get_tavily_client() -> TavilyClient:
    """Get Tavily client instance."""
    return TavilyClient()
