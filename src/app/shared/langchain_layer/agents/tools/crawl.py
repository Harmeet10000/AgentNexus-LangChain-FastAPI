"""LangChain tool for web crawling."""

from typing import Any

from langchain_core.tools import BaseTool

from app.shared.crawler import get_crawler, truncate_content
from app.shared.crawler.processor import (
    SchemaType as ProcessorSchemaType,
)
from app.shared.crawler.processor import (
    get_processor,
)


class CrawlUrlInput(BaseTool):
    """Input schema for crawl URL tool."""

    url: str
    extract_structured: bool = False
    schema_type: str | None = None
    custom_schema: dict[str, Any] | None = None
    summary: bool = False


class CrawlUrlTool(BaseTool):
    """Tool for crawling a URL and extracting content."""

    name: str = "crawl_url"
    description: str = """Crawl a specific URL and extract its content.
    Use this when you have a specific URL to fetch content from.
    Returns markdown content, optionally with structured data extraction or summary.
    """
    args_schema: type[CrawlUrlInput] = CrawlUrlInput

    async def _arun(
        self,
        url: str,
        extract_structured: bool = False,
        schema_type: str | None = None,
        custom_schema: dict[str, Any] | None = None,
        summary: bool = False,
    ) -> str:
        """
        Run the crawl tool.

        Args:
            url: URL to crawl
            extract_structured: Whether to extract structured data
            schema_type: Predefined schema type (product, article, person, job)
            custom_schema: Custom JSON schema
            summary: Whether to generate summary

        Returns:
            Formatted result string
        """
        crawler = await get_crawler()

        result = await crawler.crawl(url=url)

        if not result.success:
            return f"Failed to crawl {url}: {result.error_message}"

        markdown = result.markdown
        if markdown:
            markdown = truncate_content(markdown)

        parts = []

        if result.title:
            parts.append(f"# {result.title}\n")

        if summary and markdown:
            processor = await get_processor()
            summary_result = await processor.summarize(markdown)
            if summary_result.success:
                parts.append(f"## Summary\n{summary_result.summary}\n")

        if extract_structured and markdown:
            processor = await get_processor()
            proc_schema_type = None
            if schema_type:
                try:
                    proc_schema_type = ProcessorSchemaType(schema_type)
                except ValueError:
                    pass

            extraction = await processor.extract_structured(
                content=markdown,
                schema_type=proc_schema_type,
                custom_schema=custom_schema,
            )

            if extraction.success and extraction.extracted_data:
                import json

                parts.append(
                    f"## Extracted Data\n```json\n{json.dumps(extraction.extracted_data, indent=2)}\n```\n"
                )

        if markdown:
            parts.append(f"## Content\n{markdown}")

        if result.links:
            links = [link.get("href", "") for link in result.links[:10]]
            parts.append(
                "\n## Links Found\n" + "\n".join(f"- {link}" for link in links)
            )

        await crawler.close()

        return "\n\n".join(parts)



