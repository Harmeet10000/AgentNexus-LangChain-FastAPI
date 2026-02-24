"""Gemini processing for content extraction and summarization."""

import json
from dataclasses import dataclass
from enum import Enum
from typing import Any

from langchain_google_genai import ChatGoogleGenerativeAI

from app.config.settings import get_settings


class SchemaType(str, Enum):
    """Predefined schema types for structured extraction."""

    PRODUCT = "product"
    ARTICLE = "article"
    PERSON = "person"
    JOB = "job"
    CUSTOM = "custom"


PREDEFINED_SCHEMAS = {
    SchemaType.PRODUCT: {
        "type": "object",
        "properties": {
            "name": {"type": "string", "description": "Product name"},
            "price": {"type": "string", "description": "Product price"},
            "description": {"type": "string", "description": "Product description"},
            "sku": {"type": "string", "description": "Product SKU or model number"},
            "availability": {"type": "string", "description": "Availability status"},
            "brand": {"type": "string", "description": "Brand name"},
            "category": {"type": "string", "description": "Product category"},
            "rating": {"type": "number", "description": "Average rating"},
            "reviews_count": {"type": "integer", "description": "Number of reviews"},
            "images": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Image URLs",
            },
        },
        "required": ["name", "price"],
    },
    SchemaType.ARTICLE: {
        "type": "object",
        "properties": {
            "title": {"type": "string", "description": "Article title"},
            "author": {"type": "string", "description": "Author name"},
            "publish_date": {"type": "string", "description": "Publication date"},
            "summary": {"type": "string", "description": "Article summary"},
            "content": {"type": "string", "description": "Full content"},
            "tags": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Tags",
            },
            "category": {"type": "string", "description": "Category"},
            "read_time_minutes": {
                "type": "integer",
                "description": "Estimated read time",
            },
            "source": {"type": "string", "description": "Source name"},
        },
        "required": ["title", "author"],
    },
    SchemaType.PERSON: {
        "type": "object",
        "properties": {
            "name": {"type": "string", "description": "Person's full name"},
            "title": {"type": "string", "description": "Job title or role"},
            "bio": {"type": "string", "description": "Biography"},
            "company": {"type": "string", "description": "Company/organization"},
            "email": {"type": "string", "description": "Email address"},
            "phone": {"type": "string", "description": "Phone number"},
            "location": {"type": "string", "description": "Location"},
            "social_links": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Social media links",
            },
            "skills": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Skills or expertise",
            },
        },
        "required": ["name"],
    },
    SchemaType.JOB: {
        "type": "object",
        "properties": {
            "title": {"type": "string", "description": "Job title"},
            "company": {"type": "string", "description": "Company name"},
            "location": {"type": "string", "description": "Job location"},
            "salary_min": {"type": "integer", "description": "Minimum salary"},
            "salary_max": {"type": "integer", "description": "Maximum salary"},
            "salary_currency": {"type": "string", "description": "Currency code"},
            "job_type": {
                "type": "string",
                "description": "Full-time, part-time, contract",
            },
            "description": {"type": "string", "description": "Job description"},
            "requirements": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Job requirements",
            },
            "benefits": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Benefits offered",
            },
            "posted_date": {"type": "string", "description": "Date posted"},
            "apply_url": {"type": "string", "description": "URL to apply"},
        },
        "required": ["title", "company"],
    },
}


@dataclass
class ExtractionResult:
    """Result from Gemini extraction."""

    success: bool
    extracted_data: dict[str, Any] | None = None
    summary: str | None = None
    error: str | None = None
    tokens_used: int | None = None


class GeminiProcessor:
    """Processor for Gemini-based content extraction and summarization."""

    def __init__(self):
        settings = get_settings()
        self.model = ChatGoogleGenerativeAI(
            model=settings.GEMINI_MODEL,
            temperature=0.3,
            max_tokens=4096,
        )

    async def summarize(
        self,
        content: str,
        max_length: int = 500,
    ) -> ExtractionResult:
        """
        Summarize content using Gemini.

        Args:
            content: Content to summarize
            max_length: Maximum summary length in characters

        Returns:
            ExtractionResult with summary
        """
        try:
            prompt = f"""You are a helpful assistant. Summarize the following content 
            in a concise way (maximum {max_length} characters). 
            Focus on the main points and key information.
            
            Content:
            {content}
            
            Summary:"""

            response = self.model.invoke(prompt)
            summary = (
                response.content if hasattr(response, "content") else str(response)
            )

            return ExtractionResult(
                success=True,
                summary=summary.strip(),
            )
        except Exception as e:
            return ExtractionResult(
                success=False,
                error=str(e),
            )

    async def extract_structured(
        self,
        content: str,
        schema_type: SchemaType | None = None,
        custom_schema: dict[str, Any] | None = None,
    ) -> ExtractionResult:
        """
        Extract structured data from content using Gemini.

        Args:
            content: Content to extract from
            schema_type: Predefined schema type
            custom_schema: Custom JSON schema

        Returns:
            ExtractionResult with extracted data
        """
        if schema_type and schema_type != SchemaType.CUSTOM:
            schema = PREDEFINED_SCHEMAS.get(schema_type)
            schema_name = schema_type.value
        elif custom_schema:
            schema = custom_schema
            schema_name = "custom"
        else:
            return ExtractionResult(
                success=False,
                error="No schema provided. Use schema_type or custom_schema.",
            )

        if not schema:
            return ExtractionResult(
                success=False,
                error=f"Unknown schema type: {schema_type}",
            )

        try:
            schema_json = json.dumps(schema, indent=2)

            prompt = f"""You are a data extraction assistant. Extract information from the 
            following content and format it as JSON according to the provided schema.
            
            Schema:
            {schema_json}
            
            Content:
            {content[:15000]}
            
            Output ONLY valid JSON, no other text. If a field cannot be found, use null.
            JSON:"""

            response = self.model.invoke(prompt)
            response_text = (
                response.content if hasattr(response, "content") else str(response)
            )

            extracted_data = json.loads(response_text.strip())

            return ExtractionResult(
                success=True,
                extracted_data=extracted_data,
            )
        except json.JSONDecodeError as e:
            return ExtractionResult(
                success=False,
                error=f"Failed to parse JSON: {str(e)}",
            )
        except Exception as e:
            return ExtractionResult(
                success=False,
                error=str(e),
            )

    async def extract_and_summarize(
        self,
        content: str,
        schema_type: SchemaType | None = None,
        custom_schema: dict[str, Any] | None = None,
    ) -> ExtractionResult:
        """
        Extract structured data AND create a summary.

        Args:
            content: Content to process
            schema_type: Predefined schema type
            custom_schema: Custom JSON schema

        Returns:
            ExtractionResult with both extracted data and summary
        """
        extract_result = await self.extract_structured(
            content, schema_type, custom_schema
        )

        if not extract_result.success:
            return extract_result

        summary_result = await self.summarize(content)

        return ExtractionResult(
            success=True,
            extracted_data=extract_result.extracted_data,
            summary=summary_result.summary,
        )


def get_schema_for_type(schema_type: SchemaType) -> dict[str, Any] | None:
    """Get predefined schema for a type."""
    return PREDEFINED_SCHEMAS.get(schema_type)


async def get_processor() -> GeminiProcessor:
    """Get a Gemini processor instance."""
    return GeminiProcessor()
