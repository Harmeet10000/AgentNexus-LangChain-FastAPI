"""Gemini processing for content extraction and summarization."""

import asyncio
import copy
import json
import re
from enum import StrEnum
from typing import Any

from langchain_core.language_models import BaseChatModel
from pydantic import BaseModel
from returns.result import Failure, Success

from app.shared.langchain_layer.models import _build_chat_model

from .errors import CrawlerProcessingResult, CrawlerProcessingValidationError


class SchemaType(StrEnum):
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


class ExtractionResult(BaseModel):
    """Result from Gemini extraction."""

    success: bool
    extracted_data: dict[str, Any] | None = None
    summary: str | None = None
    error: str | None = None
    tokens_used: int | None = None


class GeminiProcessor:
    """Processor for Gemini-based content extraction and summarization."""

    def __init__(self, model: BaseChatModel | None = None):
        self.model = model or _build_chat_model()

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

            response = await self._ainvoke(prompt)
            summary = _response_text(response)

            bounded_summary = summary.strip()[:max_length]
            return ExtractionResult(
                success=True,
                summary=bounded_summary,
            )
        except Exception as e:  # noqa: BLE001 - DTO boundary preserves crawler error contract.
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
        schema_result = _resolve_extraction_schema(schema_type, custom_schema)
        if isinstance(schema_result, Failure):
            return ExtractionResult(success=False, error=schema_result.failure().message)
        schema, _schema_name = schema_result.unwrap()

        try:
            return await self._do_extract_structured(content, schema)
        except Exception as e:  # noqa: BLE001 - DTO boundary preserves crawler error contract.
            return ExtractionResult(
                success=False,
                error=str(e),
            )

    async def _ainvoke(self, prompt: str) -> object:
        """Invoke the model without blocking the event loop."""
        ainvoke = getattr(self.model, "ainvoke", None)
        if ainvoke is not None:
            return await ainvoke(prompt)
        return await asyncio.to_thread(self.model.invoke, prompt)

    async def _do_extract_structured(
        self,
        content: str,
        schema: dict[str, Any],
    ) -> ExtractionResult:
        schema_json = json.dumps(schema, indent=2)
        prompt_content = _truncate_to_token_budget(content, max_tokens=4_000)
        prompt = f"""You are a data extraction assistant. Extract information from the
        following content and format it as JSON according to the provided schema.

        Schema:
        {schema_json}

        <untrusted_content>
        {prompt_content}
        </untrusted_content>

        Output ONLY valid JSON, no other text. If a field cannot be found, use null.
        JSON:"""
        response = await self._ainvoke(prompt)
        response_text = _response_text(response)
        extraction_result = _parse_extraction_json(response_text)
        if isinstance(extraction_result, Failure):
            return ExtractionResult(success=False, error=extraction_result.failure().message)
        extracted_data = extraction_result.unwrap()
        validation_error = _validate_against_schema(extracted_data, schema)
        if validation_error is not None:
            return ExtractionResult(success=False, error=validation_error)
        return ExtractionResult(
            success=True,
            extracted_data=extracted_data,
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
        extract_result = await self.extract_structured(content, schema_type, custom_schema)

        if not extract_result.success:
            return extract_result

        summary_result = await self.summarize(content)
        if not summary_result.success:
            return ExtractionResult(
                success=False,
                extracted_data=extract_result.extracted_data,
                error=summary_result.error or "Summary generation failed",
            )

        return ExtractionResult(
            success=True,
            extracted_data=extract_result.extracted_data,
            summary=summary_result.summary,
        )


def get_schema_for_type(schema_type: SchemaType) -> dict[str, Any] | None:
    """Get predefined schema for a type."""
    schema = PREDEFINED_SCHEMAS.get(schema_type)
    return copy.deepcopy(schema) if schema is not None else None


def _resolve_extraction_schema(
    schema_type: SchemaType | None,
    custom_schema: dict[str, Any] | None,
) -> CrawlerProcessingResult[tuple[dict[str, Any], str]]:
    if schema_type is not None and schema_type != SchemaType.CUSTOM:
        schema = PREDEFINED_SCHEMAS.get(schema_type)
        if schema is not None:
            return Success((copy.deepcopy(schema), schema_type.value))
        return Failure(
            CrawlerProcessingValidationError(
                message=f"Unknown schema type: {schema_type}",
                details={"schema_type": schema_type.value},
                source="crawler_processor",
            )
        )

    if custom_schema:
        return Success((copy.deepcopy(custom_schema), "custom"))

    return Failure(
        CrawlerProcessingValidationError(
            message="No schema provided. Use schema_type or custom_schema.",
            source="crawler_processor",
        )
    )


def _response_text(response: object) -> str:
    content = getattr(response, "content", response)
    if isinstance(content, str):
        return content
    return str(content)


def _truncate_to_token_budget(content: str, *, max_tokens: int) -> str:
    """Bound prompt input with a deterministic tokenizer-independent estimate."""
    tokens = re.findall(r"\w+|[^\w\s]|\s+", content, flags=re.UNICODE)
    if len(tokens) <= max_tokens:
        return content
    return "".join(tokens[:max_tokens]).rstrip() + "\n[content truncated]"


def _parse_extraction_json(response_text: str) -> CrawlerProcessingResult[dict[str, Any]]:
    cleaned = response_text.strip()
    if cleaned.startswith("```") and cleaned.endswith("```"):
        cleaned = cleaned.split("\n", maxsplit=1)[-1][:-3].strip()
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        return Failure(
            CrawlerProcessingValidationError(
                message=f"Failed to parse JSON: {exc!s}",
                source="crawler_processor",
            )
        )

    if not isinstance(parsed, dict):
        return Failure(
            CrawlerProcessingValidationError(
                message="Structured extraction response must be a JSON object.",
                source="crawler_processor",
            )
        )

    return Success(parsed)


def _validate_against_schema(data: dict[str, Any], schema: dict[str, Any]) -> str | None:
    """Validate the JSON-schema subset used by the predefined crawler schemas."""
    required = schema.get("required", [])
    missing = [field for field in required if field not in data or data[field] is None]
    if missing:
        return f"Structured extraction is missing required fields: {', '.join(missing)}"

    properties = schema.get("properties", {})
    for field, definition in properties.items():
        value = data.get(field)
        if value is None or not isinstance(definition, dict):
            continue
        expected = definition.get("type")
        if expected == "string" and not isinstance(value, str):
            return f"Structured extraction field '{field}' must be a string"
        if expected == "integer" and (not isinstance(value, int) or isinstance(value, bool)):
            return f"Structured extraction field '{field}' must be an integer"
        if expected == "number" and (
            not isinstance(value, (int, float)) or isinstance(value, bool)
        ):
            return f"Structured extraction field '{field}' must be a number"
        if expected == "array" and not isinstance(value, list):
            return f"Structured extraction field '{field}' must be an array"
    return None


async def get_processor() -> GeminiProcessor:
    """Get a Gemini processor instance."""
    return GeminiProcessor()
