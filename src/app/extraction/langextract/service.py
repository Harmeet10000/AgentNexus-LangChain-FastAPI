from __future__ import annotations

import traceback
from enum import StrEnum
from typing import Protocol

import asyncer
import langextract as lx
from pydantic import BaseModel, ConfigDict, SecretStr

from app.observability.logging import logger


class ExtractionRequest(BaseModel):
    """One extraction operation derived from a parsed document."""

    model_config = ConfigDict(extra="forbid", frozen=True, arbitrary_types_allowed=True)

    text: str
    prompt_description: str
    examples: tuple[lx.data.ExampleData, ...] = ()
    model_id: str = "gemini-2.5-flash"
    extraction_passes: int = 3
    max_workers: int = 12


class ExtractionProvider(Protocol):
    """Synchronous provider contract dispatched outside the event loop."""

    def extract(self, request: ExtractionRequest) -> tuple[lx.data.AnnotatedDocument, ...]: ...


class ExtractionFailureCode(StrEnum):
    """Machine-readable reasons that extraction produced no usable result."""

    UNCONFIGURED = "unconfigured"
    PROVIDER_ERROR = "provider_error"
    GRAPH_WRITE_ERROR = "graph_write_error"


class ExtractionSucceeded(BaseModel):
    """A successful provider call, including a valid empty extraction."""

    model_config = ConfigDict(extra="forbid", frozen=True, arbitrary_types_allowed=True)

    documents: tuple[lx.data.AnnotatedDocument, ...]


class ExtractionFailed(BaseModel):
    """A visible, typed provider failure for the ingestion layer to handle."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    code: ExtractionFailureCode
    message: str


type ExtractionOutcome = ExtractionSucceeded | ExtractionFailed


class LangExtractClient:
    """Synchronous adapter around LangExtract's functional provider API."""

    __slots__ = ("_api_key",)

    def __init__(self, api_key: str) -> None:
        self._api_key = api_key

    def extract(self, request: ExtractionRequest) -> tuple[lx.data.AnnotatedDocument, ...]:
        raw = lx.extract(
            text_or_documents=request.text,
            prompt_description=request.prompt_description,
            examples=list(request.examples),
            model_id=request.model_id,
            extraction_passes=request.extraction_passes,
            max_workers=request.max_workers,
            api_key=self._api_key,
        )
        if isinstance(raw, list):
            return tuple(raw)
        return (raw,)


class AsyncExtractionService:
    """Async facade for a blocking extraction provider.

    The provider is constructed once with other lifespan resources. Calls are
    dispatched to AnyIO's worker pool so LangExtract cannot block the event
    loop used by the API server or ingestion worker.
    """

    __slots__ = ("_provider",)

    def __init__(self, provider: ExtractionProvider | None) -> None:
        self._provider = provider

    @classmethod
    def from_settings(cls, settings: object) -> AsyncExtractionService:
        secret = getattr(settings, "LANGEXTRACT_API_KEY", SecretStr(""))
        api_key = secret.get_secret_value().strip()
        if not api_key or api_key == "empty-langextract-api-key":
            return cls(provider=None)
        return cls(provider=LangExtractClient(api_key))

    @property
    def configured(self) -> bool:
        return self._provider is not None

    async def extract(self, request: ExtractionRequest) -> ExtractionOutcome:
        provider = self._provider
        if provider is None:
            return ExtractionFailed(
                code=ExtractionFailureCode.UNCONFIGURED,
                message="LangExtract provider is not configured",
            )

        try:
            documents = await asyncer.asyncify(function=provider.extract)(request)
        except Exception as exc:  # noqa: BLE001 -- third-party provider boundary
            exc.add_note(traceback.format_exc())
            logger.bind(
                operation="langextract_extract",
                diagnostic_notes=list(getattr(exc, "__notes__", [])),
            ).error("LangExtract provider failed")
            return ExtractionFailed(
                code=ExtractionFailureCode.PROVIDER_ERROR,
                message="LangExtract provider failed",
            )
        return ExtractionSucceeded(documents=documents)
