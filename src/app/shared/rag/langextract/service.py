from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol

import asyncer
import langextract as lx
from pydantic import SecretStr


class LangExtractSettings(Protocol):
    """Settings surface needed to construct the extraction client."""

    LANGEXTRACT_API_KEY: SecretStr


@dataclass(frozen=True, slots=True)
class ExtractionRequest:
    """One extraction operation derived from a parsed document."""

    text: str
    prompt_description: str
    examples: tuple[lx.data.ExampleData, ...] = ()
    model_id: str = "gemini-2.5-flash"
    extraction_passes: int = 3
    max_workers: int = 12


class ExtractionFailureCode(StrEnum):
    """Machine-readable reasons that extraction produced no usable result."""

    UNCONFIGURED = "unconfigured"
    PROVIDER_ERROR = "provider_error"


@dataclass(frozen=True, slots=True)
class ExtractionSucceeded:
    """A successful provider call, including a valid empty extraction."""

    documents: tuple[lx.data.AnnotatedDocument, ...]


@dataclass(frozen=True, slots=True)
class ExtractionFailed:
    """A visible, typed provider failure for the ingestion layer to handle."""

    code: ExtractionFailureCode
    message: str


type ExtractionOutcome = ExtractionSucceeded | ExtractionFailed


class SyncExtractionProvider(Protocol):
    """Blocking provider contract kept behind the async service boundary."""

    def extract(self, request: ExtractionRequest) -> tuple[lx.data.AnnotatedDocument, ...]: ...


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

    def __init__(self, provider: SyncExtractionProvider | None) -> None:
        self._provider = provider

    @classmethod
    def from_settings(cls, settings: LangExtractSettings) -> AsyncExtractionService:
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
        except Exception:  # noqa: BLE001 -- third-party provider boundary
            return ExtractionFailed(
                code=ExtractionFailureCode.PROVIDER_ERROR,
                message="LangExtract provider failed",
            )
        return ExtractionSucceeded(documents=documents)
