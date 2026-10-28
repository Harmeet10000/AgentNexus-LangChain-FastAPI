from __future__ import annotations

import asyncio
import math
import traceback
from datetime import UTC, datetime
from enum import StrEnum
from typing import Protocol

import anyio
import asyncer
import langextract as lx
from pydantic import BaseModel, ConfigDict, Field, SecretStr

from app.observability.logging import logger

from .config import (
    LANGEXTRACT_MODEL_ID,
    LangExtractModelId,
    validate_model_id,
)


class ExtractionBudget(BaseModel):
    """Checkpoint-safe per-document extraction budget."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    # Three provider passes per contract, with room for eight contracts by
    # default.  The call cap remains the hard total ceiling across retries.
    max_model_calls: int = Field(default=24, ge=0)
    max_contracts: int = Field(default=8, ge=0)
    max_input_tokens: int = Field(default=200_000, ge=0)
    max_output_tokens: int = Field(default=100_000, ge=0)
    max_cost_usd: float = Field(default=10.0, ge=0)
    cost_per_1k_tokens_usd: float = Field(default=0.0001, ge=0)
    wall_clock_seconds: float = Field(default=900.0, gt=0)
    model_calls: int = Field(default=0, ge=0)
    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)
    cost_estimate_usd: float = Field(default=0.0, ge=0)
    started_at: datetime = Field(default_factory=lambda: datetime.now(tz=UTC))
    contracts_started: tuple[int, ...] = ()


class ExtractionBudgetExhaustedError(RuntimeError):
    """Typed fail-closed signal for extraction/celery budget exhaustion."""

    def __init__(self, dimension: str, message: str) -> None:
        super().__init__(message)
        self.dimension = dimension


class ExtractionBudgetDimension(StrEnum):
    """Closed set of extraction budget dimensions exposed to callers."""

    MODEL_CALLS = "model_calls"
    INPUT_TOKENS = "input_tokens"
    OUTPUT_TOKENS = "output_tokens"
    COST = "cost"
    WALL_CLOCK = "wall_clock"
    CONTRACTS = "contracts"


class ExtractionRequest(BaseModel):
    """One extraction operation derived from a parsed document."""

    model_config = ConfigDict(extra="forbid", frozen=True, arbitrary_types_allowed=True)

    text: str
    prompt_description: str
    examples: tuple[lx.data.ExampleData, ...] = ()
    model_id: LangExtractModelId = LANGEXTRACT_MODEL_ID
    extraction_passes: int = 3
    max_workers: int = 12
    max_char_buffer: int = Field(default=1000, ge=1)
    # Stable provenance carried through provider calls.  These fields make a
    # multi-contract source an explicit set of contracts rather than one
    # whole-document extraction with ambiguous ownership.
    document_id: str | None = None
    tenant_id: str | None = None
    document_version: int = 1
    contract_index: int = 0
    budget: ExtractionBudget = Field(default_factory=ExtractionBudget)


class ExtractionProvider(Protocol):
    """Synchronous provider contract dispatched outside the event loop."""

    def extract(self, request: ExtractionRequest) -> tuple[lx.data.AnnotatedDocument, ...]: ...


# Kept as a type-level alias for existing callers; the provider API itself is
# unchanged and remains synchronous behind the async facade.
type SyncExtractionProvider = ExtractionProvider


class ExtractionFailureCode(StrEnum):
    """Machine-readable reasons that extraction produced no usable result."""

    UNCONFIGURED = "unconfigured"
    PROVIDER_ERROR = "provider_error"
    GRAPH_WRITE_ERROR = "graph_write_error"
    BOUNDARY_AMBIGUOUS = "contract_boundary_ambiguous"
    BUDGET_EXHAUSTED = "budget_exhausted"


class ExtractionSucceeded(BaseModel):
    """A successful provider call, including a valid empty extraction."""

    model_config = ConfigDict(extra="forbid", frozen=True, arbitrary_types_allowed=True)

    documents: tuple[lx.data.AnnotatedDocument, ...]
    budget: ExtractionBudget = Field(default_factory=ExtractionBudget)
    extraction_classes: tuple[str, ...] = ()


class ExtractionFailed(BaseModel):
    """A visible, typed provider failure for the ingestion layer to handle."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    code: ExtractionFailureCode
    message: str
    budget_dimension: ExtractionBudgetDimension | None = None
    budget: ExtractionBudget = Field(default_factory=ExtractionBudget)


type ExtractionOutcome = ExtractionSucceeded | ExtractionFailed


class LangExtractClient:
    """Synchronous adapter around LangExtract's functional provider API."""

    __slots__ = ("_api_key", "_model_id")

    def __init__(self, api_key: str, model_id: LangExtractModelId = LANGEXTRACT_MODEL_ID) -> None:
        self._api_key = api_key
        self._model_id = validate_model_id(model_id)

    def extract(self, request: ExtractionRequest) -> tuple[lx.data.AnnotatedDocument, ...]:
        # LangExtract 1.7.1 forwards ``language_model_params`` to the
        # provider.  Gemini's ``HttpOptions.timeout`` is milliseconds and is
        # enforced by the network client, so a provider-side timeout remains
        # in force even when the surrounding worker abandons this sync call.
        remaining_seconds = max(
            0.001,
            request.budget.wall_clock_seconds
            - max(0.0, (datetime.now(tz=UTC) - request.budget.started_at).total_seconds()),
        )
        raw = lx.extract(
            text_or_documents=request.text,
            prompt_description=request.prompt_description,
            examples=list(request.examples),
            model_id=self._model_id,
            extraction_passes=request.extraction_passes,
            max_workers=request.max_workers,
            max_char_buffer=request.max_char_buffer,
            api_key=self._api_key,
            language_model_params={
                "http_options": {"timeout": max(1, int(remaining_seconds * 1000))},
                # The application budget owns retries.  LangExtract's own
                # retry loop otherwise performs unaccounted calls after the
                # request's model-call budget has been charged.
                "max_retries": 0,
            },
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

    # A cancelled ``to_thread`` call cannot stop arbitrary synchronous Python;
    # it continues in the worker thread.  Keep those abandoned calls bounded
    # independently of AnyIO's process-wide default limiter.
    ABANDONED_CALL_CAPACITY = 4

    __slots__ = ("_limiter", "_provider")

    def __init__(self, provider: ExtractionProvider | None) -> None:
        self._provider = provider
        self._limiter = anyio.CapacityLimiter(self.ABANDONED_CALL_CAPACITY)

    @classmethod
    def from_settings(cls, settings: object) -> AsyncExtractionService:
        secret = getattr(settings, "LANGEXTRACT_API_KEY", SecretStr(""))
        api_key = secret.get_secret_value().strip()
        if not api_key or api_key == "empty-langextract-api-key":
            return cls(provider=None)
        model_id = validate_model_id(getattr(settings, "LANGEXTRACT_MODEL", LANGEXTRACT_MODEL_ID))
        return cls(provider=LangExtractClient(api_key, model_id=model_id))

    @property
    def configured(self) -> bool:
        return self._provider is not None

    async def extract(self, request: ExtractionRequest) -> ExtractionOutcome:
        provider = self._provider
        if provider is None:
            return ExtractionFailed(
                code=ExtractionFailureCode.UNCONFIGURED,
                message="LangExtract provider is not configured",
                budget=request.budget,
            )

        budget = request.budget
        now = datetime.now(tz=UTC)
        elapsed = max(0.0, (now - budget.started_at).total_seconds())
        chunk_count = max(1, math.ceil(len(request.text) / request.max_char_buffer))
        model_calls = chunk_count * max(1, request.extraction_passes)
        is_new_contract = request.contract_index not in budget.contracts_started
        if is_new_contract and len(budget.contracts_started) >= budget.max_contracts:
            return ExtractionFailed(
                code=ExtractionFailureCode.BUDGET_EXHAUSTED,
                message="LangExtract contract budget exhausted",
                budget_dimension=ExtractionBudgetDimension.CONTRACTS,
                budget=budget,
            )
        if budget.model_calls + model_calls > budget.max_model_calls:
            return ExtractionFailed(
                code=ExtractionFailureCode.BUDGET_EXHAUSTED,
                message="LangExtract model-call budget exhausted",
                budget_dimension=ExtractionBudgetDimension.MODEL_CALLS,
                budget=budget,
            )
        if elapsed >= budget.wall_clock_seconds:
            return ExtractionFailed(
                code=ExtractionFailureCode.BUDGET_EXHAUSTED,
                message="LangExtract wall-clock budget exhausted",
                budget_dimension=ExtractionBudgetDimension.WALL_CLOCK,
                budget=budget,
            )
        # LangExtract sends one prompt per source chunk and per pass.  Charge
        # the configured model-call count against a representative chunk,
        # rather than multiplying the entire document by every call (which
        # would reject otherwise valid chunked documents far too early).
        chunk_chars = min(len(request.text), request.max_char_buffer)
        input_tokens = max(1, (chunk_chars + len(request.prompt_description)) // 4)
        input_tokens *= model_calls
        if budget.input_tokens + input_tokens > budget.max_input_tokens:
            return ExtractionFailed(
                code=ExtractionFailureCode.BUDGET_EXHAUSTED,
                message="LangExtract input-token budget exhausted",
                budget_dimension=ExtractionBudgetDimension.INPUT_TOKENS,
                budget=budget,
            )
        if budget.output_tokens >= budget.max_output_tokens:
            return ExtractionFailed(
                code=ExtractionFailureCode.BUDGET_EXHAUSTED,
                message="LangExtract output-token budget exhausted",
                budget_dimension=ExtractionBudgetDimension.OUTPUT_TOKENS,
                budget=budget,
            )
        if budget.cost_estimate_usd >= budget.max_cost_usd:
            return ExtractionFailed(
                code=ExtractionFailureCode.BUDGET_EXHAUSTED,
                message="LangExtract cost budget exhausted",
                budget_dimension=ExtractionBudgetDimension.COST,
                budget=budget,
            )

        attempted = budget.model_copy(
            update={
                "model_calls": budget.model_calls + model_calls,
                "input_tokens": budget.input_tokens + input_tokens,
                "cost_estimate_usd": budget.cost_estimate_usd
                + input_tokens / 1000 * budget.cost_per_1k_tokens_usd,
                "contracts_started": (
                    (*budget.contracts_started, request.contract_index)
                    if is_new_contract
                    else budget.contracts_started
                ),
            }
        )

        try:
            async with asyncio.timeout(max(0.001, budget.wall_clock_seconds - elapsed)):
                documents = await asyncer.asyncify(
                    function=provider.extract,
                    abandon_on_cancel=True,
                    limiter=self._limiter,
                )(request)
        except TimeoutError:
            return ExtractionFailed(
                code=ExtractionFailureCode.BUDGET_EXHAUSTED,
                message="LangExtract wall-clock budget exhausted",
                budget_dimension=ExtractionBudgetDimension.WALL_CLOCK,
                budget=attempted,
            )
        except Exception as exc:  # noqa: BLE001 -- third-party provider boundary
            exc.add_note(traceback.format_exc())
            logger.bind(
                operation="langextract_extract",
                diagnostic_notes=list(getattr(exc, "__notes__", [])),
            ).error("LangExtract provider failed")
            return ExtractionFailed(
                code=ExtractionFailureCode.PROVIDER_ERROR,
                message="LangExtract provider failed",
                budget=attempted,
            )
        output_tokens = max(
            0,
            sum(
                len(getattr(extraction, "extraction_text", "")) // 4
                for document in documents
                for extraction in document.extractions
            ),
        )
        updated = attempted.model_copy(
            update={
                "output_tokens": attempted.output_tokens + output_tokens,
                "cost_estimate_usd": attempted.cost_estimate_usd
                + output_tokens / 1000 * budget.cost_per_1k_tokens_usd,
            }
        )
        if (
            updated.output_tokens > updated.max_output_tokens
            or updated.cost_estimate_usd > updated.max_cost_usd
        ):
            return ExtractionFailed(
                code=ExtractionFailureCode.BUDGET_EXHAUSTED,
                message="LangExtract output/cost budget exhausted",
                budget_dimension=(
                    ExtractionBudgetDimension.OUTPUT_TOKENS
                    if updated.output_tokens > updated.max_output_tokens
                    else ExtractionBudgetDimension.COST
                ),
                budget=updated,
            )
        return ExtractionSucceeded(documents=documents, budget=updated)
