"""Value objects shared by text-generation adapters."""

from pydantic import BaseModel, ConfigDict, Field


class TextGenerationRequest(BaseModel):
    """Provider-neutral input for one text-generation request.

    The free-text path stays the same as before: ``prompt`` plus an optional
    ``system`` message, everything else ``None``. The structured path sets
    ``response_schema`` to the Pydantic model the provider output must parse
    into; it is consumed through ``TextGenerator.generate_structured``.

    Request-level ``temperature`` (bounded 0.0-2.0, the same band the settings
    validator applies to GEMINI_TEMPERATURE) and ``model`` overrides are honored
    by adapters that can rebuild a provider model; adapters that cannot raise a
    local defect instead of silently ignoring the override.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    prompt: str = Field(min_length=1)
    system: str | None = None
    response_schema: type[BaseModel] | None = None
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    model: str | None = Field(default=None, min_length=1)


class GeneratedText(BaseModel):
    """Provider-neutral successful text-generation output."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    text: str
    # Token usage when the provider reports it; test doubles may script a
    # response without it, so absence stays a valid state.
    input_tokens: int | None = None
    output_tokens: int | None = None


class GeneratedStructured(BaseModel):
    """Provider-neutral successful schema-bound generation output.

    ``output`` is the parsed instance of the request's ``response_schema`` —
    not a re-serialization of it — so consumers receive their own field types,
    validators, and narrowing for free up to the parse.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    output: BaseModel
    input_tokens: int | None = None
    output_tokens: int | None = None
