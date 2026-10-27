"""Validated runtime configuration for the LangExtract adapter."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, cast

if TYPE_CHECKING:
    from typing import Final

LANGEXTRACT_MODEL_ID: Final = "gemini-3.5-flash-lite"
type LangExtractModelId = Literal["gemini-3.5-flash-lite"]


def validate_model_id(value: object) -> LangExtractModelId:
    """Accept only the production LangExtract model selected by this app.

    LangExtract routes Gemini by the ``model_id`` string.  Keeping this check at
    the application boundary prevents a stale environment variable or caller
    override from silently moving extraction to a weaker/deprecated model.
    """

    if value != LANGEXTRACT_MODEL_ID:
        message = (
            "LANGEXTRACT_MODEL must be the supported extraction model "
            f"{LANGEXTRACT_MODEL_ID!r}; got {value!r}"
        )
        raise ValueError(message)
    return cast("LangExtractModelId", value)
