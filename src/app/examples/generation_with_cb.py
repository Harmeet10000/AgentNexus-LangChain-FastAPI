"""FastAPI example showing the provider-neutral text-generation Seam."""

from __future__ import annotations

from fastapi import APIRouter, Depends, Request
from returns.result import Failure

from app.ai.generation import (  # noqa: TC001 — FastAPI resolves get_type_hints at runtime
    TextGenerationRequest,
    TextGenerator,
)
from app.utils import ServiceUnavailableException

router = APIRouter()


def get_text_generator(request: Request) -> TextGenerator:
    """Read the process-scoped generator provisioned by the application lifespan."""
    generator = getattr(request.app.state, "text_generator", None)
    if generator is None:
        msg = "Text generation provider is unavailable"
        raise ServiceUnavailableException(msg)
    return generator


@router.post("/generate-text")
async def generate_text(
    generator: TextGenerator = Depends(get_text_generator),
) -> dict[str, object]:
    result = await generator.generate(TextGenerationRequest(prompt="Generate text."))
    if isinstance(result, Failure):
        msg = "External API call failed"
        raise ServiceUnavailableException(msg)

    return {"generated_text": result.unwrap().text}
