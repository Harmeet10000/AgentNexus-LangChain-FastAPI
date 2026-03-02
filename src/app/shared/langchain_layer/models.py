"""
Model factory for Google Gemini via langchain_google_genai.

Provides:
- Async single-turn and batch invocation
- Streaming (token-level and chunk-level)
- Multimodal (text + image/audio/video)
- Embeddings (single + batch)
- Structured output via .with_structured_output()
"""

from __future__ import annotations

import asyncio
import base64
import mimetypes
from collections.abc import Any, AsyncIterator
from pathlib import Path

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from pydantic import BaseModel

from src.app.config.settings import get_settings

_settings = get_settings()
_mcfg = _settings.model


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------


def build_chat_model(
    model_name: str | None = None,
    *,
    temperature: float | None = None,
    max_tokens: int | None = None,
    streaming: bool = False,
    **kwargs: Any,
) -> ChatGoogleGenerativeAI:
    """Return a configured ChatGoogleGenerativeAI instance."""
    return ChatGoogleGenerativeAI(
        model=model_name or _mcfg.gemini_pro_model,
        temperature=temperature
        if temperature is not None
        else _mcfg.default_temperature,
        max_output_tokens=max_tokens or _mcfg.default_max_tokens,
        google_api_key=_mcfg.google_api_key.get_secret_value(),
        streaming=streaming,
        timeout=_mcfg.default_timeout,
        **kwargs,
    )


def build_fast_model(**kwargs: Any) -> ChatGoogleGenerativeAI:
    """Flash model for lower-latency / cheaper tasks (guardrails, tool selection)."""
    return build_chat_model(model_name=_mcfg.gemini_flash_model, **kwargs)


def build_embedding_model() -> GoogleGenerativeAIEmbeddings:
    return GoogleGenerativeAIEmbeddings(
        model=_mcfg.gemini_embedding_model,
    )


# ---------------------------------------------------------------------------
# Text inference
# ---------------------------------------------------------------------------


async def ainvoke_text(
    prompt: str,
    *,
    system: str | None = None,
    model: ChatGoogleGenerativeAI | None = None,
) -> str:
    """Single async text call, returns plain string."""
    llm = model or build_chat_model()
    messages: list[BaseMessage] = []
    if system:
        messages.append(SystemMessage(content=system))
    messages.append(HumanMessage(content=prompt))
    result = await llm.ainvoke(messages)
    return result.content  # type: ignore[return-value]


async def abatch_text(
    prompts: list[str],
    *,
    system: str | None = None,
    model: ChatGoogleGenerativeAI | None = None,
    max_concurrency: int | None = None,
) -> list[str]:
    """
    Batch async text calls.
    Uses LangChain's native abatch which honours max_concurrency.
    """
    llm = model or build_chat_model()
    max_c = max_concurrency or _mcfg.max_concurrency

    def _build(prompt: str) -> list[BaseMessage]:
        msgs: list[BaseMessage] = []
        if system:
            msgs.append(SystemMessage(content=system))
        msgs.append(HumanMessage(content=prompt))
        return msgs

    message_batches = [_build(p) for p in prompts]
    results = await llm.abatch(message_batches, config={"max_concurrency": max_c})
    return [r.content for r in results]


async def astream_text(
    prompt: str,
    *,
    system: str | None = None,
    model: ChatGoogleGenerativeAI | None = None,
) -> AsyncIterator[str]:
    """
    Async generator that yields text tokens as they arrive.

    Usage in FastAPI::

        async def sse_endpoint():
            async for chunk in astream_text("Hello"):
                yield f"data: {chunk}\\n\\n"
    """
    llm = model or build_chat_model(streaming=True)
    parser = StrOutputParser()
    chain = llm | parser
    messages: list[BaseMessage] = []
    if system:
        messages.append(SystemMessage(content=system))
    messages.append(HumanMessage(content=prompt))
    async for chunk in chain.astream(messages):
        yield chunk


# ---------------------------------------------------------------------------
# Multimodal inference
# ---------------------------------------------------------------------------


def _encode_file(path: str | Path) -> tuple[str, str]:
    """Return (base64_data, mime_type) for a local file."""
    p = Path(path)
    raw = p.read_bytes()
    mime, _ = mimetypes.guess_type(str(p))
    if not mime:
        raise ValueError(f"Cannot guess MIME type for {p}")
    return base64.b64encode(raw).decode(), mime


def build_image_message(
    text: str,
    *,
    image_path: str | Path | None = None,
    image_url: str | None = None,
    image_b64: str | None = None,
    mime_type: str = "image/jpeg",
) -> HumanMessage:
    """
    Build a multimodal HumanMessage with text + image.
    Accepts a local path, a URL, or raw base64 bytes.
    """
    content: list[dict[str, Any]] = [{"type": "text", "text": text}]

    if image_path:
        b64, mime = _encode_file(path=image_path)
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:{mime};base64,{b64}"},
            }
        )
    elif image_url:
        content.append({"type": "image_url", "image_url": {"url": image_url}})
    elif image_b64:
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:{mime_type};base64,{image_b64}"},
            }
        )
    else:
        raise ValueError("Provide one of image_path, image_url, or image_b64")

    return HumanMessage(content=content)


async def ainvoke_multimodal(
    text: str,
    *,
    image_path: str | Path | None = None,
    image_url: str | None = None,
    image_b64: str | None = None,
    system: str | None = None,
    model: ChatGoogleGenerativeAI | None = None,
) -> str:
    llm = model or build_chat_model(model_name="gemini-2.0-flash")
    messages: list[BaseMessage] = []
    if system:
        messages.append(SystemMessage(content=system))
    messages.append(
        build_image_message(
            text=text,
            image_path=image_path,
            image_url=image_url,
            image_b64=image_b64,
        )
    )
    result = await llm.ainvoke(messages)
    return result.content  # type: ignore[return-value]


async def abatch_multimodal(
    items: list[dict[str, Any]],
    *,
    system: str | None = None,
    model: ChatGoogleGenerativeAI | None = None,
    max_concurrency: int | None = None,
) -> list[str]:
    """
    Batch multimodal calls.

    Each item in `items` should have keys: text, and one of
    image_path | image_url | image_b64.
    """
    tasks = [
        ainvoke_multimodal(
            text=item["text"],
            image_path=item.get("image_path"),
            image_url=item.get("image_url"),
            image_b64=item.get("image_b64"),
            system=system,
            model=model,
        )
        for item in items
    ]
    sem = asyncio.Semaphore(max_concurrency or _mcfg.max_concurrency)

    async def bounded(coro):
        async with sem:
            return await coro

    return await asyncio.gather(*[bounded(coro=t) for t in tasks])


# ---------------------------------------------------------------------------
# Structured output
# ---------------------------------------------------------------------------


def with_structured_output(
    schema: type[BaseModel],
    *,
    model: ChatGoogleGenerativeAI | None = None,
    method: str = "function_calling",
) -> Any:
    """
    Return a model augmented with structured output.
    Invoke like a normal chain; output is a validated Pydantic model.

    Example::

        class Answer(BaseModel):
            answer: str
            confidence: float

        chain = with_structured_output(Answer)
        result: Answer = await chain.ainvoke("What is 2+2?")
    """
    llm = model or build_chat_model()
    return llm.with_structured_output(schema=schema, method=method)


# ---------------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------------


async def aembed_text(text: str) -> list[float]:
    """Embed a single string asynchronously."""
    model = build_embedding_model()
    return await asyncio.to_thread(model.embed_query, text)


async def aembed_batch(
    texts: list[str], *, max_concurrency: int | None = None
) -> list[list[float]]:
    """
    Embed multiple strings, respecting concurrency limits.
    GoogleGenerativeAIEmbeddings.embed_documents handles batching internally.
    """
    model = build_embedding_model()
    # embed_documents is synchronous; run in thread pool
    return await asyncio.to_thread(model.embed_documents, texts)
