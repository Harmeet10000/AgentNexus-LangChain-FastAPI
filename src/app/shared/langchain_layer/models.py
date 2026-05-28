"""
Model helpers for Google Gemini.

Public helpers in this module are async-only.
"""
## Architecture

# ```
# ┌─────────────────────────────────────────────────────────────┐
# │                        FastAPI Layer                        │
# │   /agents/invoke  /agents/stream  /agents/batch  /embed     │
# └───────────────────────────┬─────────────────────────────────┘
#                             │
# ┌───────────────────────────▼────────────────────────────────┐
# │                      Agent Runtime                         │
# │   create_production_agent(AgentSpec)  →  ProductionAgent   │
# │                                                            │
# │   ┌─────────────────────────────────────────────────────┐  │
# │   │              LangChain 1.0: create_agent            │  │
# │   │                                                     │  │
# │   │  model ──► [middleware stack] ──► tools ──► output  │  │
# │   │             │                                       │  │
# │   │  Middleware (before_model order):                   │  │
# │   │  1. SummarizationMiddleware  (context window)       │  │
# │   │  2. LLMToolSelectorMiddleware (reduce tool noise)   │  │
# │   │  3. ToolRetryMiddleware      (resilience)           │  │
# │   │  4. ModelRetryMiddleware     (resilience)           │  │
# │   │  5. HumanInTheLoopMiddleware (HITL - optional)      │  │
# │   │  6. GuardrailMiddleware      (safety - after_model) │  │
# │   └─────────────────────────────────────────────────────┘  │
# │                                                            │
# │   context_schema → RichContext (user_id, role, flags...)   │
# │   checkpointer   → InMemory / Postgres / Redis             │
# └───────────────────────────┬────────────────────────────────┘
#                             │
#           ┌─────────────────┴─────────────────┐
#           │                                   │
# ┌─────────▼──────────┐             ┌──────────▼──────────┐
# │  Short-term Memory │             │  Long-term Memory   │
# │  LangGraph         │             │  cognee             │
# │  Checkpointer      │             │  (semantic search)  │
# │  (per-thread)      │             │  (cross-session)    │
# └────────────────────┘             └─────────────────────┘
# ```

from __future__ import annotations

import asyncio
import base64
import mimetypes
from pathlib import Path
from typing import TYPE_CHECKING, cast

import toons
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    GoogleGenerativeAIEmbeddings,
    create_context_cache,
)
from pydantic import BaseModel

from app.config import get_settings
from app.connections.httpx_client import get_shared_httpx_client

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Callable
    from typing import Any, Literal

    from langchain_core.embeddings import Embeddings
    from langchain_core.language_models import BaseChatModel
    from langchain_core.messages import BaseMessage
    from langchain_core.tools import BaseTool

    from app.config import Settings

settings: Settings = get_settings()


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------
# TODO: Add langchain specific middlewares here LLMToolSelectMiddleware, GuardrailMiddleware, etc.


def _build_chat_model(
    model_name: str | None = None,
    *,
    temperature: float | None = None,
    top_p: float | None = None,
    top_k: int | None = None,
    max_tokens: int | None = None,
    streaming: bool = False,
    cached_content: str | None = None,
    implementation: Literal["generic", "google_genai"] = "generic",
    **kwargs: Any,
) -> BaseChatModel:
    """Return a configured Gemini chat model instance."""

    resolved_model_name: str = model_name or settings.GEMINI_PRO_MODEL
    resolved_temperature: int | float = temperature if temperature is not None else settings.GEMINI_TEMPERATURE
    resolved_top_p: int | float = top_p if top_p is not None else settings.GEMINI_TOP_P
    resolved_top_k: int = top_k if top_k is not None else settings.GEMINI_TOP_K
    resolved_max_tokens: int = max_tokens or settings.GEMINI_MAX_TOKENS

    if implementation == "google_genai":
        return ChatGoogleGenerativeAI(
            model=resolved_model_name,
            thinking_level="medium",
            temperature=resolved_temperature,
            top_p=resolved_top_p,
            top_k=resolved_top_k,
            max_tokens=resolved_max_tokens,
            api_key=settings.GEMINI_API_KEY,
            streaming=streaming,
            timeout=settings.TAVILY_TIMEOUT_SECONDS,
            cached_content=cached_content,
            **kwargs,
        )

    return init_chat_model(
        model=resolved_model_name,
        thinking_level="medium",
        temperature=resolved_temperature,
        top_p=resolved_top_p,
        top_k=resolved_top_k,
        max_output_tokens=resolved_max_tokens,
        api_key=settings.GEMINI_API_KEY,
        streaming=streaming,
        cached_content=cached_content,
        **kwargs,
        http_async_client=get_shared_httpx_client(),
    )


async def acreate_gemini_context_cache(
    messages: list[BaseMessage],
    *,
    model: BaseChatModel | None = None,
    ttl: str | None = None,
    tools: list[BaseTool | type[BaseModel] | dict[str, Any] | Callable[..., Any]] | None = None,
    tool_choice: str | bool | None = None,
) -> str:
    """
    Create an explicit Gemini context cache for large reusable prompts.

    Use this for shared context that would otherwise be re-sent on every request,
    such as long system instructions, logs, code snapshots, or uploaded file refs.
    """
    llm = cast(
        "ChatGoogleGenerativeAI",
        model if isinstance(model, ChatGoogleGenerativeAI) else _build_chat_model(
            implementation="google_genai"
        ),
    )
    resolved_ttl = ttl or settings.GEMINI_CONTEXT_CACHE_TTL
    return await asyncio.to_thread(
        create_context_cache,
        model=llm,
        messages=messages,
        ttl=resolved_ttl,
        tools=tools,
        tool_choice=tool_choice,
    )


def _build_embedding_model_gemini_full(
    model_name: str | None = None,
    *,
    task_type: str = "RETRIEVAL_DOCUMENT",
    title: str | None = None,
    client: Any | None = None,
    **kwargs: Any,
) -> GoogleGenerativeAIEmbeddings:
    """
    Build a Gemini embedding model with full configuration.

    Args:
        model_name: Embedding model (e.g., "embedding-001").
                   Defaults to configured model.
        task_type: Task for embedding optimization:
                  - "RETRIEVAL_DOCUMENT": Retrieving documents (default).
                  - "RETRIEVAL_QUERY": Querying documents.
                  - "SEMANTIC_SIMILARITY": Comparing phrases for similarity.
                  - "CLASSIFICATION": Embedding text for classification.
                  - "CLUSTERING": Embedding text for clustering.
        title: Optional content title (used with task_type).
        client: Custom Google API client (advanced use cases).
        **kwargs: Additional config (user_agent, request_options, etc.).

    Example::

        # For document indexing (high recall)
        doc_embedder = build_embedding_model_gemini_full(
            task_type="RETRIEVAL_DOCUMENT",
            title="Knowledge base documents",
        )

        # For query embedding (paired with above)
        query_embedder = build_embedding_model_gemini_full(
            task_type="RETRIEVAL_QUERY",
        )

        # For semantic similarity
        sim_embedder = build_embedding_model_gemini_full(
            task_type="SEMANTIC_SIMILARITY",
        )
    """
    _ = title
    return GoogleGenerativeAIEmbeddings(
        model=model_name or settings.GEMINI_EMBEDDING_MODEL,
        task_type=task_type,
        # title=title,
        client=client,
        **kwargs,
    )


def serialize_to_toon(payload: Any) -> str:
    """
    Serialize structured prompt context with TOON for lower token overhead than JSON.

    Use this when passing large tables, records, or nested objects into prompts.
    """
    return toons.dumps(payload)


# ---------------------------------------------------------------------------
# Text inference
# ---------------------------------------------------------------------------


async def ainvoke_text(
    prompt: str,
    *,
    system: str | None = None,
    model: BaseChatModel | None = None,
) -> str:
    """Single async text call, returns plain string."""
    llm = model or _build_chat_model()
    messages: list[BaseMessage] = []
    if system:
        messages.append(SystemMessage(content=system))
    messages.append(HumanMessage(content=prompt))
    result = await llm.ainvoke(messages)
    return str(result.content)


async def abatch_text(
    prompts: list[str],
    *,
    system: str | None = None,
    model: BaseChatModel | None = None,
    max_concurrency: int | None = None,
) -> list[str]:
    """
    Batch async text calls.
    Uses LangChain's native abatch which honours max_concurrency.
    """
    llm = model or _build_chat_model()
    max_c = max_concurrency or 5

    def _build(prompt: str) -> list[BaseMessage]:
        msgs: list[BaseMessage] = []
        if system:
            msgs.append(SystemMessage(content=system))
        msgs.append(HumanMessage(content=prompt))
        return msgs

    message_batches = [_build(p) for p in prompts]
    results = await llm.abatch(cast("Any", message_batches), config={"max_concurrency": max_c})
    return [str(r.content) for r in results]


async def astream_text(
    prompt: str,
    *,
    system: str | None = None,
    model: BaseChatModel | None = None,
) -> AsyncIterator[str]:
    """
    Async generator that yields text tokens as they arrive.

    Usage in FastAPI::

        async def sse_endpoint():
            async for chunk in astream_text("Hello"):
                yield f"data: {chunk}\\n\\n"
    """
    llm = model or _build_chat_model(streaming=True)
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
        msg = f"Cannot guess MIME type for {p}"
        raise ValueError(msg)
    return base64.b64encode(raw).decode(), mime


async def _build_image_message(
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
        b64, mime = await asyncio.to_thread(_encode_file, image_path)
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
        msg = "Provide one of image_path, image_url, or image_b64"
        raise ValueError(msg)

    return HumanMessage(content=cast("Any", content))


async def ainvoke_multimodal(
    text: str,
    *,
    image_path: str | Path | None = None,
    image_url: str | None = None,
    image_b64: str | None = None,
    system: str | None = None,
    model: BaseChatModel | None = None,
) -> str:
    llm = model or _build_chat_model(
        model_name=settings.GEMINI_VISION_MODEL,
        media_resolution="low",
    )
    messages: list[BaseMessage] = []
    if system:
        messages.append(SystemMessage(content=system))
    messages.append(
        await _build_image_message(
            text=text,
            image_path=image_path,
            image_url=image_url,
            image_b64=image_b64,
        )
    )
    result = await llm.ainvoke(messages)
    return str(result.content)


async def abatch_multimodal(
    items: list[dict[str, Any]],
    *,
    system: str | None = None,
    model: BaseChatModel | None = None,
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
    sem = asyncio.Semaphore(max_concurrency or 5)

    async def bounded(coro: Any) -> str:
        async with sem:
            return await coro

    return await asyncio.gather(*[bounded(coro=t) for t in tasks])


# ---------------------------------------------------------------------------
# Structured output
# ---------------------------------------------------------------------------


async def aget_chat_model(
    *,
    model_name: str | None = None,
    temperature: float | None = None,
    top_p: float | None = None,
    top_k: int | None = None,
    max_tokens: int | None = None,
    streaming: bool = False,
    cached_content: str | None = None,
    implementation: Literal["generic", "google_genai"] = "generic",
    **kwargs: Any,
) -> BaseChatModel:
    """Return a configured chat model, defaulting all behavior from settings."""
    return _build_chat_model(
        model_name=model_name,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        max_tokens=max_tokens,
        streaming=streaming,
        cached_content=cached_content,
        implementation=implementation,
        **kwargs,
    )


async def awith_structured_output(
    schema: type[BaseModel],
    *,
    model: BaseChatModel | None = None,
    method: str = "function_calling",
) -> Any:
    """
    Return a model augmented with structured output.
    Invoke like a normal chain; output is a validated Pydantic model.

    Example::

        class Answer(BaseModel):
            answer: str
            confidence: float

        chain = await awith_structured_output(Answer)
        result: Answer = await chain.ainvoke("What is 2+2?")
    """
    llm = model or _build_chat_model()
    return llm.with_structured_output(schema=schema, method=method)


# ---------------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------------


async def aembed_text(
    text: str,
    *,
    model: Embeddings | None = None,
) -> list[float]:
    """
    Embed a single string asynchronously.

    Args:
        text: The text to embed.
        model: Embedding model. Defaults to Gemini embeddings.
    """
    emb = model or _build_embedding_model_gemini_full()
    return await asyncio.to_thread(emb.embed_query, text)


async def aembed_batch(
    texts: list[str],
    *,
    model: Embeddings | None = None,
    max_concurrency: int | None = None,
) -> list[list[float]]:
    """
    Embed multiple strings, respecting concurrency limits.
    GoogleGenerativeAIEmbeddings.embed_documents handles batching internally.

    Args:
        texts: List of texts to embed.
        model: Embedding model. Defaults to Gemini embeddings.
        max_concurrency: Max concurrent requests (note: GoogleGenerativeAI
                        handles batch requests server-side).
    """
    _ = max_concurrency  # Note: Gemini batching is server-side; limit isn't used here
    emb = model or _build_embedding_model_gemini_full()
    # embed_documents is synchronous; run in thread pool
    return await asyncio.to_thread(emb.embed_documents, texts)
