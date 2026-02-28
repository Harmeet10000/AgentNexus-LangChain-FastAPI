"""
FastAPI router integration layer.

This file shows HOW to wire the agent system into FastAPI endpoints.
You already have FastAPI set up — just mount this router into your app.

Mount in your main.py::

    from agents.api import router as agent_router
    app.include_router(agent_router, prefix="/agents", tags=["agents"])
"""

from __future__ import annotations

import json
import logging
from typing import AsyncIterator

from agents.registry import (
    get_code_agent,
    get_general_agent,
    get_multi_agent_system,
    get_research_agent,
    get_router,
)
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
from langchain_layer.models import aembed_batch, aembed_text, ainvoke_multimodal
from middleware.guardrails import get_guardrails
from schemas.agent import (
    AgentBatchRequest,
    AgentBatchResponse,
    AgentInvokeRequest,
    AgentResponse,
    EmbedRequest,
    EmbedResponse,
    MemorySearchRequest,
    MemorySearchResponse,
    MultimodalAgentRequest,
    StreamChunk,
)

logger = logging.getLogger(__name__)
router = APIRouter()


# ---------------------------------------------------------------------------
# Helper: extract text response from agent state
# ---------------------------------------------------------------------------


def _extract_response(state: dict, thread_id: str) -> AgentResponse:
    messages = state.get("messages", [])
    content = ""
    for msg in reversed(messages):
        c = getattr(msg, "content", "")
        if c and isinstance(c, str):
            content = c
            break

    tool_calls = sum(
        1 for m in messages
        if hasattr(m, "tool_calls") and m.tool_calls
    )

    return AgentResponse(
        content=content,
        thread_id=thread_id,
        structured_output=state.get("structured_output"),
        blocked=state.get("blocked", False),
        block_reason=state.get("block_reason"),
        tool_calls_made=tool_calls,
    )


# ---------------------------------------------------------------------------
# Single-turn invoke
# ---------------------------------------------------------------------------


@router.post("/invoke/{agent_name}", response_model=AgentResponse)
async def invoke_agent(agent_name: str, request: AgentInvokeRequest):
    """Invoke a named agent and return the full response."""
    agents = {
        "general": get_general_agent,
        "research": get_research_agent,
        "code": get_code_agent,
    }
    if agent_name not in agents:
        raise HTTPException(status_code=404, detail=f"Agent '{agent_name}' not found")

    # Deterministic guardrail on input
    guardrails = get_guardrails()
    safe, reason = guardrails.check_input(request.message)
    if not safe:
        raise HTTPException(status_code=400, detail=f"Input blocked: {reason}")

    agent = agents[agent_name]()
    try:
        state = await agent.ainvoke(
            request.message,
            thread_id=request.thread_id,
            user_id=request.user_id,
        )
        return _extract_response(state, request.thread_id)
    except Exception as exc:
        logger.exception("Agent invocation failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


# ---------------------------------------------------------------------------
# Streaming invoke
# ---------------------------------------------------------------------------


@router.post("/stream/{agent_name}")
async def stream_agent(agent_name: str, request: AgentInvokeRequest):
    """Stream agent responses via Server-Sent Events."""
    agents = {"general": get_general_agent, "research": get_research_agent, "code": get_code_agent}
    if agent_name not in agents:
        raise HTTPException(status_code=404, detail=f"Agent '{agent_name}' not found")

    agent = agents[agent_name]()

    async def event_generator() -> AsyncIterator[str]:
        try:
            async for chunk in agent.astream(
                request.message,
                thread_id=request.thread_id,
                stream_mode="messages",
            ):
                # chunk is (message, metadata) tuple in "messages" mode
                if isinstance(chunk, tuple):
                    msg, meta = chunk
                    content = getattr(msg, "content", "")
                    if content:
                        sc = StreamChunk(type="token", content=content)
                        yield f"data: {sc.model_dump_json()}\n\n"
            yield f"data: {StreamChunk(type='done').model_dump_json()}\n\n"
        except Exception as exc:
            sc = StreamChunk(type="error", content=str(exc))
            yield f"data: {sc.model_dump_json()}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


# ---------------------------------------------------------------------------
# Batch invoke
# ---------------------------------------------------------------------------


@router.post("/batch/{agent_name}", response_model=AgentBatchResponse)
async def batch_invoke(agent_name: str, request: AgentBatchRequest):
    """Invoke an agent on multiple messages concurrently."""
    agents = {"general": get_general_agent, "research": get_research_agent}
    if agent_name not in agents:
        raise HTTPException(status_code=404, detail=f"Agent '{agent_name}' not found")

    if len(request.messages) != len(request.thread_ids):
        raise HTTPException(status_code=422, detail="messages and thread_ids must have the same length")

    agent = agents[agent_name]()
    states = await agent.abatch(
        request.messages,
        thread_ids=request.thread_ids,
        user_id=request.user_id,
        max_concurrency=request.max_concurrency,
    )

    responses = [_extract_response(s, tid) for s, tid in zip(states, request.thread_ids)]
    failed = sum(1 for r in responses if r.blocked)
    return AgentBatchResponse(responses=responses, total=len(responses), failed=failed)


# ---------------------------------------------------------------------------
# Multimodal
# ---------------------------------------------------------------------------


@router.post("/multimodal", response_model=AgentResponse)
async def multimodal_invoke(request: MultimodalAgentRequest):
    """Invoke the agent with text + image input."""
    if not request.image_url and not request.image_b64:
        raise HTTPException(status_code=422, detail="Provide image_url or image_b64")

    try:
        content = await ainvoke_multimodal(
            request.message,
            image_url=request.image_url,
            image_b64=request.image_b64,
        )
        return AgentResponse(content=content, thread_id=request.thread_id)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


# ---------------------------------------------------------------------------
# Smart router endpoint
# ---------------------------------------------------------------------------


@router.post("/route", response_model=AgentResponse)
async def route_and_invoke(request: AgentInvokeRequest):
    """
    Automatically route the request to the best agent.
    The LLM decides which agent handles it.
    """
    r = get_router()
    try:
        state = await r.route(
            request.message,
            thread_id=request.thread_id,
            user_id=request.user_id,
        )
        return _extract_response(state, request.thread_id)
    except Exception as exc:
        logger.exception("Router failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


# ---------------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------------


@router.post("/embed", response_model=EmbedResponse)
async def embed_texts(request: EmbedRequest):
    """Generate embeddings for one or more texts."""
    from config.settings import get_settings
    embeddings = await aembed_batch(request.texts)
    return EmbedResponse(
        embeddings=embeddings,
        model=get_settings().model.gemini_embedding_model,
        dimensions=len(embeddings[0]) if embeddings else 0,
    )


# ---------------------------------------------------------------------------
# Human-in-the-loop approval
# ---------------------------------------------------------------------------


@router.post("/approve/{agent_name}/{thread_id}")
async def approve_action(agent_name: str, thread_id: str):
    """
    Resume a paused agent after human approval.
    Call this after reviewing the pending action at GET /state/{agent_name}/{thread_id}.
    """
    agents = {"code": get_code_agent}
    if agent_name not in agents:
        raise HTTPException(status_code=404, detail=f"Agent '{agent_name}' not found")

    agent = agents[agent_name]()
    try:
        state = await agent.resume_after_approval(thread_id)
        return _extract_response(state, thread_id)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/state/{agent_name}/{thread_id}")
async def get_agent_state(agent_name: str, thread_id: str):
    """Get the current state of an agent thread (for HITL review)."""
    agents = {"code": get_code_agent, "general": get_general_agent, "research": get_research_agent}
    if agent_name not in agents:
        raise HTTPException(status_code=404, detail=f"Agent '{agent_name}' not found")

    agent = agents[agent_name]()
    state = agent.get_state(thread_id)
    if not state:
        raise HTTPException(status_code=404, detail="Thread not found")

    return {
        "thread_id": thread_id,
        "pending_approval": state.values.get("pending_approval"),
        "blocked": state.values.get("blocked", False),
        "todo_list": state.values.get("todo_list", []),
        "message_count": len(state.values.get("messages", [])),
    }
