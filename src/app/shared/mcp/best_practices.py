"""Reusable MCP client best-practice helpers.

These helpers are intentionally not imported by the deep research graph. They
preserve useful MCP patterns at the MCP boundary: OAuth token exchange, durable
token storage through LangGraph's store, and user-friendly MCP tool errors.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

import httpx
from langchain_core.tools import ToolException
from langgraph.config import get_store
from mcp import McpError

from app.utils import logger

if TYPE_CHECKING:
    from typing import Any

    from langchain_core.runnables import RunnableConfig
    from langchain_core.tools import StructuredTool


async def exchange_subject_token_for_mcp_token(
    subject_token: str,
    base_mcp_url: str,
    *,
    http_client: httpx.AsyncClient | None = None,
) -> dict[str, Any] | None:
    """Exchange an application access token for an MCP access token."""
    close_client = http_client is None
    client = http_client or httpx.AsyncClient(timeout=httpx.Timeout(15.0))
    token_url = f"{base_mcp_url.rstrip('/')}/oauth/token"
    form_data = {
        "client_id": "mcp_default",
        "subject_token": subject_token,
        "grant_type": "urn:ietf:params:oauth:grant-type:token-exchange",
        "resource": f"{base_mcp_url.rstrip('/')}/mcp",
        "subject_token_type": "urn:ietf:params:oauth:token-type:access_token",
    }
    try:
        response = await client.post(
            token_url,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            data=form_data,
        )
        response.raise_for_status()
        data = response.json()
        return data if isinstance(data, dict) else None
    except httpx.HTTPError as exc:
        logger.bind(error=str(exc), mcp_url=base_mcp_url).warning("mcp_token_exchange_failed")
        return None
    finally:
        if close_client:
            await client.aclose()


async def get_stored_mcp_tokens(config: RunnableConfig) -> dict[str, Any] | None:
    """Retrieve non-expired MCP tokens from the LangGraph store."""
    store = get_store()
    thread_id = config.get("configurable", {}).get("thread_id")
    user_id = config.get("metadata", {}).get("owner")
    if not thread_id or not user_id:
        return None

    tokens = await store.aget((user_id, "tokens"), "mcp")
    if not tokens:
        return None

    expires_in = tokens.value.get("expires_in")
    if not isinstance(expires_in, int):
        return tokens.value

    expires_at = tokens.created_at + timedelta(seconds=expires_in)
    if datetime.now(tz=UTC) <= expires_at:
        return tokens.value

    await store.adelete((user_id, "tokens"), "mcp")
    return None


async def set_stored_mcp_tokens(config: RunnableConfig, tokens: dict[str, Any]) -> None:
    """Store MCP tokens in the LangGraph store."""
    store = get_store()
    thread_id = config.get("configurable", {}).get("thread_id")
    user_id = config.get("metadata", {}).get("owner")
    if not thread_id or not user_id:
        return
    await store.aput((user_id, "tokens"), "mcp", tokens)


def wrap_mcp_interaction_errors(tool: StructuredTool) -> StructuredTool:
    """Convert MCP interaction-required errors into model-visible tool errors."""
    original_coroutine = tool.coroutine
    if original_coroutine is None:
        return tool

    async def interaction_error_wrapper(**kwargs: Any) -> Any:
        try:
            return await original_coroutine(**kwargs)
        except BaseException as original_error:
            mcp_error = _find_mcp_error(original_error)
            if mcp_error is None:
                raise
            error_details = mcp_error.error
            error_code = getattr(error_details, "code", None)
            error_data = getattr(error_details, "data", None) or {}
            if error_code != -32003:
                raise

            message_payload = error_data.get("message", {})
            error_message = "Required interaction"
            if isinstance(message_payload, dict):
                error_message = message_payload.get("text") or error_message
            if url := error_data.get("url"):
                error_message = f"{error_message} {url}"
            raise ToolException(error_message) from original_error

    tool.coroutine = interaction_error_wrapper
    return tool


def _find_mcp_error(exc: BaseException) -> McpError | None:
    """Search nested exception groups for an MCP error."""
    if isinstance(exc, McpError):
        return exc
    nested = getattr(exc, "exceptions", None)
    if not nested:
        return None
    for child in nested:
        found = _find_mcp_error(child)
        if found is not None:
            return found
    return None
