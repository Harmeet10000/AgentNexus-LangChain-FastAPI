"""
Unified memory manager.

Short-term memory: LangGraph checkpointer (InMemory / Postgres / Redis).
  - Persists full message history across invocations for a thread_id.
  - Managed automatically by LangGraph.

Long-term memory: Mem0
  - Semantic memory extracted from conversations and stored across sessions.
  - Retrieved at agent startup and injected into the system prompt.
"""

from __future__ import annotations

import logging
from typing import Any

from config.settings import get_settings

logger = logging.getLogger(__name__)
_mem_cfg = get_settings().memory


# ---------------------------------------------------------------------------
# Short-term: checkpointer factory
# ---------------------------------------------------------------------------


def build_checkpointer(backend: str | None = None) -> Any:
    """
    Build a LangGraph checkpointer based on config.

    Returns:
        InMemorySaver | AsyncPostgresSaver | AsyncRedisSaver
    """
    backend = backend or _mem_cfg.checkpointer_backend

    if backend == "memory":
        from langgraph.checkpoint.memory import InMemorySaver
        return InMemorySaver()

    elif backend == "postgres":
        if not _mem_cfg.postgres_uri:
            raise ValueError("POSTGRES_URI is required for postgres checkpointer")
        # Requires: pip install langgraph-checkpoint-postgres
        from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
        return AsyncPostgresSaver.from_conn_string(_mem_cfg.postgres_uri)

    elif backend == "redis":
        if not _mem_cfg.redis_url:
            raise ValueError("REDIS_URL is required for redis checkpointer")
        # Requires: pip install langgraph-checkpoint-redis
        from langgraph.checkpoint.redis.aio import AsyncRedisSaver
        return AsyncRedisSaver.from_url(_mem_cfg.redis_url)

    else:
        raise ValueError(f"Unknown checkpointer backend: {backend}")


# ---------------------------------------------------------------------------
# Long-term: Mem0 integration
# ---------------------------------------------------------------------------


class LongTermMemory:
    """
    Wrapper around Mem0 for semantic long-term memory.

    Mem0 automatically extracts and stores facts from conversations.
    On new sessions, relevant memories are retrieved and injected into
    the agent's system prompt as context.
    """

    def __init__(self) -> None:
        self._client = self._build_client()

    def _build_client(self) -> Any:
        cfg = get_settings().memory
        if not cfg.mem0_api_key:
            logger.warning("MEM0_API_KEY not set; using in-process Mem0 (no persistence across restarts)")
            try:
                from mem0 import Memory
                return Memory()
            except ImportError:
                logger.error("mem0 not installed. Run: pip install mem0ai")
                return None
        else:
            try:
                from mem0 import MemoryClient
                return MemoryClient(api_key=cfg.mem0_api_key.get_secret_value())
            except ImportError:
                logger.error("mem0 not installed. Run: pip install mem0ai")
                return None

    async def add(
        self,
        messages: list[dict[str, str]],
        *,
        user_id: str,
        session_id: str | None = None,
        agent_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """
        Extract and store memories from a conversation.
        Call at the end of each session/turn.
        """
        if not self._client:
            return []
        try:
            import asyncio
            result = await asyncio.to_thread(
                self._client.add,
                messages,
                user_id=user_id,
                session_id=session_id,
                agent_id=agent_id,
                metadata=metadata or {},
            )
            return result.get("results", [])
        except Exception as exc:
            logger.error("Mem0 add failed: %s", exc)
            return []

    async def search(
        self,
        query: str,
        *,
        user_id: str,
        agent_id: str | None = None,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """
        Retrieve relevant memories for the current query.
        Returns list of memory dicts with 'memory' and 'score' keys.
        """
        if not self._client:
            return []
        try:
            import asyncio
            results = await asyncio.to_thread(
                self._client.search,
                query,
                user_id=user_id,
                agent_id=agent_id,
                limit=limit,
            )
            return results.get("results", [])
        except Exception as exc:
            logger.error("Mem0 search failed: %s", exc)
            return []

    async def delete_all(self, *, user_id: str) -> None:
        """Delete all memories for a user."""
        if not self._client:
            return
        try:
            import asyncio
            await asyncio.to_thread(self._client.delete_all, user_id=user_id)
        except Exception as exc:
            logger.error("Mem0 delete_all failed: %s", exc)

    async def format_for_prompt(
        self,
        query: str,
        *,
        user_id: str,
        agent_id: str | None = None,
        limit: int = 5,
    ) -> str:
        """
        Retrieve memories and format them as a system-prompt block.
        Inject the result into the agent's system prompt.
        """
        memories = await self.search(query, user_id=user_id, agent_id=agent_id, limit=limit)
        if not memories:
            return ""

        lines = ["## Long-term Memory (from previous sessions)"]
        for m in memories:
            mem_text = m.get("memory", "")
            score = m.get("score", 0.0)
            if mem_text:
                lines.append(f"- {mem_text} (relevance: {score:.2f})")

        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Unified manager
# ---------------------------------------------------------------------------


class MemoryManager:
    """
    Combines short-term (checkpointer) and long-term (Mem0) memory.
    Used by the agent runtime.
    """

    def __init__(self, *, backend: str | None = None) -> None:
        self.checkpointer = build_checkpointer(backend)
        self.long_term = LongTermMemory()

    async def inject_long_term_context(
        self,
        messages: list[Any],
        *,
        user_id: str,
        agent_id: str | None = None,
    ) -> list[Any]:
        """
        Prepend a memory block to the message list.
        Call before invoking the agent for a new turn.
        """
        from langchain_core.messages import HumanMessage, SystemMessage

        # Extract the user's latest query for memory search
        query = ""
        for msg in reversed(messages):
            if isinstance(msg, HumanMessage) and isinstance(msg.content, str):
                query = msg.content
                break

        if not query:
            return messages

        mem_block = await self.long_term.format_for_prompt(
            query, user_id=user_id, agent_id=agent_id
        )
        if not mem_block:
            return messages

        # Insert after the first SystemMessage (if any)
        result = list(messages)
        for i, msg in enumerate(result):
            if isinstance(msg, SystemMessage):
                current = msg.content if isinstance(msg.content, str) else ""
                result[i] = SystemMessage(content=f"{current}\n\n{mem_block}")
                return result

        # No system message found — prepend a new one
        result.insert(0, SystemMessage(content=mem_block))
        return result

    async def save_session(
        self,
        messages: list[Any],
        *,
        user_id: str,
        session_id: str | None = None,
        agent_id: str | None = None,
    ) -> None:
        """
        Extract long-term memories from the current session.
        Call at the end of each conversation.
        """
        from langchain_core.messages import AIMessage, HumanMessage

        # Convert to mem0 message format
        mem0_messages = []
        for msg in messages:
            if isinstance(msg, HumanMessage):
                content = msg.content if isinstance(msg.content, str) else str(msg.content)
                mem0_messages.append({"role": "user", "content": content})
            elif isinstance(msg, AIMessage):
                content = msg.content if isinstance(msg.content, str) else str(msg.content)
                if content:
                    mem0_messages.append({"role": "assistant", "content": content})

        if mem0_messages:
            await self.long_term.add(
                mem0_messages,
                user_id=user_id,
                session_id=session_id,
                agent_id=agent_id,
            )
