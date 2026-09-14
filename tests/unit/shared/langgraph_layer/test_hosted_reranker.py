"""Hosted reranker behind the protocol (agentic-retrieval tasks 2.2, 2.3).

The stub provider returns twenty candidates' ordering; the node-facing
contract is twenty in, `limit` out, in the provider's returned order. A
provider failure degrades to the fused order truncated to `limit` — no
exception escapes.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from app.shared.langgraph_layer.retrieval_kb.reranker import HostedReranker

if TYPE_CHECKING:
    from typing import Any
from app.shared.langgraph_layer.retrieval_kb.state import RetrievedChunk


def _chunk(chunk_id: str) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        chunk_text=f"text {chunk_id}",
        preamble="pre",
        clause_type="other",
        parent_doc_id="doc-1",
        metadata_={},
        custom_metadata={},
        score=0.0,
    )


class _StubResponse:
    def __init__(self, payload: dict[str, Any]) -> None:
        self._payload = payload

    def raise_for_status(self) -> None:
        pass

    def json(self) -> dict[str, Any]:
        return self._payload


class _StubClient:
    """Provider double: reverses twenty candidates regardless of content."""

    def __init__(self, n: int = 20) -> None:
        self.requests: list[dict[str, Any]] = []
        self._n = n

    async def post(self, url: str, **kwargs: Any) -> _StubResponse:
        self.requests.append({"url": url, **kwargs})
        order = list(reversed(range(self._n)))
        return _StubResponse(
            {"results": [{"index": i, "relevance_score": 1.0 - i * 0.01} for i in order]}
        )

    async def aclose(self) -> None:
        pass


_PROVIDER_DOWN = "provider unreachable"


class _FailingClient:
    async def post(self, url: str, **kwargs: Any) -> Any:
        raise ConnectionError(_PROVIDER_DOWN)

    async def aclose(self) -> None:
        pass


async def test_hosted_reranker_returns_limit_in_provider_order() -> None:
    chunks = [_chunk(f"c-{i}") for i in range(20)]
    reranker = HostedReranker(
        api_key="key",
        model="rerank-test",
        endpoint="https://rerank.example/v2/rerank",
        client=_StubClient(),  # type: ignore[arg-type]
    )

    reranked = await reranker.rerank("query", chunks, limit=5)

    assert [chunk.chunk_id for chunk in reranked] == [f"c-{i}" for i in (19, 18, 17, 16, 15)]
    assert len(reranked) == 5


async def test_hosted_reranker_sends_documents_and_top_n() -> None:
    chunks = [_chunk(f"c-{i}") for i in range(3)]
    client = _StubClient(n=3)
    reranker = HostedReranker(
        api_key="key",
        model="rerank-test",
        endpoint="https://rerank.example/v2/rerank",
        client=client,  # type: ignore[arg-type]
    )

    await reranker.rerank("query", chunks, limit=3)

    assert len(client.requests) == 1
    body = client.requests[0]["json"]
    assert body["top_n"] == 3
    assert len(body["documents"]) == 3


async def test_hosted_reranker_failure_degrades_to_fused_order() -> None:
    chunks = [_chunk(f"c-{i}") for i in range(20)]
    reranker = HostedReranker(
        api_key="key",
        model="rerank-test",
        endpoint="https://rerank.example/v2/rerank",
        client=_FailingClient(),  # type: ignore[arg-type]
    )

    reranked = await reranker.rerank("query", chunks, limit=5)

    assert [chunk.chunk_id for chunk in reranked] == [f"c-{i}" for i in range(5)]


async def test_hosted_reranker_without_key_degrades_without_calling() -> None:
    chunks = [_chunk(f"c-{i}") for i in range(4)]
    client = _StubClient(n=4)
    reranker = HostedReranker(
        api_key="",
        model="rerank-test",
        endpoint="https://rerank.example/v2/rerank",
        client=client,  # type: ignore[arg-type]
    )

    reranked = await reranker.rerank("query", chunks, limit=2)

    assert [chunk.chunk_id for chunk in reranked] == ["c-0", "c-1"]
    assert client.requests == []
