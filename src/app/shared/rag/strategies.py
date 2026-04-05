"""Reusable advanced RAG strategy helpers built around app-scoped dependencies."""

from __future__ import annotations

import asyncio
import importlib
import math
import re
from collections.abc import Awaitable, Callable, Sequence
from typing import TYPE_CHECKING, Any

import orjson
from pydantic import BaseModel, ConfigDict, PrivateAttr
from sqlalchemy import bindparam, text

from app.shared.langchain_layer.models import (
    ainvoke_text,
    build_chat_model,
    build_embedding_model,
    build_fast_model,
)
from app.utils import logger

if TYPE_CHECKING:
    from collections.abc import Mapping

    from fastapi import FastAPI
    from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker


VectorEmbedder = Callable[[str], list[float]]
AsyncTextResolver = Callable[[str], Awaitable[str]]
AsyncGraphSearcher = Callable[[str, int], Awaitable[Sequence[Any]]]


class RetrievedDocument(BaseModel):
    """Normalized retrieval result row."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: int | None
    document_id: str
    title: str
    content: str
    meta_data: dict[str, Any]
    similarity: float | None = None

    @classmethod
    def from_mapping(cls, row: Mapping[str, Any]) -> RetrievedDocument:
        return cls(
            id=row.get("id"),
            document_id=str(row.get("document_id", "")),
            title=str(row.get("title", "")),
            content=str(row.get("content", "")),
            meta_data=deserialize_metadata(row.get("meta_data")),
            similarity=_coerce_float(row.get("similarity")),
        )


class LateChunk(BaseModel):
    """Chunk content paired with a late-chunked embedding."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    content: str
    embedding: list[float]


class QueryExpansionResult(BaseModel):
    """Original query plus parsed alternates."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    original: str
    variants: list[str]


class AgenticRAGResult(BaseModel):
    """Selected retrieval tool and generated answer."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    selected_tool: str
    context: str
    answer: str


def split_text(text: str, *, size: int = 500) -> list[str]:
    """Split text into fixed-size chunks while dropping empty fragments."""
    return [text[i : i + size].strip() for i in range(0, len(text), size) if text[i : i + size].strip()]


def deduplicate_strings(values: list[str]) -> list[str]:
    """Preserve order while removing duplicate strings."""
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        ordered.append(value)
    return ordered


def serialize_metadata(metadata: dict[str, Any] | None) -> str:
    """Serialize metadata with orjson for PostgreSQL JSONB storage."""
    return orjson.dumps(metadata or {}).decode("utf-8")


def deserialize_metadata(value: Any) -> dict[str, Any]:
    """Normalize JSONB-style metadata values into dictionaries."""
    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    if isinstance(value, bytes):
        return orjson.loads(value)
    if isinstance(value, str):
        try:
            decoded = orjson.loads(value)
        except orjson.JSONDecodeError:
            return {}
        return decoded if isinstance(decoded, dict) else {}
    return {}


def cosine_similarity(left: list[float], right: list[float]) -> float:
    """Calculate cosine similarity between two vectors."""
    dot = sum(a * b for a, b in zip(left, right, strict=False))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return dot / (left_norm * right_norm)


def semantic_chunk_text(
    text: str,
    *,
    embedder: VectorEmbedder,
    similarity_threshold: float = 0.8,
) -> list[str]:
    """Chunk text by sentence-to-sentence semantic similarity."""
    sentences = [part.strip() for part in re.split(r"\.\s+", text.strip()) if part.strip()]
    if not sentences:
        return []

    sentence_embeddings = [embedder(sentence) for sentence in sentences]
    chunks: list[str] = []
    current_chunk = [sentences[0]]

    for index in range(len(sentences) - 1):
        similarity = cosine_similarity(sentence_embeddings[index], sentence_embeddings[index + 1])
        if similarity >= similarity_threshold:
            current_chunk.append(sentences[index + 1])
            continue
        chunks.append(". ".join(current_chunk))
        current_chunk = [sentences[index + 1]]

    chunks.append(". ".join(current_chunk))
    return chunks


def mean_pool_embeddings(vectors: list[list[float]]) -> list[float]:
    """Average multiple vectors into one pooled representation."""
    if not vectors:
        return []
    dimensions = len(vectors[0])
    return [sum(vector[index] for vector in vectors) / len(vectors) for index in range(dimensions)]


def late_chunk_text(
    text: str,
    *,
    embedder: VectorEmbedder,
    chunk_size: int = 512,
) -> list[LateChunk]:
    """Approximate late chunking by blending full-document and local chunk embeddings."""
    tokens = text.split()
    if not tokens:
        return []

    document_embedding = embedder(text)
    chunks: list[LateChunk] = []

    for start in range(0, len(tokens), chunk_size):
        chunk_content = " ".join(tokens[start : start + chunk_size]).strip()
        if not chunk_content:
            continue
        chunk_embedding = embedder(chunk_content)
        chunks.append(
            LateChunk(
                content=chunk_content,
                embedding=mean_pool_embeddings([document_embedding, chunk_embedding]),
            )
        )

    return chunks


def parse_query_variants(original: str, raw_response: str, *, limit: int = 3) -> list[str]:
    """Parse LLM-generated query variants, preserving the original query first."""
    cleaned: list[str] = [original]
    for line in raw_response.splitlines():
        candidate = re.sub(r"^\s*(?:[-*]|\d+[.)])\s*", "", line).strip()
        if not candidate or candidate.lower() == original.lower():
            continue
        cleaned.append(candidate)
        if len(cleaned) >= limit + 1:
            break
    return deduplicate_strings(cleaned)


def prepare_training_data() -> list[tuple[str, str]]:
    """Placeholder fine-tuning pairs kept from the original demo set."""
    return [
        ("What is EBITDA?", "financial_doc_about_ebitda.txt"),
        ("Explain capital expenditure", "capex_explanation.txt"),
    ]


def format_retrieved_documents(documents: Sequence[RetrievedDocument]) -> str:
    """Render retrieved documents into prompt context."""
    parts: list[str] = []
    for document in documents:
        heading = document.meta_data.get("heading")
        label = document.title if not heading else f"{document.title} | {heading}"
        if document.similarity is None:
            parts.append(f"[Source: {label}]\n{document.content}")
            continue
        parts.append(f"[Source: {label} | Similarity: {document.similarity:.3f}]\n{document.content}")
    return "\n\n".join(parts)


def format_graph_results(results: Sequence[Any]) -> str:
    """Render graph search results into plain text context."""
    context_parts: list[str] = []
    for item in results:
        if isinstance(item, dict):
            node_name = item.get("node_name") or item.get("entity") or "Unknown"
            node_type = item.get("node_type") or item.get("type") or "Unknown"
            summary = item.get("context") or item.get("summary") or ""
            relationships = item.get("relationships") or item.get("edges") or []
        else:
            node = getattr(item, "node", None)
            node_name = getattr(node, "name", "Unknown")
            node_type = getattr(node, "type", "Unknown")
            summary = getattr(item, "context", "")
            relationships = getattr(item, "relationships", [])

        context_parts.append(
            "\n".join(
                [
                    f"Entity: {node_name}",
                    f"Type: {node_type}",
                    f"Context: {summary}",
                    f"Relationships: {relationships}",
                ]
            )
        )

    return "\n---\n".join(context_parts)


def _coerce_float(value: Any) -> float | None:
    try:
        return None if value is None else float(value)
    except (TypeError, ValueError):
        return None


def _to_mapping(row: Any) -> dict[str, Any]:
    return dict(row)


def _serialize_embedding(vector: list[float]) -> str:
    return "[" + ",".join(str(value) for value in vector) + "]"


class RAGStrategyService(BaseModel):
    """Reusable strategy service backed by app-scoped DB resources."""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    engine: AsyncEngine
    session_local: async_sessionmaker[AsyncSession]
    reranker_model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    fine_tuned_model_path: str | None = None
    _embedding_model: Any = PrivateAttr(default=None)
    _chat_model: Any = PrivateAttr(default=None)
    _fast_model: Any = PrivateAttr(default=None)
    _reranker: Any = PrivateAttr(default=None)
    _fine_tuned_embedding_model: Any = PrivateAttr(default=None)

    @classmethod
    def from_app(
        cls,
        app: FastAPI,
        *,
        reranker_model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        fine_tuned_model_path: str | None = None,
    ) -> RAGStrategyService:
        """Create the service from FastAPI app state."""
        return cls(
            engine=app.state.db_engine,
            session_local=app.state.db_session_local,
            reranker_model_name=reranker_model_name,
            fine_tuned_model_path=fine_tuned_model_path,
        )

    @property
    def embedding_model(self) -> Any:
        if self._embedding_model is None:
            self._embedding_model = build_embedding_model()
        return self._embedding_model

    @property
    def chat_model(self) -> Any:
        if self._chat_model is None:
            self._chat_model = build_chat_model()
        return self._chat_model

    @property
    def fast_model(self) -> Any:
        if self._fast_model is None:
            self._fast_model = build_fast_model()
        return self._fast_model

    async def _get_reranker(self) -> Any:
        if self._reranker is None:
            sentence_transformers = importlib.import_module("sentence_transformers")
            cross_encoder = sentence_transformers.CrossEncoder

            logger.info("Loading RAG reranker", model_name=self.reranker_model_name)
            self._reranker = await asyncio.to_thread(cross_encoder, self.reranker_model_name)
        return self._reranker

    async def _get_fine_tuned_embedding_model(self) -> Any:
        if self._fine_tuned_embedding_model is None:
            if not self.fine_tuned_model_path:
                raise ValueError("fine_tuned_model_path is required for fine-tuned embeddings")
            sentence_transformers = importlib.import_module("sentence_transformers")
            sentence_transformer = sentence_transformers.SentenceTransformer

            logger.info("Loading fine-tuned embedding model", model_path=self.fine_tuned_model_path)
            self._fine_tuned_embedding_model = await asyncio.to_thread(
                sentence_transformer,
                self.fine_tuned_model_path,
            )
        return self._fine_tuned_embedding_model

    async def embed_query(self, text_value: str, *, use_fine_tuned: bool = False) -> list[float]:
        """Embed a query with either the default or fine-tuned model."""
        if use_fine_tuned:
            model = await self._get_fine_tuned_embedding_model()
            vector = await asyncio.to_thread(model.encode, text_value)
            return vector.tolist()
        return await asyncio.to_thread(self.embedding_model.embed_query, text_value)

    async def contextualize_chunk(self, document_text: str, chunk: str) -> str:
        """Generate a short contextual prefix for a chunk."""
        prompt = (
            "Provide a short contextual prefix for this chunk.\n\n"
            f"Document preview:\n{document_text[:700]}\n\nChunk:\n{chunk}"
        )
        prefix = await ainvoke_text(prompt, model=self.fast_model)
        return f"{prefix.strip()}\n{chunk}".strip()

    async def answer_with_context(
        self,
        question: str,
        *,
        context: str,
        instructions: str | None = None,
    ) -> str:
        """Generate an answer grounded in supplied retrieval context."""
        prompt = (
            (instructions or "Answer the question using only the provided context.")
            + f"\n\nQuestion: {question}\n\nContext:\n{context}"
        )
        return await ainvoke_text(prompt, model=self.chat_model)

    async def expand_query(self, query: str, *, variants: int = 3) -> QueryExpansionResult:
        """Generate alternate phrasings for a user query."""
        prompt = (
            f"Generate {variants} alternate phrasings of this query.\n"
            "Return one query per line without numbering.\n\n"
            f"Query: {query}"
        )
        raw = await ainvoke_text(prompt, model=self.fast_model)
        parsed = parse_query_variants(query, raw, limit=variants)
        return QueryExpansionResult(original=query, variants=parsed)

    async def grade_relevance(self, query: str, document: str) -> float:
        """Use the fast model to estimate query-document relevance."""
        prompt = (
            "Rate relevance from 0 to 1. Return only a number.\n\n"
            f"Query: {query}\n\nDocument:\n{document}"
        )
        raw = (await ainvoke_text(prompt, model=self.fast_model)).strip()
        try:
            return float(raw)
        except ValueError:
            return 0.0

    async def _insert_document_rows(self, rows: Sequence[dict[str, Any]]) -> list[RetrievedDocument]:
        insert_sql = text(
            """
            INSERT INTO document_vectors (user_id, document_id, title, content, embedding, meta_data)
            VALUES (
                :user_id,
                :document_id,
                :title,
                :content,
                CAST(:embedding AS vector),
                CAST(:meta_data AS jsonb)
            )
            RETURNING id, document_id, title, content, meta_data
            """
        )

        async with self.session_local() as session:
            inserted_rows: list[RetrievedDocument] = []
            for row in rows:
                result = await session.execute(insert_sql, row)
                inserted_rows.append(RetrievedDocument.from_mapping(_to_mapping(result.mappings().one())))
            await session.commit()
        return inserted_rows

    async def ingest_document(
        self,
        *,
        user_id: str,
        document_id: str,
        title: str,
        text_value: str,
        chunk_size: int = 500,
        meta_data: dict[str, Any] | None = None,
        use_fine_tuned_embeddings: bool = False,
    ) -> list[RetrievedDocument]:
        """Ingest a document with standard fixed-size chunking."""
        rows: list[dict[str, Any]] = []
        for index, chunk in enumerate(split_text(text_value, size=chunk_size)):
            embedding = await self.embed_query(chunk, use_fine_tuned=use_fine_tuned_embeddings)
            payload = {
                **(meta_data or {}),
                "strategy": "standard" if not use_fine_tuned_embeddings else "fine_tuned_embeddings",
                "chunk_index": index,
            }
            rows.append(
                {
                    "user_id": user_id,
                    "document_id": document_id,
                    "title": title,
                    "content": chunk,
                    "embedding": _serialize_embedding(embedding),
                    "meta_data": serialize_metadata(payload),
                }
            )
        return await self._insert_document_rows(rows)

    async def ingest_contextual_document(
        self,
        *,
        user_id: str,
        document_id: str,
        title: str,
        text_value: str,
        chunk_size: int = 500,
        meta_data: dict[str, Any] | None = None,
    ) -> list[RetrievedDocument]:
        """Ingest a document with contextual-retrieval enrichment."""
        rows: list[dict[str, Any]] = []
        for index, chunk in enumerate(split_text(text_value, size=chunk_size)):
            contextualized = await self.contextualize_chunk(text_value, chunk)
            embedding = await self.embed_query(contextualized)
            payload = {
                **(meta_data or {}),
                "strategy": "contextual_retrieval",
                "chunk_index": index,
            }
            rows.append(
                {
                    "user_id": user_id,
                    "document_id": document_id,
                    "title": title,
                    "content": contextualized,
                    "embedding": _serialize_embedding(embedding),
                    "meta_data": serialize_metadata(payload),
                }
            )
        return await self._insert_document_rows(rows)

    async def ingest_context_aware_document(
        self,
        *,
        user_id: str,
        document_id: str,
        title: str,
        text_value: str,
        similarity_threshold: float = 0.8,
        meta_data: dict[str, Any] | None = None,
    ) -> list[RetrievedDocument]:
        """Ingest a document using semantic chunking."""
        chunks = semantic_chunk_text(
            text_value,
            embedder=self.embedding_model.embed_query,
            similarity_threshold=similarity_threshold,
        )
        rows: list[dict[str, Any]] = []
        for index, chunk in enumerate(chunks):
            embedding = await self.embed_query(chunk)
            payload = {
                **(meta_data or {}),
                "strategy": "context_aware_chunking",
                "chunk_index": index,
            }
            rows.append(
                {
                    "user_id": user_id,
                    "document_id": document_id,
                    "title": title,
                    "content": chunk,
                    "embedding": _serialize_embedding(embedding),
                    "meta_data": serialize_metadata(payload),
                }
            )
        return await self._insert_document_rows(rows)

    async def ingest_late_chunked_document(
        self,
        *,
        user_id: str,
        document_id: str,
        title: str,
        text_value: str,
        chunk_size: int = 512,
        meta_data: dict[str, Any] | None = None,
    ) -> list[RetrievedDocument]:
        """Ingest a document with approximate late-chunked embeddings."""
        late_chunks = late_chunk_text(
            text_value,
            embedder=self.embedding_model.embed_query,
            chunk_size=chunk_size,
        )
        rows = [
            {
                "user_id": user_id,
                "document_id": document_id,
                "title": title,
                "content": chunk.content,
                "embedding": _serialize_embedding(chunk.embedding),
                "meta_data": serialize_metadata(
                    {
                        **(meta_data or {}),
                        "strategy": "late_chunking",
                        "chunk_index": index,
                    }
                ),
            }
            for index, chunk in enumerate(late_chunks)
        ]
        return await self._insert_document_rows(rows)

    async def ingest_hierarchical_document(
        self,
        *,
        user_id: str,
        document_id: str,
        title: str,
        text_value: str,
        parent_chunk_size: int = 2_000,
        child_chunk_size: int = 500,
        meta_data: dict[str, Any] | None = None,
    ) -> list[RetrievedDocument]:
        """Store parent and child chunks in one table using hierarchy metadata."""
        rows: list[dict[str, Any]] = []

        for parent_index, parent_chunk in enumerate(split_text(text_value, size=parent_chunk_size)):
            hierarchy_id = f"{document_id}:parent:{parent_index}"
            parent_meta = {
                **(meta_data or {}),
                "strategy": "hierarchical",
                "level": "parent",
                "hierarchy_id": hierarchy_id,
                "heading": f"{title} - Section {parent_index + 1}",
                "chunk_index": parent_index,
            }
            parent_embedding = await self.embed_query(parent_chunk)
            rows.append(
                {
                    "user_id": user_id,
                    "document_id": document_id,
                    "title": title,
                    "content": parent_chunk,
                    "embedding": _serialize_embedding(parent_embedding),
                    "meta_data": serialize_metadata(parent_meta),
                }
            )

            for child_index, child_chunk in enumerate(split_text(parent_chunk, size=child_chunk_size)):
                child_embedding = await self.embed_query(child_chunk)
                child_meta = {
                    **(meta_data or {}),
                    "strategy": "hierarchical",
                    "level": "child",
                    "hierarchy_id": hierarchy_id,
                    "parent_chunk_index": parent_index,
                    "chunk_index": child_index,
                    "heading": parent_meta["heading"],
                }
                rows.append(
                    {
                        "user_id": user_id,
                        "document_id": document_id,
                        "title": title,
                        "content": child_chunk,
                        "embedding": _serialize_embedding(child_embedding),
                        "meta_data": serialize_metadata(child_meta),
                    }
                )

        return await self._insert_document_rows(rows)

    async def vector_search(
        self,
        *,
        user_id: str,
        query: str,
        limit: int = 5,
        strategy: str | None = None,
        level: str | None = None,
        use_fine_tuned_embeddings: bool = False,
    ) -> list[RetrievedDocument]:
        """Search persisted chunks with pgvector similarity."""
        base_sql = """
            SELECT
                id,
                document_id,
                title,
                content,
                meta_data,
                1 - (embedding <=> CAST(:embedding AS vector)) AS similarity
            FROM document_vectors
            WHERE user_id = :user_id
              AND embedding IS NOT NULL
        """
        if strategy is not None:
            base_sql += "\n              AND meta_data->>'strategy' = :strategy"
        if level is not None:
            base_sql += "\n              AND meta_data->>'level' = :level"
        base_sql += "\n            ORDER BY embedding <=> CAST(:embedding AS vector)\n            LIMIT :limit"
        search_sql = text(base_sql)

        embedding = await self.embed_query(query, use_fine_tuned=use_fine_tuned_embeddings)
        params: dict[str, Any] = {
            "user_id": user_id,
            "embedding": _serialize_embedding(embedding),
            "limit": limit,
            "strategy": strategy,
            "level": level,
        }

        async with self.session_local() as session:
            result = await session.execute(search_sql, params)
            return [RetrievedDocument.from_mapping(_to_mapping(row)) for row in result.mappings().all()]

    async def search_with_reranking(
        self,
        *,
        user_id: str,
        query: str,
        limit: int = 5,
        candidate_limit: int = 20,
    ) -> list[RetrievedDocument]:
        """Run vector retrieval followed by cross-encoder reranking."""
        candidates = await self.vector_search(user_id=user_id, query=query, limit=candidate_limit)
        if not candidates:
            return []

        reranker = await self._get_reranker()
        pairs = [(query, candidate.content) for candidate in candidates]
        scores = await asyncio.to_thread(reranker.predict, pairs)

        rescored: list[RetrievedDocument] = []
        for candidate, score in zip(candidates, scores, strict=False):
            rescored.append(
                RetrievedDocument(
                    id=candidate.id,
                    document_id=candidate.document_id,
                    title=candidate.title,
                    content=candidate.content,
                    meta_data=candidate.meta_data,
                    similarity=float(score),
                )
            )
        return sorted(rescored, key=lambda item: item.similarity or 0.0, reverse=True)[:limit]

    async def search_with_query_expansion(
        self,
        *,
        user_id: str,
        query: str,
        variants: int = 3,
        per_query_limit: int = 3,
    ) -> list[RetrievedDocument]:
        """Expand a query and union the retrieved results."""
        expansion = await self.expand_query(query, variants=variants)
        return await self._search_across_queries(
            user_id=user_id,
            queries=expansion.variants,
            per_query_limit=per_query_limit,
        )

    async def search_with_multi_query(
        self,
        *,
        user_id: str,
        query: str,
        variants: int = 4,
        per_query_limit: int = 5,
    ) -> list[RetrievedDocument]:
        """Search over multiple query perspectives and deduplicate results."""
        expansion = await self.expand_query(query, variants=max(variants - 1, 1))
        return await self._search_across_queries(
            user_id=user_id,
            queries=expansion.variants,
            per_query_limit=per_query_limit,
        )

    async def _search_across_queries(
        self,
        *,
        user_id: str,
        queries: Sequence[str],
        per_query_limit: int,
    ) -> list[RetrievedDocument]:
        searches = await asyncio.gather(
            *[
                self.vector_search(user_id=user_id, query=query, limit=per_query_limit)
                for query in queries
            ]
        )

        deduped: dict[tuple[str, str], RetrievedDocument] = {}
        for search_result in searches:
            for document in search_result:
                key = (document.document_id, document.content)
                existing = deduped.get(key)
                if existing is None or (document.similarity or 0.0) > (existing.similarity or 0.0):
                    deduped[key] = document

        return sorted(deduped.values(), key=lambda item: item.similarity or 0.0, reverse=True)

    async def search_hierarchical(
        self,
        *,
        user_id: str,
        query: str,
        limit: int = 3,
    ) -> list[RetrievedDocument]:
        """Retrieve child chunks first, then return their parent contexts."""
        child_rows = await self.vector_search(
            user_id=user_id,
            query=query,
            limit=limit,
            strategy="hierarchical",
            level="child",
        )
        if not child_rows:
            return []

        hierarchy_ids = deduplicate_strings(
            [row.meta_data.get("hierarchy_id", "") for row in child_rows if row.meta_data.get("hierarchy_id")]
        )
        parent_sql = text(
            """
            SELECT id, document_id, title, content, meta_data
            FROM document_vectors
            WHERE user_id = :user_id
              AND meta_data->>'strategy' = 'hierarchical'
              AND meta_data->>'level' = 'parent'
              AND meta_data->>'hierarchy_id' IN :hierarchy_ids
            """
        ).bindparams(bindparam("hierarchy_ids", expanding=True))

        async with self.session_local() as session:
            result = await session.execute(
                parent_sql,
                {"user_id": user_id, "hierarchy_ids": hierarchy_ids},
            )
            parent_rows = [RetrievedDocument.from_mapping(_to_mapping(row)) for row in result.mappings().all()]

        by_hierarchy_id = {row.meta_data.get("hierarchy_id"): row for row in parent_rows}
        ordered: list[RetrievedDocument] = []
        for child_row in child_rows:
            hierarchy_id = child_row.meta_data.get("hierarchy_id")
            parent = by_hierarchy_id.get(hierarchy_id)
            if parent and parent not in ordered:
                ordered.append(parent)
        return ordered

    async def search_knowledge_graph(
        self,
        *,
        query: str,
        graph_searcher: AsyncGraphSearcher,
        top_k: int = 5,
    ) -> str:
        """Query an external graph retriever and answer from the returned context."""
        results = await graph_searcher(query, top_k)
        context = format_graph_results(results)
        if not context:
            return "No graph context found."
        return await self.answer_with_context(
            query,
            context=context,
            instructions="Answer the query using the graph context.",
        )

    async def run_agentic_rag(
        self,
        *,
        user_id: str,
        question: str,
        sql_runner: AsyncTextResolver | None = None,
        web_runner: AsyncTextResolver | None = None,
    ) -> AgenticRAGResult:
        """Route a question to the best available retrieval tool."""
        router_prompt = (
            "Choose one tool for the question: vector, sql, or web. "
            f"Question: {question}. Return only the tool name."
        )
        selected = (await ainvoke_text(router_prompt, model=self.fast_model)).strip().lower()

        if "sql" in selected and sql_runner is not None:
            context = await sql_runner(question)
            selected_tool = "sql"
        elif "web" in selected and web_runner is not None:
            context = await web_runner(question)
            selected_tool = "web"
        else:
            documents = await self.vector_search(user_id=user_id, query=question, limit=5)
            context = format_retrieved_documents(documents)
            selected_tool = "vector"

        logger.info("Agentic RAG selected tool", selected_tool=selected_tool)
        answer = await self.answer_with_context(question, context=context)
        return AgenticRAGResult(selected_tool=selected_tool, context=context, answer=answer)

    async def run_self_reflective_rag(
        self,
        *,
        user_id: str,
        query: str,
        limit: int = 5,
        relevance_threshold: float = 0.7,
    ) -> str:
        """Retrieve, grade, optionally refine the query, and verify the answer."""
        initial_documents = await self.vector_search(user_id=user_id, query=query, limit=limit)
        if not initial_documents:
            return "No relevant context found."

        relevant_documents: list[RetrievedDocument] = []
        for document in initial_documents:
            score = await self.grade_relevance(query, document.content)
            if score >= relevance_threshold:
                relevant_documents.append(document)

        quality = len(relevant_documents) / max(len(initial_documents), 1)
        active_query = query

        if quality < 0.5:
            refine_prompt = (
                "Improve this retrieval query based on the weak context.\n"
                f"Original query: {query}\n\nDocs:\n"
                f"{format_retrieved_documents(initial_documents)}"
            )
            active_query = (await ainvoke_text(refine_prompt, model=self.fast_model)).strip()
            relevant_documents = await self.vector_search(user_id=user_id, query=active_query, limit=limit)

        context = format_retrieved_documents(relevant_documents)
        answer = await self.answer_with_context(active_query, context=context)
        verify_prompt = (
            "Is this answer fully supported by the context? Reply only YES or NO.\n\n"
            f"Answer:\n{answer}\n\nContext:\n{context}"
        )
        verdict = (await ainvoke_text(verify_prompt, model=self.fast_model)).strip().upper()
        return answer if verdict.startswith("YES") else "Need more context"

    async def retrieve_full_document(
        self,
        *,
        user_id: str,
        document_id: str | None = None,
        title_query: str | None = None,
    ) -> str:
        """Retrieve the full stored content for a document grouping."""
        params: dict[str, Any] = {"user_id": user_id}

        if document_id:
            query_sql = text(
                """
                SELECT title, content, meta_data
                FROM document_vectors
                WHERE user_id = :user_id
                  AND document_id = :document_id
                ORDER BY id
                """
            )
            params["document_id"] = document_id
        elif title_query:
            query_sql = text(
                """
                SELECT title, content, meta_data
                FROM document_vectors
                WHERE user_id = :user_id
                  AND title ILIKE :title_query
                ORDER BY id
                """
            )
            params["title_query"] = f"%{title_query}%"
        else:
            raise ValueError("document_id or title_query is required")

        async with self.session_local() as session:
            result = await session.execute(query_sql, params)
            rows = result.mappings().all()

        if not rows:
            return "No matching document found."

        title = str(rows[0]["title"])
        body = "\n\n".join(str(row["content"]) for row in rows)
        return f"{title}\n\n{body}".strip()


__all__ = [
    "AgenticRAGResult",
    "LateChunk",
    "QueryExpansionResult",
    "RAGStrategyService",
    "RetrievedDocument",
    "cosine_similarity",
    "deduplicate_strings",
    "deserialize_metadata",
    "format_graph_results",
    "format_retrieved_documents",
    "late_chunk_text",
    "mean_pool_embeddings",
    "parse_query_variants",
    "prepare_training_data",
    "semantic_chunk_text",
    "serialize_metadata",
    "split_text",
]
