"""Context-aware chunking using embedding similarity + LangChain retrieval."""

from __future__ import annotations

import math

import psycopg2
from pgvector.psycopg2 import register_vector

from app.shared.langchain_layer.models import build_chat_model, build_embedding_model

_CONN = psycopg2.connect("dbname=rag_db")
register_vector(_CONN)
_EMBEDDINGS = build_embedding_model()
_CHAT = build_chat_model()


def _embed(text: str) -> list[float]:
    return _EMBEDDINGS.embed_query(text)


def _cosine_similarity(v1: list[float], v2: list[float]) -> float:
    dot = sum(a * b for a, b in zip(v1, v2, strict=False))
    n1 = math.sqrt(sum(a * a for a in v1))
    n2 = math.sqrt(sum(b * b for b in v2))
    if n1 == 0 or n2 == 0:
        return 0.0
    return dot / (n1 * n2)


def semantic_chunk(text: str, *, similarity_threshold: float = 0.8) -> list[str]:
    sentences = [part.strip() for part in text.split(". ") if part.strip()]
    if not sentences:
        return []
    sentence_embeddings = [_embed(sentence) for sentence in sentences]
    chunks: list[str] = []
    current_chunk = [sentences[0]]
    for idx in range(len(sentences) - 1):
        similarity = _cosine_similarity(sentence_embeddings[idx], sentence_embeddings[idx + 1])
        if similarity > similarity_threshold:
            current_chunk.append(sentences[idx + 1])
            continue
        chunks.append(". ".join(current_chunk))
        current_chunk = [sentences[idx + 1]]
    chunks.append(". ".join(current_chunk))
    return chunks


def ingest_document(text: str) -> None:
    chunks = semantic_chunk(text)
    with _CONN.cursor() as cur:
        for chunk in chunks:
            cur.execute(
                "INSERT INTO chunks (content, embedding) VALUES (%s, %s)",
                (chunk, _embed(chunk)),
            )
    _CONN.commit()


def search_knowledge_base(query: str, *, top_k: int = 3) -> str:
    with _CONN.cursor() as cur:
        cur.execute(
            "SELECT content FROM chunks ORDER BY embedding <=> %s LIMIT %s",
            (_embed(query), top_k),
        )
        context = "\n\n".join(row[0] for row in cur.fetchall())
    return str(_CHAT.invoke(f"Question: {query}\n\nContext:\n{context}").content)
