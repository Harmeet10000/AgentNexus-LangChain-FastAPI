"""Hierarchical RAG - child retrieval with parent-context synthesis in LangChain."""

from __future__ import annotations

import json

import psycopg2
from pgvector.psycopg2 import register_vector

from app.shared.langchain_layer.models import build_chat_model, build_embedding_model

_CONN = psycopg2.connect("dbname=rag_db")
register_vector(_CONN)
_EMBEDDINGS = build_embedding_model()
_CHAT = build_chat_model()


def _embed(text: str) -> list[float]:
    return _EMBEDDINGS.embed_query(text)


def _chunk_text(text: str, size: int) -> list[str]:
    return [text[i : i + size] for i in range(0, len(text), size)]


def ingest_document(text: str, doc_title: str) -> None:
    parent_chunks = _chunk_text(text, 2000)
    with _CONN.cursor() as cur:
        for parent_id, parent in enumerate(parent_chunks):
            metadata = json.dumps({"heading": f"{doc_title} - Section {parent_id}", "type": "detail"})
            cur.execute(
                "INSERT INTO parent_chunks (id, content, metadata) VALUES (%s, %s, %s)",
                (parent_id, parent, metadata),
            )
            for child in _chunk_text(parent, 500):
                cur.execute(
                    "INSERT INTO child_chunks (content, embedding, parent_id) VALUES (%s, %s, %s)",
                    (child, _embed(child), parent_id),
                )
    _CONN.commit()


def search_knowledge_base(query: str, *, top_k: int = 3) -> str:
    with _CONN.cursor() as cur:
        cur.execute(
            """
            SELECT p.content, p.metadata
            FROM child_chunks c
            JOIN parent_chunks p ON c.parent_id = p.id
            ORDER BY c.embedding <=> %s
            LIMIT %s
            """,
            (_embed(query), top_k),
        )
        rows = cur.fetchall()
    context = "\n\n".join(
        f"[{json.loads(metadata)['heading']}]\n{content}" for content, metadata in rows
    )
    prompt = f"Answer using hierarchical context.\n\nQuestion: {query}\n\nContext:\n{context}"
    return str(_CHAT.invoke(prompt).content)
