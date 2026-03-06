"""Self-reflective RAG using iterative LangChain retrieval and grading."""

from __future__ import annotations

import psycopg2
from pgvector.psycopg2 import register_vector

from app.shared.langchain_layer.models import build_chat_model, build_embedding_model

_CONN = psycopg2.connect("dbname=rag_db")
register_vector(_CONN)
_EMBEDDINGS = build_embedding_model()
_CHAT = build_chat_model()


def _embed(text: str) -> list[float]:
    return _EMBEDDINGS.embed_query(text)


def _chunk_text(text: str, size: int = 500) -> list[str]:
    return [text[i : i + size] for i in range(0, len(text), size)]


def ingest_document(text: str) -> None:
    with _CONN.cursor() as cur:
        for chunk in _chunk_text(text):
            cur.execute(
                "INSERT INTO chunks (content, embedding) VALUES (%s, %s)",
                (chunk, _embed(chunk)),
            )
    _CONN.commit()


def _grade_relevance(query: str, doc: str) -> float:
    prompt = (
        "Rate relevance from 0 to 1. Return only a number.\n\n"
        f"Query: {query}\n\nDocument:\n{doc}"
    )
    raw = str(_CHAT.invoke(prompt).content).strip()
    try:
        return float(raw)
    except ValueError:
        return 0.0


def _retrieve(query: str, *, top_k: int = 5) -> list[str]:
    with _CONN.cursor() as cur:
        cur.execute(
            "SELECT content FROM chunks ORDER BY embedding <=> %s LIMIT %s",
            (_embed(query), top_k),
        )
        return [row[0] for row in cur.fetchall()]


def run_self_reflective_rag(query: str) -> str:
    docs = _retrieve(query)
    if not docs:
        return "No relevant context found."
    relevant_docs = [doc for doc in docs if _grade_relevance(query, doc) > 0.7]
    quality = len(relevant_docs) / max(len(docs), 1)
    active_query = query
    if quality < 0.5:
        refine_prompt = (
            "Improve this retrieval query based on the weak context.\n"
            f"Original query: {query}\n\nDocs:\n" + "\n\n".join(docs)
        )
        active_query = str(_CHAT.invoke(refine_prompt).content).strip()
        relevant_docs = _retrieve(active_query)
    context = "\n\n".join(relevant_docs)
    answer_prompt = (
        "Answer the question using only the context.\n"
        f"Question: {active_query}\n\nContext:\n{context}"
    )
    answer = str(_CHAT.invoke(answer_prompt).content)
    verify_prompt = (
        "Is this answer fully supported by the context? Reply only YES or NO.\n\n"
        f"Answer:\n{answer}\n\nContext:\n{context}"
    )
    verdict = str(_CHAT.invoke(verify_prompt).content).strip().upper()
    return answer if verdict.startswith("YES") else "Need more context"
