"""Gemini embedding helpers for search."""

from __future__ import annotations

from langchain_google_genai import GoogleGenerativeAIEmbeddings

from app.config import get_settings


def build_embedding_client() -> GoogleGenerativeAIEmbeddings:
    """Construct the shared Gemini embedding client used by search."""
    settings = get_settings()
    return GoogleGenerativeAIEmbeddings(
        model=settings.GEMINI_EMBEDDING_MODEL,
        api_key=settings.GEMINI_API_KEY,
        output_dimensionality=768,
    )
