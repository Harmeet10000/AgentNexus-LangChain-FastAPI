"""Gemini embedding helpers for search."""

from __future__ import annotations

from typing import TYPE_CHECKING

from langchain_google_genai import GoogleGenerativeAIEmbeddings

from app.config import get_settings

if TYPE_CHECKING:
    from app.config import Settings


def build_embedding_client() -> GoogleGenerativeAIEmbeddings:
    """Construct the shared Gemini embedding client used by search."""
    settings: Settings = get_settings()
    return GoogleGenerativeAIEmbeddings(
        model=settings.GEMINI_EMBEDDING_MODEL,
        api_key=settings.GEMINI_API_KEY,
        # The width this client *produces*, which must match the width the columns
        # *store*. A3's proof greps only for a literal width inside `Vector(...)`,
        # so this line would have survived it — and a producer pinned to a literal
        # against a configurable column turns a one-line config change into a
        # psycopg insert error at ingestion time instead of a validation error at
        # startup.
        output_dimensionality=settings.EMBEDDING_DIMENSION,
    )
