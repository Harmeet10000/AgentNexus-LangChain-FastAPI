"""Typed config for Cognee setup — credential-free summary."""

from __future__ import annotations

from pydantic import BaseModel, Field


class CogneeSetupConfig(BaseModel):
    """Credential-free summary of Cognee configuration (no secrets)."""

    model_config = {"frozen": True}

    llm_model: str = Field(description="LLM model name")
    embedding_model: str = Field(description="Embedding model name")
    embedding_dimension: int = Field(description="Embedding dimension")
    neo4j_uri: str = Field(description="Neo4j URI (without credentials)")
    postgres_host: str = Field(description="Postgres host")
    postgres_database: str = Field(description="Postgres database")
    vector_provider: str = Field(description="Vector provider")
    schema_name: str = Field(description="Cognee DB schema")
    access_control_enabled: bool = Field(description="Whether access control is enabled")
