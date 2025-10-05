"""Shared enums and constants."""

from enum import Enum


class Environment(str, Enum):
    """Application environments."""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    TESTING = "testing"


class DocumentStatus(str, Enum):
    """Document processing status."""
    PENDING = "pending"
    PROCESSING = "processing"
    PROCESSED = "processed"
    FAILED = "failed"


class MessageRole(str, Enum):
    """Chat message roles."""
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class ModelProvider(str, Enum):
    """AI model providers."""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"
    COHERE = "cohere"


class VectorStore(str, Enum):
    """Vector store providers."""
    PINECONE = "pinecone"
    CHROMA = "chroma"
    WEAVIATE = "weaviate"
    QDRANT = "qdrant"


# HTTP Status Messages
HTTP_STATUS_MESSAGES = {
    200: "Success",
    201: "Created successfully",
    400: "Bad request",
    401: "Unauthorized",
    403: "Forbidden",
    404: "Not found",
    422: "Validation error",
    500: "Internal server error"
}

# Error Messages
ERROR_MESSAGES = {
    "SOMETHING_WENT_WRONG": "Something went wrong. Please try again.",
    "INVALID_CREDENTIALS": "Invalid email or password",
    "USER_NOT_FOUND": "User not found",
    "USER_ALREADY_EXISTS": "User with this email already exists",
    "UNAUTHORIZED_ACCESS": "Unauthorized access",
    "FORBIDDEN_ACCESS": "Access forbidden",
    "VALIDATION_FAILED": "Validation failed",
    "DOCUMENT_NOT_FOUND": "Document not found",
    "VECTOR_STORE_ERROR": "Vector store operation failed",
    "MODEL_ERROR": "AI model operation failed"
}