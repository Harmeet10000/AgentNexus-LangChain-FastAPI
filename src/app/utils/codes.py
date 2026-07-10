"""Centralized error codes shared by API exceptions and internal Result errors.

Add codes here only when they map to a typed exception class.
For truly one-off codes in routes, use inline strings with a `# noqa: S` comment.
"""

from enum import StrEnum


class ErrorCode(StrEnum):
    VALIDATION_ERROR = "VALIDATION_ERROR"
    NOT_FOUND = "NOT_FOUND"
    UNAUTHORIZED = "UNAUTHORIZED"
    FORBIDDEN = "FORBIDDEN"
    CONFLICT = "CONFLICT"
    TOO_MANY_REQUESTS = "TOO_MANY_REQUESTS"
    SERVICE_UNAVAILABLE = "SERVICE_UNAVAILABLE"
    DATABASE_ERROR = "DATABASE_ERROR"
    EXTERNAL_SERVICE_ERROR = "EXTERNAL_SERVICE_ERROR"
    INVALID_TOKEN = "INVALID_TOKEN"  # noqa: S105 — error code string, not a password
    TOKEN_EXPIRED = "TOKEN_EXPIRED"  # noqa: S105 — error code string, not a password
    REFRESH_TOKEN_INVALID = "REFRESH_TOKEN_INVALID"  # noqa: S105 — error code string, not a password
    INFRASTRUCTURE_ERROR = "INFRASTRUCTURE_ERROR"
    INTERNAL_SERVER_ERROR = "INTERNAL_SERVER_ERROR"

    # Repository-level codes (used in AppError payloads from repositories)
    DOCUMENT_NOT_FOUND = "DOCUMENT_NOT_FOUND"
    STATUS_NOT_FOUND = "STATUS_NOT_FOUND"
    SEARCH_DOCUMENT_NOT_FOUND = "SEARCH_DOCUMENT_NOT_FOUND"
    USER_NOT_FOUND = "USER_NOT_FOUND"


class Environment(StrEnum):
    """Application environment."""

    DEVELOPMENT = "development"
    PRODUCTION = "production"
    TESTING = "testing"
