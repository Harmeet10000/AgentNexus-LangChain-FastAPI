from enum import Enum


class Environment(str, Enum):
    """Application environment."""

    DEVELOPMENT = "development"
    PRODUCTION = "production"
    TESTING = "testing"


# Error messages
SOMETHING_WENT_WRONG = "Something went wrong"
INTERNAL_SERVER_ERROR = "Internal server error"
VALIDATION_ERROR = "Validation error"
NOT_FOUND = "Resource not found"
UNAUTHORIZED = "Unauthorized"
FORBIDDEN = "Forbidden"
