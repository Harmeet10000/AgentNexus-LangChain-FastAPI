from fastapi import HTTPException, Request
from typing import Any, Optional
import os
from enum import Enum

class ApplicationEnvironment(str, Enum):
    DEVELOPMENT = "development"
    PRODUCTION = "production"
    STAGING = "staging"

class APIException(HTTPException):
    """Custom exception for API errors"""
    def __init__(
        self,
        status_code: int,
        message: str,
        data: Any = None,
        name: str = "APIError"
    ):
        self.name = name
        self.message = message
        self.data = data
        super().__init__(status_code=status_code, detail=message)
