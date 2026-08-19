"""Response envelope for credit API endpoints."""

from typing import TypeVar

from app.utils.response_type import APIResponse

from .consumption_dto import CreditConsumptionResult, ConsumedCredit
from .credit_dto import (
    ConsumptionRecord,
    CreditBalanceResponse,
    CreditGrantResponse,
    CreditHistoryResponse,
    CreditRecord,
)

# Type variable for generic API responses
T = TypeVar("T")

# Type aliases for common API responses
CreditGrantApiResponse = APIResponse[CreditGrantResponse]
CreditBalanceApiResponse = APIResponse[CreditBalanceResponse]
CreditHistoryApiResponse = APIResponse[CreditHistoryResponse]
CreditConsumptionApiResponse = APIResponse[CreditConsumptionResult]

# Re-export for convenience
__all__ = [
    "APIResponse",
    "CreditGrantApiResponse",
    "CreditBalanceApiResponse",
    "CreditHistoryApiResponse",
    "CreditConsumptionApiResponse",
    "CreditGrantResponse",
    "CreditBalanceResponse",
    "CreditHistoryResponse",
    "CreditConsumptionResult",
    "CreditRecord",
    "ConsumptionRecord",
    "ConsumedCredit",
]
