"""Response envelope for credit API endpoints."""

from typing import TypeVar

from app.utils.response_type import APIResponse

from .consumption_dto import ConsumedCredit, CreditConsumptionResult
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
    "ConsumedCredit",
    "ConsumptionRecord",
    "CreditBalanceApiResponse",
    "CreditBalanceResponse",
    "CreditConsumptionApiResponse",
    "CreditConsumptionResult",
    "CreditGrantApiResponse",
    "CreditGrantResponse",
    "CreditHistoryApiResponse",
    "CreditHistoryResponse",
    "CreditRecord",
]
