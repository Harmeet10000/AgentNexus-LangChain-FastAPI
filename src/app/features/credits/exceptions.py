"""Credit-specific exceptions."""

from typing import Any

from app.utils.exceptions import APIException


class CreditException(APIException):
    """Base exception for credit-related errors."""

    def __init__(
        self,
        detail: str,
        error_code: str,
        status_code: int = 400,
        data: Any = None,
        headers: dict[str, str] | None = None,
    ):
        super().__init__(
            status_code=status_code,
            detail=detail,
            error_code=error_code,
            data=data,
            headers=headers,
        )


class InsufficientCreditsException(CreditException):
    """Raised when user has insufficient credits."""

    def __init__(
        self,
        user_id: str,  # noqa: ARG002
        required: int,
        available: int,
    ):
        super().__init__(
            status_code=403,
            detail=f"Insufficient credits. Required: {required}, Available: {available}",
            error_code="INSUFFICIENT_CREDITS",
        )


class InvalidCreditAmountException(CreditException):
    """Raised when credit amount is invalid."""

    def __init__(self, amount: int | float):
        super().__init__(
            status_code=400,
            detail=f"Invalid credit amount: {amount}",
            error_code="INVALID_CREDIT_AMOUNT",
        )


class CreditTransactionNotFoundException(CreditException):
    """Raised when a credit transaction is not found."""

    def __init__(self, transaction_id: str):
        super().__init__(
            status_code=404,
            detail=f"Credit transaction not found: {transaction_id}",
            error_code="CREDIT_TRANSACTION_NOT_FOUND",
        )


class CreditLimitExceededException(CreditException):
    """Raised when credit limit is exceeded."""

    def __init__(
        self,
        user_id: str,  # noqa: ARG002
        limit: int,
        attempted: int,
    ):
        super().__init__(
            status_code=400,
            detail=f"Credit limit exceeded. Limit: {limit}, Attempted: {attempted}",
            error_code="CREDIT_LIMIT_EXCEEDED",
        )
