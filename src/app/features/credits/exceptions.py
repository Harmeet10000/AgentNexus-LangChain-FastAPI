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


class CreditAmountMustBePositiveException(CreditException):
    """Raised when credit amount is not positive (minimum 1 paisa)."""

    def __init__(self, amount: int | float):
        super().__init__(
            status_code=422,
            detail=f"Credit amount must be positive (minimum 1 paisa), got {amount}",
            error_code="CREDIT_AMOUNT_MUST_BE_POSITIVE",
        )


class CreditInvalidDateRangeException(CreditException):
    """Raised when valid_from > valid_until."""

    def __init__(self):
        super().__init__(
            status_code=422,
            detail="valid_until cannot be earlier than valid_from",
            error_code="CREDIT_INVALID_DATE_RANGE",
        )


class CreditMetadataMissingException(CreditException):
    """Raised when ADMIN_GRANT is missing admin_user_id in metadata."""

    def __init__(self):
        super().__init__(
            status_code=422,
            detail="ADMIN_GRANT requires admin_user_id in metadata",
            error_code="CREDIT_METADATA_MISSING",
        )


class CreditNotFoundException(CreditException):
    """Raised when a credit record is not found."""

    def __init__(self, credit_id: str):
        super().__init__(
            status_code=404,
            detail=f"Credit not found: {credit_id}",
            error_code="CREDIT_NOT_FOUND",
        )


class CreditInsufficientBalanceException(CreditException):
    """Raised when user has insufficient credit balance."""

    def __init__(self, user_id: str, required: int, available: int):
        super().__init__(
            status_code=400,
            detail=f"Insufficient credits. Required: {required}, Available: {available}",
            error_code="CREDIT_INSUFFICIENT_BALANCE",
            data={"user_id": user_id, "required": required, "available": available},
        )


class CreditExpiredException(CreditException):
    """Raised when attempting to consume an expired credit."""

    def __init__(self, credit_id: str):
        super().__init__(
            status_code=400,
            detail=f"Credit has expired: {credit_id}",
            error_code="CREDIT_EXPIRED",
        )


class CreditAlreadyConsumedException(CreditException):
    """Raised when attempting to consume a fully consumed credit."""

    def __init__(self, credit_id: str):
        super().__init__(
            status_code=400,
            detail=f"Credit has been fully consumed: {credit_id}",
            error_code="CREDIT_ALREADY_CONSUMED",
        )


class CreditTransactionRollbackException(CreditException):
    """Raised when a credit transaction is rolled back due to payment failure."""

    def __init__(self, invoice_id: str):
        super().__init__(
            status_code=500,
            detail=f"Transaction rolled back due to payment failure for invoice: {invoice_id}",
            error_code="CREDIT_TRANSACTION_ROLLBACK",
        )
