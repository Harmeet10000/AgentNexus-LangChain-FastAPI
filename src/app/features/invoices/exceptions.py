"""Invoice feature typed exceptions."""

from __future__ import annotations

from app.utils import ValidationException


class InvoiceGenerationException(ValidationException):
    """Invoice generation failed validation or tax-consistency checks."""

    def __init__(self, detail: str, data: dict[str, object] | None = None) -> None:
        super().__init__(detail=detail, data=data)
