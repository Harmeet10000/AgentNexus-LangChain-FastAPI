"""DTOs for credit consumption operations."""

from pydantic import BaseModel, ConfigDict, Field


class CreditConsumptionResult(BaseModel):
    """Result of credit application to an invoice."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    credit_applied: int = Field(
        serialization_alias="creditApplied",
        description="Total credit applied in paisa",
    )
    credit_applied_rupees: float = Field(
        serialization_alias="creditAppliedRupees",
        description="Total credit applied in rupees (for convenience)",
    )
    cash_due: int = Field(
        serialization_alias="cashDue",
        description="Remaining amount to be charged in paisa",
    )
    cash_due_rupees: float = Field(
        serialization_alias="cashDueRupees",
        description="Remaining amount to be charged in rupees (for convenience)",
    )
    credits_consumed: list["ConsumedCredit"] = Field(serialization_alias="creditsConsumed")
    invoice_paid_in_full: bool = Field(
        serialization_alias="invoicePaidInFull",
        description="True if invoice is fully covered by credit",
    )


class ConsumedCredit(BaseModel):
    """Individual credit consumption record."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    credit_id: str = Field(serialization_alias="creditId")
    consumed_amount: int = Field(
        serialization_alias="consumedAmount",
        description="Amount consumed in paisa",
    )
    remaining_balance: int = Field(
        serialization_alias="remainingBalance",
        description="New remaining balance in paisa",
    )
