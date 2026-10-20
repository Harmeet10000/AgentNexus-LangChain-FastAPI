"""Construction seam for provider instances used by non-request workflows."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .razorpay import RazorpayBillingProvider

if TYPE_CHECKING:
    from .protocol import BillingProvider


def build_billing_provider() -> BillingProvider:
    """Build one provider adapter for a worker invocation.

    Jobs depend on this narrow factory rather than constructing a raw
    ``RazorpayClient``. Tests can inject an in-memory adapter at the job seam.
    """

    return RazorpayBillingProvider()
